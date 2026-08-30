# MLflow model registry + promotion gate — integration plan

## Context

Signal Score's architecture (`docs/signalscore-design.md`, `docs/signalscore-architecture.mermaid`) and CLAUDE.md already specify a full MLflow-backed model registry and promotion-gate design — but none of it is built. `src/signalscore/registry.py`, `evaluation/`, `serving/` are docstring-only stubs; `mlflow` isn't even a dependency yet. Meanwhile `training/train_baseline.py` is a real, tested sklearn baseline that already produces the exact metrics (`pr_auc_macro`, `minority_f1`, `brier_macro`, `per_class_f1`) the gate needs, and the frozen held-out set the gate must compare against already exists and is DVC-tracked (`data/processed/kubernetes-kubernetes/eval_set_v1.jsonl`, frozen at git tag `eval-set-v1`). This is a greenfield build on top of real, reusable pieces — not a redesign.

Two things needed resolving before implementation: (1) `docs/design-patterns-guide.md` internally contradicts itself on whether the promotion gate is a Template Method or a Chain of Responsibility; (2) the guide's `ModelRegistry` sketch uses MLflow's deprecated Stages API. Both are resolved below. This plan was produced by two senior-engineer design passes (core architecture, then an MLOps red-team review), revised once more after direct review comments on the gate pattern and config handling, and cross-checked against the actual `train_baseline.py` source and current MLflow docs (filesystem tracking/registry backend deprecation as of Feb 2026, confirmed live: https://github.com/mlflow/mlflow/issues/18534).

Goal: implement the registry, the promotion gate, minimal training/serving integration, and the tests/CI wiring needed to make CLAUDE.md's promotion-gate rule real — while staying v0-appropriately minimal (one experiment logged so far, zero production traffic).

## Design decisions

**1. `registry.py` — `ModelRegistry` class (Adapter, per design-patterns-guide.md's own trigger: stateful client, 3+ callers).** Goes beyond the guide's literal skeleton for two reasons: that skeleton uses MLflow's deprecated Stages enum, and (revised after further review) plain tags aren't the right primitive for "the current production version" — MLflow's **alias** mechanism (`set_registered_model_alias`/`get_model_version_by_alias`) is the purpose-built, structurally-enforced single pointer for exactly this ([MLflow docs](https://mlflow.org/docs/latest/ml/model-registry/workflow/)). Aliases hold the *authoritative* current-production pointer; a `stage` tag (`staging`/`production`/`rejected`/`archived`) stays alongside it purely as a human-readable label for the UI/audit trail, and `prior_production` (`"true"` on the most recently demoted version) still lets `rollback()` find what to restore.

This is a real correctness improvement, not just idiom-matching: MLflow enforces that only one version can ever hold a given alias at a time, so the earlier "raise loudly if `get_production()` ever finds >1 version tagged production" defensive check is now unnecessary — that failure mode is structurally impossible rather than merely detected. The remaining non-atomicity in `promote()` (see below) only affects the *descriptive* tags, never which version is actually live.

```python
DEFAULT_MODEL_NAME = "signalscore-priority"
PRODUCTION_ALIAS = "production"


@dataclass(frozen=True)
class ModelVersionInfo:
    version: str
    run_id: str
    stage: str  # sourced from OUR `stage` tag, never MLflow's own deprecated
    # `ModelVersion.current_stage` attribute — the two are unrelated
    # and mixing them up is an easy implementation-time mistake.
    source: str


class ModelRegistry:
    def __init__(
        self, client: MlflowClient | None = None, model_name: str = DEFAULT_MODEL_NAME
    ) -> None: ...
    def register_candidate(self, run_id: str, artifact_path: str = "model") -> ModelVersionInfo: ...
    def get_production(self) -> ModelVersionInfo | None:
        """Reads the @production alias via get_model_version_by_alias(); catches
        MlflowException (RESOURCE_DOES_NOT_EXIST) and returns None if the alias
        has never been set. No 'what if >1 tagged production' case exists — the
        alias makes that structurally impossible."""

    def get_version(self, version: str) -> ModelVersionInfo: ...
    def resolve_artifact_path(self, info: ModelVersionInfo) -> Path:
        """Downloads/locates the artifact dir. Does NOT load it — callers
        joblib.load() themselves. Keeps registry.py artifact-format-agnostic."""

    def promote(
        self, version: str, gate_result: str = "pass", eval_set_tag: str | None = None
    ) -> None:
        """Clears prior_production=true from every OTHER version first (see bug
        note below), reads the current @production alias holder (if any), tags
        it stage=archived + prior_production=true for history/rollback, then
        reassigns the @production alias to `version` and tags it stage=production
        + gate_result=<gate_result> (structured, not just EXPERIMENTS.md prose —
        needed later for the design doc's "N consecutive losing retrains"
        escalation rule) + eval_set_tag=<eval_set_tag> when given. The alias
        reassignment is the one call that actually matters for correctness and
        is a single atomic pointer move at the MLflow API level; the surrounding
        tag writes are bookkeeping only — if one of them fails, the worst case
        is a stale label, never two versions serving as production
        simultaneously."""

    def reject(self, version: str, reason: str, eval_set_tag: str | None = None) -> None:
        """Tags stage=rejected, rejection_reason=<reason>, gate_result=fail,
        eval_set_tag=<eval_set_tag> when given — same structured-tag reasoning
        as promote()."""

    def rollback(self) -> str:
        """Finds the prior_production=true version and calls promote() on it
        — reuses the same invariant rather than duplicating it. This is the
        design doc's 'one-line tag revert', called via a single governed
        `ModelRegistry().rollback()` call (no bespoke CLI needed — rare,
        human-triggered action)."""
```

**Bug caught in review, fixed here:** `prior_production` as originally sketched had no uniqueness guarantee — unlike the alias, MLflow doesn't enforce "only one version can hold this tag." After `promote(v1) → promote(v2) → promote(v3)`, both v1 and v2 could end up tagged `prior_production=true`, making `rollback()`'s "find the tagged version" query ambiguous. Two promotions (the plan's own rollback test) wouldn't have caught this. Fix: `promote()` must clear `prior_production` from every version that currently holds it (via `search_model_versions(f"name='{model_name}'")`, filtering on the tag) *before* setting it on the version it's about to demote — cheap at this scale (a handful of versions), and guarantees at most one version ever holds the tag.

**`eval_set_tag` — added per direct request.** Every production model gets tagged with the frozen eval-set tag (`FROZEN_EVAL_TAG`, currently `"eval-set-v1"`, defined once in `contract.py` and threaded through from `run_gate.py`) it actually cleared the gate against. This is lineage, not decoration: `docs/signalscore-design.md` treats the frozen eval set as versioned (§2's "test is frozen... never reshuffled" implies a future `eval-set-v2` is possible once enough time passes for a new held-out window), and without this tag a production model browsed later in the MLflow UI would have no record of which eval baseline it was validated against — the same gap the ML-engineer review flagged more generally as missing lineage metadata. Rejected candidates get the same tag for the same reason (§1's `reject()` signature above).

**2. Promotion gate — Chain of Responsibility, with a fixed (non-configurable) chain.** `docs/design-patterns-guide.md`'s Hard Warning #4 names the promotion gate specifically: "Prefer Chain of Responsibility over Template Method for the promotion gate... where a step must hard-fail rather than be skippable. Template Method's premise is that a subclass can override or skip a step." That's the authoritative, specific directive — it beats the more generic §2 Template Method worked example, which itself points back to this same warning.

The reason a plain-function resolution was considered and rejected: §5's own trigger describes CoR as "independent... steps... addable/removable without touching the others" (its worked example is serving's `/score` validation, genuinely freely-recomposable). That shape is wrong for a gate whose steps are ordered and dependent, and where "removable" is exactly what must not happen. The resolution is to use CoR's *mechanism* (each handler decides whether to hard-fail or hand off to the next) without its *configurability* — the chain is wired once, in one function, and is not a parameter any caller can shorten or reorder:

```python
@dataclass(frozen=True)
class GateContext:
    """Everything a GateStep needs; pinned down here per review (was previously
    referenced but undefined, which would have forced an undocumented judgment
    call at implementation time)."""

    candidate: BaselineArtifact
    production: BaselineArtifact | None  # None only in the first-ever-promotion case
    eval_rows: list[FeatureRow]


class GateStep(ABC):
    name: ClassVar[str]

    def __init__(self, next_step: "GateStep | None" = None) -> None:
        self._next = next_step

    @abstractmethod
    def _check(self, ctx: GateContext) -> str | None: ...  # None = pass, else the failure reason

    def handle(self, ctx: GateContext) -> GateResult:
        reason = self._check(ctx)
        if reason is not None:
            return GateResult(passed=False, failed_stage=self.name, detail=reason)
        if self._next is None:
            return GateResult(passed=True, failed_stage=None, detail=None)
        return self._next.handle(ctx)


class UnitTestStep(GateStep):
    name = "unit_tests"

    def _check(self, ctx: GateContext) -> str | None: ...  # subprocess pytest, see below


class ContractStep(GateStep):
    name = "contract"

    def _check(
        self, ctx: GateContext
    ) -> str | None: ...  # delegates to contract.check_model_contract


class MarginStep(GateStep):
    name = "margin"

    def _check(
        self, ctx: GateContext
    ) -> (
        str | None
    ): ...  # delegates to margin.evaluate_margin; None (auto-pass) if ctx.production is None


def build_promotion_gate() -> GateStep:
    """The one place the chain is wired. Fixed order, not caller-configurable —
    unlike serving's /score ValidationChain (design-patterns-guide.md §5), which
    IS meant to be freely recomposed. No code path here constructs a shorter or
    reordered chain, which is what actually satisfies Hard Warning #4's concern
    (steps must not be skippable) — a caller-supplied `list[GateStep]` would
    reintroduce exactly that risk via omission, so this stays a plain function
    returning a hard-coded chain, never a public constructor parameter."""
    return UnitTestStep(next_step=ContractStep(next_step=MarginStep()))
```

```
src/signalscore/evaluation/
    __init__.py   (existing docstring, extend with module list)
    contract.py   # step 2 check logic: schema, valid probabilities, no NaNs, latency bound — all 4 checks per the existing docstring
    margin.py     # step 3 check logic: head-to-head comparison on the frozen eval set
    gate.py       # GateStep/UnitTestStep/ContractStep/MarginStep + build_promotion_gate()
    run_gate.py   # CLI: loads candidate/production via ModelRegistry, calls build_promotion_gate().handle(ctx), promotes/rejects, logs EXPERIMENTS.md row
```

- `UnitTestStep._check()` uses `subprocess.run([sys.executable, "-m", "pytest", "tests/features/", ...], timeout=...)` — scoped to `tests/features/` specifically (not the whole suite) with an explicit timeout, since this is the repo's first use of `subprocess` and an unscoped call risks self-recursion/hangs.
- **Eval-set integrity check (revised after further review — supersedes a row-count assertion).** A row-count check is a weak proxy — it catches truncation but not same-length corruption. The eval set is already DVC-tracked and frozen at git tag `eval-set-v1` (confirmed live: `data/processed/kubernetes-kubernetes.dvc` is a directory-level DVC pointer, and `dvc diff eval-set-v1 --targets data/processed/kubernetes-kubernetes/eval_set_v1.jsonl --json` returns `{}` when the working file exactly matches what's frozen at that tag, or reports `added`/`modified`/`deleted` keys otherwise) — so `check_model_contract()` (step 2) verifies against DVC's actual content hash instead of reimplementing a weaker one:
  ```python
  FROZEN_EVAL_TAG = "eval-set-v1"


  def _parse_dvc_diff_output(raw_json: str) -> str | None:
      """None = clean (matches the frozen tag exactly); otherwise a message
      naming what changed. Pure function — trivial to unit-test with literal
      JSON fixtures, no subprocess involved."""
      diff = json.loads(raw_json)
      if not diff:
          return None
      changed = [k for k in ("added", "modified", "deleted", "renamed") if diff.get(k)]
      return f"eval set does not match frozen tag {FROZEN_EVAL_TAG}: {', '.join(changed)}"


  def verify_eval_set_integrity(eval_set_path: Path, frozen_tag: str = FROZEN_EVAL_TAG) -> str | None:
      """Thin subprocess wrapper. Catches CalledProcessError/FileNotFoundError/
      TimeoutExpired and returns a failure string rather than raising — a
      missing dvc binary or a missing tag must hard-fail the gate, never pass
      silently."""
      try:
          result = subprocess.run(
              ["dvc", "diff", frozen_tag, "--targets", str(eval_set_path), "--json"],
              capture_output=True,
              text=True,
              timeout=30,
              check=True,
          )
      except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
          return f"could not verify eval set against frozen tag {frozen_tag}: {e}"
      return _parse_dvc_diff_output(result.stdout)
  ```
  `dvc` is already a `dev`-group dependency (not `ml`/`serving`) — this check only ever runs in the evaluation/gate stage, which always executes in a dev/CI context per `make check`; `serving`'s runtime never needs `dvc` at all, so this doesn't change §7's dependency placement. Testing: `_parse_dvc_diff_output` gets ordinary literal-JSON-fixture unit tests; `verify_eval_set_integrity` gets one real integration test building a tiny `tmp_path` git+dvc repo (init, add a file, commit, tag) and asserting clean-vs-modified — matching the repo's no-mocking convention (real subprocess calls) without needing the actual multi-megabyte eval set in the test suite.
- **Margin (revised after review — supersedes an earlier fixed-delta-only MVP):** the original "no resampling infra needed" justification for skipping bootstrap CI was wrong — a percentile bootstrap needs no model refitting, only resampling the row indices of predictions *already computed* by `compute_metrics()`-shaped calls, so it's ~20 lines and zero new dependencies. With only ~442 minority-class rows in the 2,380-row eval set, a bare point-estimate delta can pass or fail on sampling noise, which is a real problem for a check making live production-swap decisions. Final design mixes statistical rigor where it matters with simple guards where a fixed floor is more appropriate:
  - **PR-AUC improvement — bootstrap CI, not a fixed delta.** Resample eval-row indices with replacement (`N_BOOTSTRAP = 1000`, seeded), recompute `pr_auc_macro(candidate) - pr_auc_macro(production)` per resample, and require the 95% CI lower bound of that distribution to exceed 0 — i.e., the improvement must be significant at the eval set's actual noise level, not just a positive point estimate.
  - **Minority F1 — plain non-regression floor, not CI-based:** `candidate.minority_f1 >= production.minority_f1` (point estimate). This is a downside guard ("must never get worse on the safety-critical class"), not an upside claim needing a significance test — keeping it simple is deliberate, not an oversight.
  - **Brier score — fixed absolute cap**, unchanged: `MAX_BRIER_REGRESSION = 0.01`, same reasoning (downside guard on calibration slip).
  - Both artifacts are scored with their own fitted vectorizers via `build_feature_matrix`/`FittedVectorizers` (verified against `train_baseline.py` — this is exactly how it already works, so the comparison is apples-to-apples).
- **Edge case — first-ever promotion** (no current production exists): run pre-checks (steps 1–2) as a safety net, **then apply an absolute quality floor instead of a pure pass-through** — `candidate.pr_auc_macro > FIRST_PROMOTION_MIN_PR_AUC` (a hardcoded reference derived from the one real baseline run already logged, PR-AUC 0.5653, giving the very first promotion an actual bar to clear instead of "cleared unit tests + I/O contract" being sufficient on its own). Distinct from the **empty-registry-at-serving-startup** case (§5) — flagged separately, don't conflate.
- **Pass/fail handling:** pre-check failure (steps 1–2) → stays `staging`, retry-worthy, **and still gets an `EXPERIMENTS.md` row** via `format_experiments_row(gate_result="pre-check failed: <stage>")` — the repo's own invariant is "every real training run gets a row," not just margin losses. Margin failure (step 3) → `registry.reject(version, reason, eval_set_tag=FROZEN_EVAL_TAG)`, also logged, plus the structured `gate_result`/`rejection_reason`/`eval_set_tag` tags per §1's revised `reject()`. A passing gate → `registry.promote(version, gate_result="pass", eval_set_tag=FROZEN_EVAL_TAG)`.

**3. Local MLflow infra — SQLite tracking URI, no server process, no docker-compose.**
`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. Confirmed current (Aug 2026): MLflow's filesystem `FileStore` backend is deprecated as of Feb 2026 for both tracking and registry use; `sqlite:///mlflow.db` works directly with `MlflowClient()`, no `mlflow server` needed. Artifacts still land on local disk under `./mlruns/.../artifacts/` (unaffected by the metadata-store deprecation; `.gitignore` already has `mlruns/`). No docker-compose: no persistent CI volume, no multi-user need yet, and CLAUDE.md already bans this class of premature infra ("No Dockerfile until serving/ has real code"). `.env.example` gets `MLFLOW_TRACKING_URI=sqlite:///mlflow.db` replacing its placeholder comment; `.gitignore` gets `mlflow.db` added. **Degradation note:** if the deprecation claim turns out wrong on closer inspection during implementation, the only change needed is the default URI string — registry/gate code only ever calls `MlflowClient` tag/search methods, never raw SQL. Frozen held-out set: already solved, use `data/processed/kubernetes-kubernetes/eval_set_v1.jsonl` directly, no new DVC work.

**4. Training integration — additive, confined to `train_baseline.py::main()`.** `run_training_pipeline()` stays untouched (existing tests unaffected).

```python
mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)
with mlflow.start_run(run_name=f"{MODEL_VERSION}-{datetime.now(UTC):%Y%m%dT%H%M%S}") as run:
    mlflow.log_params(
        {
            "model_version": MODEL_VERSION,
            "word_ngram_range": "1-2",
            "word_min_df": 3,
            "char_ngram_range": "3-5",
            "char_min_df": 5,
            "char_max_features": 300_000,
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            # reproducibility: which library versions produced this artifact --
            # joblib.load() is sensitive to exact minor-version mismatches, and
            # mlflow.log_artifact() on a raw joblib dump (not mlflow.sklearn.log_model)
            # skips MLflow's automatic environment capture, so this is the only record.
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            # data snapshot identity: repo_dir_name alone (already in EXPERIMENTS.md)
            # doesn't say WHICH pull of that repo produced this candidate.
            "train_row_count": len(train_rows),
            "val_row_count": len(val_rows),
        }
    )
    metrics = run_training_pipeline(args.train, args.val, model_out, metrics_out)
    mlflow.log_metrics({k: v for k, v in metrics.items() if k != "per_class_f1"})
    mlflow.log_dict(metrics["per_class_f1"], "per_class_f1.json")
    mlflow.log_artifact(str(model_out), artifact_path="model")
    candidate = ModelRegistry().register_candidate(run_id=run.info.run_id)
```
`train_row_count`/`val_row_count` need `train_rows`/`val_rows` visible in `main()` — either `run_training_pipeline()` returns them alongside `metrics`, or `main()` calls `load_feature_rows()` once more itself (cheap; these are already-parsed JSONL, not a re-fetch) rather than changing `run_training_pipeline()`'s tested return signature. Prefer the second: keeps §4's "run_training_pipeline() stays untouched" rule intact.

The hyperparameter params are literal copies of the hardcoded values already inside `fit_vectorizers`/`train_classifier`, not threaded through as a config object — deliberate at v0, since `run_training_pipeline()`'s signature and its existing tests stay untouched. This does mean the two hardcoded value sets must be kept in sync by hand until they actually start varying. The moment a hyperparameter sweep or a second model config becomes real work (not hypothetical), that's the trigger to extract a small config dataclass threaded through both `fit_vectorizers`/`train_classifier` and `main()`'s `log_params` call, so there's one source of truth instead of two — not before, per the guide's own YAGNI stance on speculative extension points.

Hoist `MODEL_ARTIFACT_FILENAME = "model.joblib"` as a module constant (currently an inline literal at `train_baseline.py:244`), reused by registry/evaluation/serving so the filename lives in one place. New small test: `main()` against a `tmp_path`-scoped SQLite URI registers exactly one `staging` version with its params and metrics logged.

**5. Serving MVP — startup-time load, not per-request.**
```python
@app.on_event("startup")
def load_production_model() -> None:
    info = ModelRegistry().get_production()
    if info is None:
        raise RuntimeError(
            "no production-tagged model exists yet"
        )  # fail fast, tested — distinct from the gate's "first promotion" edge case: this is what happens if serving boots against a registry with zero versions ever registered (fresh clone/CI/dev env)
    ...


@app.post("/score")
def score(request: ScoreRequest) -> ScoreResponse:
    started = time.perf_counter()
    result = _score_with_loaded_model(request)
    logger.info(
        "score_request",
        extra={
            "model_version": _model_version,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "priority_class": result.priority_class,
        },
    )
    return result
```
Rare, human/CI-triggered promotion events don't justify per-request registry reads; hot-reload is legitimate future work, not built now. One structured log line per `/score` call (model version, latency, outcome) — cheap, standard for anything meant to run as a real service, and previously missing from this sketch entirely.

**Explicitly out of scope, and why (not an oversight):** no `/healthz` readiness probe and no handling for MLflow becoming unreachable mid-process. No orchestrator exists yet to probe a health endpoint (CLAUDE.md bans Dockerfile/deploy infra until serving has real code, so there's nothing to wire a probe into), and startup-time loading means `/score` never touches MLflow again after boot — "MLflow dies mid-request" is moot by construction, not a gap this design failed to consider.

**6. Config — a `pydantic` `Settings` class in `src/signalscore/settings.py`.** `pydantic.BaseSettings` moved to the separate `pydantic-settings` package in Pydantic v2, so that package is added as a new core dependency alongside the existing `pydantic`. One shared, typed settings object is read once at each of the 3 call sites (training, gate CLI, serving) instead of three ad hoc `os.environ.get()` calls — this also gives the MLflow tracking URI real validation for free and a single place to add future MLflow/W&B config keys, rather than duplicating env-var names as string literals across three files.

```python
# src/signalscore/settings.py
"""Shared runtime settings, loaded once from the environment / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
```

Usage: `from signalscore.settings import Settings; Settings().mlflow_tracking_uri`. `fetch_issues.py`'s existing `os.environ.get("GITHUB_TOKEN")` pattern is left as-is (out of scope for this integration) — `Settings` covers only the new MLflow-related config this plan introduces.

**7. Dependencies/CI — `mlflow` added to the existing `ml` extra**, plus `pydantic-settings` added to core `dependencies` (not an extra — `settings.py` is imported by `training`, `evaluation`, and `serving`, so it must resolve under plain `dev`-only installs too, same reasoning as why `mlflow` goes in `ml` rather than a narrower extra). `make install` only syncs `dev` + `ml`; `registry.py`/`settings.py` are imported by `evaluation`/`serving`, both of which need them resolvable without the `serving` extra. This claim holds **through PR 4** — no CI workflow changes needed there.

**Correction from review — PR 5 is not a no-op.** `serving/app.py` imports `fastapi`, which lives in the `serving` extra, but `Makefile`'s `install` target (`uv sync --group dev --extra ml`) never syncs it, and CI (`.github/workflows/ci.yml`) just runs `make install && make check`. Left as originally written, `make check` — pyright strict *and* pytest — breaks the moment PR 5's `import fastapi` lands, in CI and for anyone running `make check` locally without having manually installed the `serving` extra. **Fix, scoped into PR 5 itself:** add `--extra serving` to `Makefile`'s `install` target (`uv sync --group dev --extra ml --extra serving`) as part of that PR, not deferred to a hypothetical PR 6. PR 6 (§ below) is genuinely likely a no-op now that this is caught upstream.

CI does not run the gate as a real deploy-gating step (no CD pipeline exists to gate); it only unit-tests gate/registry logic against a real `tmp_path`-scoped local SQLite MLflow instance.

**8. Testing — real local MLflow (SQLite, `tmp_path`-scoped), zero mocking,** matching the repo's established no-mocking convention. Each test gets its own `tmp_path`-scoped tracking URI. Each `GateStep` subclass is tested individually (`ContractStep._check` etc.) against tiny synthetic `FeatureRow` lists + real tiny `BaselineArtifact`s built via `train_baseline.py`'s own `fit_vectorizers`/`train_classifier`; `build_promotion_gate()` gets its own test asserting the fixed order (a failing `UnitTestStep` short-circuits before `ContractStep`/`MarginStep` ever run). `check_model_contract`'s latency check uses a `time.sleep()` inside a fake `predict_proba` rather than mocking time. `UnitTestStep._check()` is tested by pointing at a tiny throwaway pass/fail test file under `tmp_path`, not the real `tests/features/` dir (keeps it fast, decoupled from real suite size). `margin.py`'s bootstrap CI gets its own test against a tiny synthetic eval set (e.g. 20 rows) with a fixed seed — asserts the CI-lower-bound logic itself (a clear win passes, a tie/loss doesn't), not full-scale statistical validity, which is what the 1,000-resample default is for at real eval-set size, not test time.

**9. Rollback — one explicit test satisfying the design doc's "should be tested once" instruction:** promote v1, promote v2 (demotes v1), call `rollback()`, assert production is back to v1 and v2 is now archived. **Revised after review to promote a third version, v3, before asserting anything** — the original two-promotion test wouldn't have caught the `prior_production`-uniqueness bug §1 fixes (it only manifests from the third promotion onward, when two versions could simultaneously carry the tag). Full sequence: promote v1 → promote v2 → promote v3 → assert exactly one version (v2) carries `prior_production=true` → `rollback()` → assert production is v2, and v3 is now archived with `prior_production=true` in v2's place.

## Files to change

- `src/signalscore/registry.py` — implement `ModelRegistry` per §1
- `src/signalscore/training/train_baseline.py` — `main()` MLflow integration per §4, hoist `MODEL_ARTIFACT_FILENAME`
- `src/signalscore/evaluation/__init__.py`, new `contract.py`, `margin.py`, `gate.py`, `run_gate.py` per §2
- `src/signalscore/serving/__init__.py` → new `app.py` (FastAPI `/score` MVP) per §5
- `pyproject.toml` — add `mlflow` to the `ml` extra, `pydantic-settings` to core `dependencies`
- `src/signalscore/settings.py` — new `Settings` class per §6
- `Makefile` — add `--extra serving` to the `install` target (PR 5, per the §7 correction)
- `.env.example`, `.gitignore` — MLflow tracking URI / `mlflow.db`
- `EXPERIMENTS.md` — new rows logged via existing `format_experiments_row()`, for both gate passes and losses (pre-check and margin)
- New tests: `tests/test_registry.py`, `tests/evaluation/test_contract.py`, `test_margin.py`, `test_gate.py`, `tests/training/` addition for the MLflow integration, `tests/serving/` smoke test

## PR sequencing (each ships its own tests, per repo convention)

1. **`registry.py` + `settings.py` + MLflow infra** — dependencies (`mlflow`, `pydantic-settings`), env/gitignore, `Settings` class, `ModelRegistry` fully implemented (including the `prior_production`-uniqueness fix), `tests/test_registry.py` (register/promote/reject/rollback across **three** promotions per §9's revised test, plus `get_production()` returning `None` before any promotion, i.e. before the `@production` alias has ever been set). Retires the SQLite-registry assumption risk first, before anything depends on it.
2. **Training integration** — `train_baseline.py::main()` MLflow logging (params incl. library versions + data snapshot row counts, timestamped run names) + `register_candidate()`, one new test.
3. **Evaluation gate logic** — `GateContext`, `contract.py` (incl. the DVC-based eval-set integrity check against the `eval-set-v1` tag), `margin.py` (bootstrap CI + non-regression floors), `gate.py`, fully tested against synthetic artifacts, decoupled from the registry.
4. **Gate CLI + registry wiring** — `run_gate.py`: loads via `ModelRegistry`, runs the gate, promotes/rejects (including the first-promotion quality floor), logs `EXPERIMENTS.md` rows + structured registry tags for both outcomes. Integration test exercising the full loop against a `tmp_path` SQLite DB.
5. **Serving MVP** — `/score` endpoint + startup-time load + fail-fast-on-empty-registry + per-request structured logging, smoke test. **Also adds `--extra serving` to `Makefile`'s `install` target** (§7 correction) in the same PR that introduces the `fastapi` import, so `make check`/CI never breaks.
6. **CI check** — now genuinely likely a no-op, since the one real CI gap (item 5's `serving` extra) is fixed upstream in PR 5 itself; only open this PR if something else turns up.

## Verification

- `make check` (lint, format-check, pyright strict, pytest+coverage) after each PR — must pass with zero new suppressions beyond the established per-line `# pyright: ignore[reportSpecificCode]` convention for weakly-typed MLflow/sklearn calls.
- Manually run the full loop once end-to-end locally: `uv run python -m signalscore.training.train_baseline --train ... --val ...` (registers a staging candidate) → `uv run python -m signalscore.evaluation.run_gate` (runs gate; first-ever promotion must clear the absolute PR-AUC floor, not just pass-through) → confirm `EXPERIMENTS.md` gained a real row and the registered version carries the structured `gate_result`/`eval_set_tag` tags → train a second (deliberately worse) candidate → confirm the gate rejects it via the bootstrap-CI/non-regression checks and logs the loss → promote a third candidate → call `ModelRegistry().rollback()` and confirm production reverts to the *second* version (not the first), and that only one version ever carries `prior_production=true` (satisfies both the design doc's explicit rollback-test requirement and the review-caught uniqueness bug in §1).
- Start the FastAPI app (`uvicorn signalscore.serving.app:app`) against the promoted model and hit `/score` once with a real GitHub-issue-shaped payload — confirm the startup-load path, the response schema, and the new structured log line all work end-to-end.
- Open the local MLflow UI (`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`) and spot-check that a run shows its hyperparameters, library versions, and data row counts together — the actual point of logging them.
