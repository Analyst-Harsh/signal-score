# Design patterns guide

## Purpose & scope

This doc sets the default for **new** code in Signal Score and flags two
existing modules as near-term refactor candidates (§9). It governs one
question: when should a piece of logic be a class (optionally implementing a
Gang-of-Four pattern) instead of a plain function?

It does **not** touch data-modeling classes. `pydantic.BaseModel`, `TypedDict`,
`dataclass`, and `StrEnum` used purely as typed containers — `FeatureRow`,
`FeatureBundle`, `DropReport`, `NearDuplicatePair`, `LabelMappingConfig` — are
the established convention here and are unaffected by anything below. A
`BaseModel` with no methods is not a "should this be a class" decision in the
GoF sense; it's just a typed value.

## The core rule

> Prefer a class only where the code has a genuine, currently-real extension
> point — multiple interchangeable behaviors selected at runtime, or state
> that must persist and mutate across calls. Once that's established,
> implement it with the least structure that still provides that
> extensibility. A class with one method and no state is a function wearing a
> costume — simplifying "as much as possible" includes taking the costume
> off.

"Prefer classes" and "simplify as much as possible" answer different
questions, not competing ones: the core rule decides *which shape wins* when
a real design decision exists; everything past that point decides *how much
structure* that shape is allowed to get. Neither instruction licenses
building structure for a decision that doesn't exist yet.

## Quick decision checklist

Run top to bottom. Stop at the first row that matches.

| Question | If yes → |
|---|---|
| Does this unit need to remember something across calls (a fitted vectorizer, a trained model, an open client, accumulated audit state)? | **Class.** A free function forces that state into a global, a closure, or a parameter threaded through every call site. |
| Are there ≥2 real (not hypothetical) implementations of the same operation, selected by a config value or CLI flag? | **Strategy** + Factory Method to construct it. |
| Do ≥2 call sites share an ordered sequence of steps where only 1–2 steps differ, or is the sequence a hard, frozen rule (like the promotion gate)? | **Template Method** (or Chain of Responsibility if steps must be independently addable/removable — see §5.5 vs §5.2). |
| Does building the object take ≥4 ordered steps with intermediate state that shouldn't be "half-valid"? | **Builder.** |
| Does a value pass through an ordered sequence of independent accept/reject/transform steps that should be addable/removable without touching the others? | **Chain of Responsibility.** |
| Are you calling an untyped or unstable third-party surface (sklearn, XGBoost, MLflow, Evidently) from more than one place, holding a client/state across calls? | **Adapter** (class). A stateless one-shot call to the same surface stays a wrapper *function*. |
| Does one event need to notify a growing/pluggable set of independent listeners? | **Observer.** |
| None of the above — one input, one output, no state, one implementation? | **Plain function. Stop here.** |

## Smell-test checklist

Before writing any class/pattern from the catalog below, answer these in
order. Stop and use a plain function the moment one fails.

1. **Count real implementations that exist right now** (not "might exist
   later"). Is it ≥2? If it's 1, stop — no Strategy, no Factory, no ABC.
2. **Is there cross-call state?** Fields set in `__init__` read by more than
   one method, or a resource (file handle, model weights, HTTP session) held
   open across calls? If no state and one public method, it's a function.
3. **Delete-and-inline test:** if you deleted the class and pasted its one
   method's body at the call site, would any caller's behavior change? If
   no, delete the class.
4. **Is the extension point written down somewhere real** — a design doc, an
   actual config value flipped today — or is it speculative "we might need
   this later"? Speculative → skip it, no doc overrides YAGNI here.
5. **Is dispatch actually runtime-selected** among ≥2 real options, or is the
   `if/elif` in the "factory" dead code with one live branch?
6. **Does the pattern reduce branching/complexity, or just relocate it?** If
   cyclomatic complexity moves from one function into three files with the
   same total branches, that's added indirection, not simplification.
7. **One-file test:** do the class and its only call site live in the same
   short function/module? If a reader has to jump files to understand a
   5-line operation, the pattern made the code harder to read — that fails
   "simplify as much as possible" outright.
8. **Would a senior reviewer ask "why is this a class?"** If the honest
   answer is "it might be reused" or "it's more testable" with no concrete
   second caller or concrete untestable behavior today, that's not a yes.

If every check that applies passes — genuinely ≥2 implementations, real
state, a written-down extension point, actual runtime dispatch — use the
class/pattern. Otherwise, function.

## Pattern catalog

### 1. Strategy — interchangeable behaviors selected at runtime

**Trigger:** a config value selects between ≥2 real implementations of the
same operation.

**Where it fires today:** training's model configs (sklearn logistic
regression vs. XGBoost — `docs/signalscore-design.md` already names both).
**Where it will fire soon:** `features/pipeline.py`'s `build_features` today
hardcodes exactly two extractors (text, metadata); BGE embeddings are the
named third feature category in the design doc, which is a concrete,
documented second/third implementation arriving — the textbook trigger. When
that work starts, extract to:

```python
class FeatureExtractor(Protocol):
    def extract(self, title: str, body: str, author_association: str | None) -> dict[str, Any]: ...

class TextFeatureExtractor:
    def extract(self, title: str, body: str, author_association: str | None) -> dict[str, Any]:
        return extract_text_features(title, body)

class MetadataFeatureExtractor:
    def extract(self, title: str, body: str, author_association: str | None) -> dict[str, Any]:
        return extract_metadata_features(author_association)

class EmbeddingFeatureExtractor:  # added when BGE embedding work actually starts
    def __init__(self, model_name: str) -> None:
        self._model = load_bge_model(model_name)  # real cross-call state: loaded weights

    def extract(self, title: str, body: str, author_association: str | None) -> dict[str, Any]:
        return {"embedding": self._model.encode(f"{title}\n{body}")}

class FeaturePipeline:
    def __init__(self, extractors: list[FeatureExtractor]) -> None:
        self._extractors = extractors

    def build(self, title: str, body: str, author_association: str | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for extractor in self._extractors:
            merged.update(extractor.extract(title, body, author_association))
        return merged
```

`build_features` stays the one frozen call site injected identically into
training, evaluation, serving, and monitoring (its own docstring's guarantee)
— `FeaturePipeline` makes the *composition* swappable without loosening that
guarantee, and turns "text-only vs. text+metadata vs. text+metadata+embedding"
ablations (already run and logged per `docs/v0-feature-selection.md`) into a
config change instead of a code change.

**Anti-pattern:** don't build this for a single algorithm/extractor "in case
a second one shows up later." One implementation today → plain function,
exactly as `build_features` is now. Extract the interface in the same commit
that adds the second real implementation.

### 2. Template Method — a frozen, ordered sequence of steps

**Trigger:** a hard rule specifies a fixed step order where individual steps
vary, or ≥2 call sites share that order.

**Where it fires:** the promotion gate (not yet built). CLAUDE.md's rule —
"unit tests → contract test → margin win, never hand-flipped" — is a frozen
skeleton with swappable steps, which is the definition of Template Method:

```python
class PromotionGate:
    """Skeleton is frozen per CLAUDE.md: unit tests -> contract -> margin eval."""

    def run(self, candidate: TrainedModel, production: TrainedModel, eval_set: EvalSet) -> GateResult:
        if not self.unit_tests_pass():
            return GateResult(passed=False, failed_stage="unit_tests")
        if not self.contract_test_pass(candidate):
            return GateResult(passed=False, failed_stage="contract")
        margin = self.head_to_head_margin(candidate, production, eval_set)
        return GateResult(passed=margin > 0, failed_stage=None, margin=margin)

    def unit_tests_pass(self) -> bool: ...
    def contract_test_pass(self, candidate: TrainedModel) -> bool: ...
    def head_to_head_margin(self, candidate: TrainedModel, production: TrainedModel, eval_set: EvalSet) -> float: ...
```

**Anti-pattern:** don't reach for this over Chain of Responsibility (§5) when
a step must be skippable/omittable by a caller — see the hard warning in §7.
A one-off sequence used at exactly one call site with no variance isn't a
template method either; it's a function calling three other functions in
order (see how `run_pipeline` in `training/build_dataset.py` does exactly
this today and is correctly left as a function, per §9).

### 3. Builder — multi-step construction with real intermediate state

**Trigger:** ≥4 ordered steps with intermediate state that shouldn't be
usable half-built.

**Where it fires:** `training/build_dataset.py`'s `run_pipeline` today
threads `deduped` → `kept_rows, report` → `frame` → `cutoff` → `assignments`
through eight sequential steps via local variables and file writes — the
Builder shape, currently tolerable as one CLI-script function because there's
exactly one entry point (`main`). It becomes worth extracting the moment a
second real entry point needs the same sequence (see §9):

```python
class DatasetBuilder:
    def __init__(self, repo: str, config: LabelMappingConfig) -> None:
        self._repo = repo
        self._config = config
        self._kept_rows: list[dict[str, Any]] = []
        self._frame: list[FeatureRow] = []

    def load_raw(self, paths: list[Path]) -> "DatasetBuilder":
        deduped = dedupe_by_issue_number(load_raw_jsonl(paths))
        self._kept_rows, self.drop_report = resolve_and_filter(
            deduped, self._repo, self._config, total_input_rows=len(deduped)
        )
        return self

    def extract_features(self) -> "DatasetBuilder":
        self._frame = build_feature_frame(self._kept_rows)
        return self

    def build(self) -> list[FeatureRow]:
        return self._frame
```

**Anti-pattern:** don't builder-ify a 2-3 step construction with no
meaningful intermediate state — `build_features` calling
`extract_text_features` then `extract_metadata_features` is a function
calling two functions; leave it exactly as-is.

### 4. Factory Method — selecting a concrete implementation from config

**Trigger:** the concrete type is chosen by a string/enum read at runtime,
and there are ≥3 real variants. Below 3, a plain `dict[str, Callable]` or
`if/elif` dispatch is simpler and is what ruff's SIM rule will ask you to
revert to anyway.

```python
class TrainingStrategyFactory:
    _registry: ClassVar[dict[str, type[TrainingStrategy]]] = {
        "sklearn_logreg": SklearnLogRegStrategy,
        "xgboost": XGBoostStrategy,
    }

    @classmethod
    def create(cls, model_type: str, **hyperparams: Any) -> TrainingStrategy:
        try:
            strategy_cls = cls._registry[model_type]
        except KeyError:
            raise ValueError(f"unknown model_type {model_type!r}") from None
        return strategy_cls(**hyperparams)
```

**Anti-pattern:** two variants and no sign of a third this quarter → skip the
factory, construct the strategy directly at the call site.

### 5. Chain of Responsibility — independent, addable/removable steps

**Trigger:** ≥3 independent accept/reject/transform steps over one value that
need to be addable/removable without touching the others. At 1-2 steps, an
`if` in the calling function is simpler.

**Where it fires:** serving's `/score` request validation (not yet built) —
new validation rules will be added by different people at different times,
and shouldn't require touching a monolithic `if` ladder:

```python
class ValidationStep(Protocol):
    def check(self, payload: ScoreRequest) -> str | None: ...  # None = pass

class RequiredFieldsStep:
    def check(self, payload: ScoreRequest) -> str | None:
        return "title must not be empty" if not payload.title.strip() else None

class ValidationChain:
    def __init__(self, steps: list[ValidationStep]) -> None:
        self._steps = steps

    def validate(self, payload: ScoreRequest) -> list[str]:
        return [msg for step in self._steps if (msg := step.check(payload)) is not None]
```

**Anti-pattern note:** `features/loading.py`'s `resolve_and_filter` has two
drop reasons (excluded label, ambiguous multi-priority label) inline in one
function — that's correct as-is; two branches don't justify a chain. If a
third and fourth drop reason show up, that's the trigger to convert, not
before.

### 6. Adapter — walling off untyped/unstable third-party surfaces

**Trigger:** any call into sklearn, XGBoost, MLflow, or Evidently.

**Rule of thumb:** stateless third-party call, one call site → typed wrapper
*function*. `training/leakage_audit.py::_pairwise_cosine_similarity` already
does exactly this correctly — it isolates sklearn's untyped `Unknown` return
types behind one typed function, fits a throwaway vectorizer per call, and
stays a function because it holds no state and has one caller shape. Leave it
as-is.

Stateful third-party client, called from multiple modules → wrapper *class*.
`registry.py` (currently a docstring-only stub, named by CLAUDE.md as the
shared MLflow client for `training`/`evaluation`/`serving`) is where this
becomes a class once implemented, because it holds a client + model name
across calls from three different modules:

```python
class ModelRegistry:
    """Every mlflow call in the repo goes through here."""

    def __init__(self, client: MlflowClient, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def promote_to_production(self, version: str) -> None:
        self._client.transition_model_version_stage(self._model_name, version, stage="Production")

    def get_production_model(self) -> TrainedModel:
        mv = self._client.get_latest_versions(self._model_name, stages=["Production"])[0]
        return TrainedModel.load(mv.source)
```

**Anti-pattern:** don't wrap a single one-shot external call in an
`Adapter`/`Gateway` class with one method — that's renaming a function call,
not isolating a dependency.

### 7. Observer — one event, a growable set of independent listeners

**Trigger:** an event needs to notify ≥2 real, independently-growable
listeners.

**Where it fires:** `monitoring`'s Evidently drift watcher observing live
`/score` traffic (not yet built), where a served prediction needs to reach a
drift buffer without the scoring code knowing what watches it:

```python
class DriftObserver(Protocol):
    def on_prediction(self, request: ScoreRequest, prediction: Priority) -> None: ...

class ScoringService:
    def __init__(self, model: TrainedModel, observers: list[DriftObserver]) -> None:
        self._model = model
        self._observers = observers

    def score(self, request: ScoreRequest) -> Priority:
        prediction = self._model.predict(request)
        for observer in self._observers:
            observer.on_prediction(request, prediction)
        return prediction
```

**Anti-pattern:** one known listener, no plan for a second → call it
directly, skip the list/Protocol machinery until a second listener is real.

## Anti-pattern guardrails — do NOT reach for these

1. **No Strategy/ABC/Protocol for a single implementation.** One concrete
   behavior today is a function, full stop.
2. **No Factory for one product type.** An `if/elif` with one live branch is
   indirection with no payoff — call the constructor.
3. **No class wrapping a single pure function.** One public method, no
   `__init__` state read by more than one method, no lifecycle → it's a
   function in a costume. This is the single most common failure mode when
   "prefer classes" is applied eagerly: turning
   `def title_length(issue): return len(issue.title)` into a
   `TitleLengthExtractor` class. Don't.
4. **No Observer/pub-sub for a same-call-stack function call.** One
   "listener" known at write time isn't an event system.
5. **No Builder for a small, single-call construction.** If building the
   object is one constructor call with keyword args, a `Builder` class adds a
   second thing to maintain in lockstep with the first.
6. **No Singleton.** This is a stateless FastAPI app plus batch scripts.
   "Only one instance" is already achieved by a module-level object or an
   `lru_cache`d factory function — a `Singleton` class adds global mutable
   state and import-order hazards for no benefit here.
7. **No Template Method with one subclass.** Same disease as #1, aimed at
   inheritance: collapse a base class + single-subclass pair into one
   function calling its helper functions directly.
8. **No Adapter class for one external call.** A thin wrapper function (or
   nothing) is enough unless you're adapting a genuinely awkward multi-method
   interface used from several places.
9. **No Repository pattern over one data source with no swap-out plan.**
10. **No "DI theater."** Injecting an interface purely so a test can pass a
    fake, when a plain argument or `unittest.mock.patch` at the module
    boundary would do, is an abstraction paid for entirely by testability. If
    the only consumer of an interface is the test suite, don't build the
    interface.
11. **No inheritance depth beyond one level in `features/`/`training/`
    code.** This codebase does data transforms, not GUI widget trees. Flatten
    to composition (pass a function or a small config object) instead.
12. **No class introduced solely because a ruff SIM/PERF finding needs
    "somewhere to live."** Check whether deleting code satisfies the linter
    before adding a class satisfies it.

## Hard warnings specific to this ML pipeline

1. **No Singleton for fitted transformers or embedding models.** A shared,
   module-level fitted vectorizer/embedder (a) survives across pytest cases
   unless explicitly reset, silently coupling test outcomes; (b) if ever
   refit on a serving-path request instead of loaded from the exact
   training-time artifact, produces instant train/serve skew with no error —
   just quietly wrong scores; (c) reused across CV folds, leaks val/test
   vocabulary into train. `leakage_audit.py::_pairwise_cosine_similarity`
   already gets this right by instantiating a fresh `TfidfVectorizer()` per
   call — don't "optimize" that into a shared instance for speed.
2. **No Strategy on `training/split.py`'s frozen split logic.**
   `determine_eval_cutoff` and `hash_split` are deliberately locked to one
   algorithm — the design doc specifies the cutoff is computed once and
   never regenerated, and comparisons are never re-shuffled between retrain
   cycles. A pluggable split-strategy interface invites swapping split logic
   per run, which breaks the frozen-comparison guarantee the promotion gate
   depends on. Keep this a single concrete implementation, no interface.
3. **No stateful Builder that caches derived state across folds/splits.**
   `assemble_splits` recomputes `frozen_eval_ids` fresh from the frozen file
   on every call specifically to avoid a stale-cache bug. Any class-based
   rewrite of dataset assembly must preserve "always re-derive from the
   frozen file," never "cache eval ids as instance state."
4. **Prefer Chain of Responsibility over Template Method for the promotion
   gate and leakage audits where a step must hard-fail rather than be
   skippable.** Template Method's premise is that a subclass can override or
   skip a step — exactly wrong for "pre-checks gate the comparison" and for
   `audit_label_drift`, which today hard-fails the pipeline
   (`raise SystemExit`) rather than being silently bypassable via an
   inheritance hook.
5. **Watch cache/proxy decorators around `model.predict()` in serving.** A
   caching or logging decorator on the scoring path is fine, but if its
   cache key doesn't include `model_version`, a promotion-gate tag flip
   (staging → production) can silently keep serving stale predictions from
   the old model.

## Working with ruff (SIM/PERF) and pyright strict

- If ruff flags a class you wrote under this doc, first check whether it
  meets one of the checklist's triggers. If it doesn't, delete the class —
  ruff is right. If it does (e.g. it's a Strategy implementation with a real
  sibling already in the codebase), the justification belongs in a short
  class docstring, not a `# noqa`.
- PERF flags apply identically in class or function form — a
  `Strategy.fit()` method looping over a DataFrame in Python is exactly as
  wrong as a function doing the same.
- Every `Protocol`/`ABC` used for Strategy/Adapter/Observer needs full type
  annotations on every abstract method so pyright strict can verify
  implementations at the call site — catching a missing `fit()` method at
  typecheck time instead of at runtime three stages downstream is the actual
  payoff of a typed interface over a duck-typed dict-of-callables here.

## Migration stance / refactor candidates

`features/text.py`, `features/metadata.py`, `features/labels.py`,
`features/loading.py`, and `training/{split,leakage_audit}.py` stay
function-based as-is — they're pure, stateless, already tested, and none of
the checklist's triggers apply. Converting them would itself be the
anti-pattern this doc warns against.

Two files are explicitly flagged as near-term refactor candidates — refactor
when next touched for the reason named below, not on a scheduled pass:

1. **`src/signalscore/features/pipeline.py`** — `build_features`/
   `build_feature_frame` → the Strategy pattern in §Pattern-catalog-1
   (`FeatureExtractor` Protocol + `FeaturePipeline`), triggered by the
   already-planned BGE embeddings work adding a genuine second/third
   interchangeable extractor.
2. **`src/signalscore/training/build_dataset.py`** — `run_pipeline`'s fixed
   multi-step orchestration → `DatasetBuilder` (§Pattern-catalog-3),
   triggered when a second real dataset-assembly entry point is needed (e.g.
   the k8s-sigs volume supplement named in CLAUDE.md's module map).

## Worked example (composed reference)

One sketch showing Strategy, Factory Method, Template Method, and Adapter
cooperating, once `evaluation/`, `serving/`, and `registry.py` are actually
built:

```python
# training: Strategy + Factory Method choose the algorithm
strategy = TrainingStrategyFactory.create("xgboost", max_depth=6, n_estimators=200)
candidate = strategy.fit(X_train, y_train)

# registry.py: Adapter isolates the MLflow client
registry = ModelRegistry(client=MlflowClient(), model_name="signal-score-priority")
production = registry.get_production_model()

# evaluation: Template Method runs the frozen 3-step gate
gate = PromotionGate()
result = gate.run(candidate, production, eval_set=load_frozen_eval_set(FROZEN_EVAL_PATH))

if result.passed:
    registry.promote_to_production(candidate.version)
else:
    log_to_experiments_md(result)  # CLAUDE.md: losses get logged, never discarded
```
