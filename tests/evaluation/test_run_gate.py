"""Integration tests for signalscore.evaluation.run_gate.

Real local SQLite MLflow instance (tmp_path-scoped), zero mocking, per repo
convention -- matches tests/test_registry.py's own idiom for registering real
logged artifacts, and tests/training/test_train_baseline.py's idiom for
pointing `main()` at a tmp_path tracking URI via the MLFLOW_TRACKING_URI env
var (Settings() reads it; ModelRegistry()'s default MlflowClient() needs the
process-global tracking URI set before construction).

The real frozen eval set is DVC-tracked and gitignored (only its `.dvc`
pointer is git-committed) -- it lives in a machine-local DVC remote
(`dvc-storage`, a sibling directory outside the repo) and is never present on
a CI runner. Every test here builds its own throwaway git+dvc repo (via
`tests.evaluation._helpers.init_frozen_git_dvc_repo`) and points `run_gate`'s
`--eval-set` at it, so `ContractStep`'s real DVC integrity check runs for
real -- just against a small, self-contained, CI-safe fixture instead of the
real production data. Candidate/production PR-AUC is whatever real scoring
against that fixture set produces, not a hand-picked number: a "weak"
artifact is trained on same-text-every-class noise rows (genuinely no
class-discriminative signal, so it scores near chance level against any real
eval set); a "strong" artifact is trained directly on the eval set's own rows
(genuine signal against that same set, a deliberate leakage-based
determinism shortcut, not a claim about real generalization) -- both are real
BaselineArtifacts fit via train_baseline.py's own
fit_vectorizers/train_classifier, scored honestly, never fabricated metrics.
"""

from pathlib import Path

import joblib
import mlflow
import pytest
from mlflow.tracking import MlflowClient

from signalscore.evaluation import contract, run_gate
from signalscore.evaluation.margin import score_artifact
from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.registry import DEFAULT_MODEL_NAME, ModelRegistry, ModelVersionInfo
from signalscore.training.train_baseline import (
    MODEL_ARTIFACT_FILENAME,
    BaselineArtifact,
    compute_metrics,
)
from tests.evaluation._helpers import (
    build_artifact,
    init_frozen_git_dvc_repo,
    make_feature_row,
    make_rows,
)

_NOISE_TEXT = "system observed event occurred process ran executed completed"


def _noise_rows(n_per_class: int) -> list[FeatureRow]:
    """Identical text across all three classes -- carries no
    class-discriminative signal, so a model trained on this genuinely can't
    separate classes and scores near chance-level PR-AUC on any real eval
    set, deterministically below FIRST_PROMOTION_MIN_PR_AUC.
    """
    rows: list[FeatureRow] = []
    issue = 0
    for label in Priority:
        for i in range(n_per_class):
            rows.append(make_feature_row(issue, f"{_NOISE_TEXT} entry {i}", label=label))
            issue += 1
    return rows


def _frozen_eval_set(tmp_path: Path) -> tuple[Path, list[FeatureRow]]:
    """Builds a real, throwaway, DVC-frozen eval set under tmp_path -- plays
    the role the real production eval_set_v1.jsonl plays in real usage, but
    self-contained so ContractStep's DVC check is exercised for real without
    depending on data that only exists on this machine.
    """
    rows = make_rows(n_per_class=40)
    content = "\n".join(row.model_dump_json() for row in rows)
    eval_file = init_frozen_git_dvc_repo(tmp_path, content=content)
    return eval_file, rows


def _register_artifact(
    client: MlflowClient,
    registry: ModelRegistry,
    artifact: BaselineArtifact,
    tmp_path: Path,
    subdir: str,
) -> ModelVersionInfo:
    """Registers `artifact` as a real logged MLflow model version -- same
    idiom as tests/test_registry.py's
    test_resolve_artifact_path_downloads_real_logged_artifact: a real run,
    a real logged artifact file, then register_candidate() against it.
    """
    local_dir = tmp_path / subdir
    local_dir.mkdir()
    artifact_file = local_dir / MODEL_ARTIFACT_FILENAME
    joblib.dump(artifact, artifact_file)  # pyright: ignore[reportUnknownMemberType]
    run = client.create_run(experiment_id="0")
    run_id: str = run.info.run_id
    client.log_artifact(run_id, str(artifact_file), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
    return registry.register_candidate(run_id=run_id)


def _run_main(
    tracking_uri: str, monkeypatch: pytest.MonkeyPatch, candidate_version: str, eval_set_path: Path
) -> None:
    """Points run_gate.main() at the tmp_path SQLite DB the same way
    test_train_baseline.py's test_main_registers_staging_candidate... does,
    restoring the true prior global tracking URI afterward so this test
    can't leak into any other test. `--eval-set` points ContractStep's DVC
    check and the rows scored/gated against at a throwaway fixture, never the
    real production eval set (see module docstring).
    """
    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        run_gate.main(["--candidate-version", candidate_version, "--eval-set", str(eval_set_path)])
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]


def _weak_artifact() -> BaselineArtifact:
    """Same-text-every-class noise rows (tests/evaluation/_helpers.py) --
    carries no class-discriminative signal, so it reliably scores near
    chance-level PR-AUC when honestly evaluated against any real eval set.
    """
    return build_artifact(_noise_rows(n_per_class=40))


def _strong_artifact(eval_rows: list[FeatureRow], n_train_rows: int) -> BaselineArtifact:
    """Trained on a slice of the eval set's own rows -- genuine signal
    against that same set (the model has actually seen this vocabulary/label
    pairing), so it reliably scores well above the first-promotion floor when
    evaluated against the full eval set. This is a deliberate test-fixture
    shortcut (leakage), not a claim about real generalization performance --
    it exists only to make pass/fail deterministic without hand-picking a
    fabricated metric.
    """
    return build_artifact(eval_rows[:n_train_rows])


def test_parse_args_requires_candidate_version() -> None:
    args = run_gate.parse_args(["--candidate-version", "3"])
    assert args.candidate_version == "3"
    assert args.dataset_snapshot == "kubernetes-kubernetes"
    assert args.notes == ""
    assert args.eval_set == contract.EVAL_SET_PATH


def test_first_promotion_that_clears_the_floor_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_set_path, eval_rows = _frozen_eval_set(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)

    strong = _strong_artifact(eval_rows, n_train_rows=len(eval_rows))
    y_true, y_pred, y_proba, classes = score_artifact(strong, eval_rows)
    pr_auc = compute_metrics(y_true, y_pred, y_proba, classes)["pr_auc_macro"]
    assert pr_auc > run_gate.FIRST_PROMOTION_MIN_PR_AUC  # sanity check on the fixture itself

    candidate = _register_artifact(client, registry, strong, tmp_path, "candidate")

    assert registry.get_production() is None  # no promotion has happened yet

    _run_main(tracking_uri, monkeypatch, candidate.version, eval_set_path)

    production = registry.get_production()
    assert production is not None
    assert production.version == candidate.version
    version_tags = client.get_model_version(DEFAULT_MODEL_NAME, candidate.version).tags
    assert version_tags.get("gate_result") == "pass (first promotion)"
    assert version_tags.get("eval_set_tag") == contract.FROZEN_EVAL_TAG


def test_first_promotion_that_does_not_clear_the_floor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_set_path, _ = _frozen_eval_set(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)

    weak = _weak_artifact()
    candidate = _register_artifact(client, registry, weak, tmp_path, "candidate")

    _run_main(tracking_uri, monkeypatch, candidate.version, eval_set_path)

    assert registry.get_production() is None  # never set
    rejected = registry.get_version(candidate.version)
    assert rejected.stage == "rejected"
    tags = client.get_model_version(DEFAULT_MODEL_NAME, candidate.version).tags
    assert "first-promotion floor not cleared" in tags.get("rejection_reason", "")
    assert tags.get("gate_result") == "fail"


def test_second_candidate_beats_production_on_margin_and_is_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_set_path, eval_rows = _frozen_eval_set(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)

    weak_production = _register_artifact(client, registry, _weak_artifact(), tmp_path, "production")
    registry.promote(
        weak_production.version,
        gate_result="pass (first promotion)",
        eval_set_tag=contract.FROZEN_EVAL_TAG,
    )

    strong_candidate = _register_artifact(
        client,
        registry,
        _strong_artifact(eval_rows, n_train_rows=len(eval_rows)),
        tmp_path,
        "candidate",
    )

    _run_main(tracking_uri, monkeypatch, strong_candidate.version, eval_set_path)

    production = registry.get_production()
    assert production is not None
    assert production.version == strong_candidate.version
    assert registry.get_version(weak_production.version).stage == "archived"
    tags = client.get_model_version(DEFAULT_MODEL_NAME, strong_candidate.version).tags
    assert tags.get("gate_result") == "pass"


def test_second_candidate_loses_on_margin_and_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_set_path, eval_rows = _frozen_eval_set(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)

    strong_production = _register_artifact(
        client,
        registry,
        _strong_artifact(eval_rows, n_train_rows=len(eval_rows)),
        tmp_path,
        "production",
    )
    registry.promote(
        strong_production.version,
        gate_result="pass (first promotion)",
        eval_set_tag=contract.FROZEN_EVAL_TAG,
    )

    weak_candidate = _register_artifact(client, registry, _weak_artifact(), tmp_path, "candidate")

    _run_main(tracking_uri, monkeypatch, weak_candidate.version, eval_set_path)

    production = registry.get_production()
    assert production is not None
    assert production.version == strong_production.version  # unchanged

    rejected = registry.get_version(weak_candidate.version)
    assert rejected.stage == "rejected"
    tags = client.get_model_version(DEFAULT_MODEL_NAME, weak_candidate.version).tags
    assert tags.get("gate_result") == "fail"
    assert tags.get("rejection_reason")
