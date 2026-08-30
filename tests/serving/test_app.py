"""Smoke tests for signalscore.serving.app.

Real local SQLite MLflow instance (tmp_path-scoped), zero mocking, per repo
convention -- same idiom as tests/test_registry.py and
tests/evaluation/test_run_gate.py: a real MlflowClient, a real logged
artifact registered via `register_candidate`/`promote`, then `main()`-shaped
code (here, the FastAPI lifespan) pointed at that tmp_path DB via the
MLFLOW_TRACKING_URI env var.
"""

from pathlib import Path

import joblib
import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient

from signalscore.features.labels import Priority
from signalscore.registry import DEFAULT_MODEL_NAME, ModelRegistry
from signalscore.serving.app import app
from signalscore.training.train_baseline import MODEL_ARTIFACT_FILENAME, BaselineArtifact
from tests.evaluation._helpers import build_artifact, make_rows


def _register_and_promote_production(
    client: MlflowClient, registry: ModelRegistry, tmp_path: Path
) -> str:
    """Registers a real BaselineArtifact (fit via train_baseline.py's own
    fit_vectorizers/train_classifier, per tests/evaluation/_helpers.py) as a
    real logged MLflow model version and promotes it straight to
    production -- app.py always reads DEFAULT_MODEL_NAME, so this must
    register under that exact name (unlike tests/test_registry.py's
    "test-model", which app.py never looks at).
    """
    artifact: BaselineArtifact = build_artifact(make_rows(n_per_class=8))
    local_dir = tmp_path / "model"
    local_dir.mkdir()
    artifact_file = local_dir / MODEL_ARTIFACT_FILENAME
    joblib.dump(artifact, artifact_file)  # pyright: ignore[reportUnknownMemberType]
    run = client.create_run(experiment_id="0")
    run_id: str = run.info.run_id
    client.log_artifact(run_id, str(artifact_file), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
    info = registry.register_candidate(run_id=run_id)
    registry.promote(info.version, gate_result="pass (first promotion)", eval_set_tag="eval-set-v1")
    return info.version


def test_score_returns_prediction_when_production_model_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    expected_version = _register_and_promote_production(client, registry, tmp_path)

    # app.py's lifespan re-reads Settings() (env-driven) and calls
    # mlflow.set_tracking_uri() itself -- capture/restore the true prior
    # global state so this test can't leak its tmp_path DB into any other
    # test, matching tests/evaluation/test_run_gate.py's own convention.
    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/score",
                json={
                    "text": "critical outage pod crash cluster panic urgent failure",
                    "is_member_plus": True,
                },
            )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]

    assert response.status_code == 200
    body = response.json()
    valid_classes = {p.value for p in Priority}
    assert body["priority_class"] in valid_classes
    assert set(body["priority_probabilities"]) == valid_classes
    assert abs(sum(body["priority_probabilities"].values()) - 1.0) < 1e-6
    assert body["model_version"] == expected_version


def test_startup_fails_fast_when_no_production_model_registered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # fresh, empty registry
    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with pytest.raises(RuntimeError, match="no production-tagged model"), TestClient(app):
            pass
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
