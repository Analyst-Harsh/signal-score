"""Smoke tests for signalscore.serving.app.

Real local SQLite MLflow instance (tmp_path-scoped), zero mocking, per repo
convention -- same idiom as tests/test_registry.py and
tests/evaluation/test_run_gate.py: a real MlflowClient, a real logged
artifact registered via `register_candidate`/`promote`, then `main()`-shaped
code (here, the FastAPI lifespan) pointed at that tmp_path DB via the
MLFLOW_TRACKING_URI env var.
"""

import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray

from signalscore.features.labels import Priority
from signalscore.registry import DEFAULT_MODEL_NAME, ModelRegistry
from signalscore.serving.app import (
    ScoreRequest,
    _to_feature_row,  # pyright: ignore[reportPrivateUsage]
    app,
)
from signalscore.training.train_baseline import MODEL_ARTIFACT_FILENAME, BaselineArtifact
from tests.evaluation._helpers import build_artifact, make_rows

_BGE_CLASSES = [p.value for p in Priority]


class _FixedBgeModel:
    """A real (picklable) object with hardcoded predict/predict_proba output --
    same idea as tests/evaluation/test_margin.py's _FixedModel -- so the mocked
    BGE end-to-end /score test below doesn't need a real trained classifier.
    """

    def __init__(self, predicted_class: str, proba_row: list[float]) -> None:
        self._predicted_class = predicted_class
        self._proba_row = np.array(proba_row)
        self.classes_ = np.array(_BGE_CLASSES)

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.str_]:
        return np.array([self._predicted_class] * x.shape[0])

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.tile(self._proba_row, (x.shape[0], 1))


def test_to_feature_row_goes_through_the_frozen_feature_contract() -> None:
    """_to_feature_row must call build_features(), not hand-roll title+body
    concatenation or the author_association -> is_member_plus mapping --
    that's the whole point of the fix (see features/pipeline.py's own
    docstring: "the frozen serving contract... so train/serve skew is
    structurally impossible").
    """
    payload = ScoreRequest(
        title="pod crash",
        body="cluster panic",
        author_association="MEMBER",
    )

    row = _to_feature_row(payload)

    assert row.text == "pod crash\ncluster panic"
    assert row.is_member_plus is True
    assert row.title_chars == len("pod crash")
    assert row.body_chars == len("cluster panic")

    anonymous_row = _to_feature_row(ScoreRequest(title="t", body="b", author_association=None))
    assert anonymous_row.is_member_plus is False


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


def _register_and_promote_bge_production(
    client: MlflowClient,
    registry: ModelRegistry,
    tmp_path: Path,
    *,
    revision_sha: str = "test-revision-sha",
) -> str:
    """Same idea as _register_and_promote_production, but registers a
    feature_source='bge' artifact -- word_vectorizer/char_vectorizer=None,
    extra carrying the revision_sha app.py's lifespan reads to load the BGE
    model.
    """
    artifact = BaselineArtifact(
        word_vectorizer=None,
        char_vectorizer=None,
        model=_FixedBgeModel(Priority.P0_CRITICAL.value, [0.7, 0.2, 0.1]),
        classes=_BGE_CLASSES,
        model_type="logreg",
        extra={"revision_sha": revision_sha},
        feature_source="bge",
    )
    local_dir = tmp_path / "bge-model"
    local_dir.mkdir()
    artifact_file = local_dir / MODEL_ARTIFACT_FILENAME
    joblib.dump(artifact, artifact_file)  # pyright: ignore[reportUnknownMemberType]
    run = client.create_run(experiment_id="0")
    run_id: str = run.info.run_id
    client.log_artifact(run_id, str(artifact_file), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
    info = registry.register_candidate(run_id=run_id)
    registry.promote(info.version, gate_result="pass (test)", eval_set_tag="eval-set-v1")
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
                    "title": "critical outage",
                    "body": "pod crash cluster panic urgent failure",
                    "author_association": "MEMBER",
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


def test_lifespan_sets_bge_model_for_bge_production_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    _register_and_promote_bge_production(client, registry, tmp_path, revision_sha="my-revision")

    sentinel_model = object()
    load_calls: list[str] = []

    def fake_load_bge_model(revision_sha: str) -> object:
        load_calls.append(revision_sha)
        return sentinel_model

    monkeypatch.setattr("signalscore.serving.app.load_bge_model", fake_load_bge_model)

    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with TestClient(app):
            assert app.state.bge_model is sentinel_model
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
        # `app` is a module-level singleton shared across every test in this
        # file -- clear the attribute this test set so it can't leak into a
        # later test's `not hasattr(app.state, "bge_model")` assertion.
        del app.state.bge_model

    assert load_calls == ["my-revision"]


def test_lifespan_does_not_set_bge_model_for_tfidf_production_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    _register_and_promote_production(client, registry, tmp_path)

    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with TestClient(app):
            assert not hasattr(app.state, "bge_model")
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]


def test_score_end_to_end_with_bge_production_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mocked BGE model load + embedding, per this file's own established
    convention of no real model loads outside a @pytest.mark.slow test --
    mirrors tests/evaluation/test_margin.py's mocked score_artifact BGE
    tests, just driven through the real /score HTTP handler instead of
    calling score_artifact directly.
    """
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    expected_version = _register_and_promote_bge_production(client, registry, tmp_path)

    sentinel_model = object()

    def fake_load_bge_model(revision_sha: str) -> object:
        del revision_sha
        return sentinel_model

    def fake_embed_texts(texts: list[str], model: object) -> NDArray[np.float64]:
        assert model is sentinel_model
        return np.zeros((len(texts), 384))

    monkeypatch.setattr("signalscore.serving.app.load_bge_model", fake_load_bge_model)
    monkeypatch.setattr("signalscore.evaluation.margin.embed_texts", fake_embed_texts)

    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with TestClient(app) as test_client:
            response = test_client.post(
                "/score",
                json={
                    "title": "critical outage",
                    "body": "pod crash cluster panic urgent failure",
                    "author_association": "MEMBER",
                },
            )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
        del app.state.bge_model

    assert response.status_code == 200
    body = response.json()
    valid_classes = {p.value for p in Priority}
    assert body["priority_class"] in valid_classes
    assert set(body["priority_probabilities"]) == valid_classes
    assert abs(sum(body["priority_probabilities"].values()) - 1.0) < 1e-6
    assert body["model_version"] == expected_version


@pytest.mark.slow
def test_score_end_to_end_with_real_bge_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No mocking: a real BGE model loads at lifespan startup (the confirmed
    segfault hazard -- see app.py's import-ordering comment) and a real
    LogisticRegression, trained on real BGE embeddings, predicts through the
    real /score handler. If the xgboost-before-torch import order in app.py
    ever regressed, this reliably segfaults the interpreter instead of
    silently passing (same reasoning as
    tests/evaluation/test_margin.py's own real-model BGE test).
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from signalscore.features.embeddings import BGE_REVISION_SHA, embed_texts, load_bge_model
    from signalscore.training.strategies import get_strategy
    from signalscore.training.train_baseline import build_bge_feature_matrix, extract_labels

    rows = make_rows(n_per_class=3)
    real_model = load_bge_model()
    embeddings = embed_texts([row.text for row in rows], real_model)
    x = build_bge_feature_matrix(rows, embeddings)
    y = extract_labels(rows)
    trained_model, extra = get_strategy("logreg").train(x, y)
    classes = [str(c) for c in trained_model.classes_]  # pyright: ignore

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    artifact = BaselineArtifact(
        word_vectorizer=None,
        char_vectorizer=None,
        model=trained_model,
        classes=classes,
        model_type="logreg",
        extra={**extra, "revision_sha": BGE_REVISION_SHA},
        feature_source="bge",
    )
    local_dir = tmp_path / "bge-model"
    local_dir.mkdir()
    artifact_file = local_dir / MODEL_ARTIFACT_FILENAME
    joblib.dump(artifact, artifact_file)  # pyright: ignore[reportUnknownMemberType]
    run = client.create_run(experiment_id="0")
    run_id: str = run.info.run_id
    client.log_artifact(run_id, str(artifact_file), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
    info = registry.register_candidate(run_id=run_id)
    registry.promote(info.version, gate_result="pass (test)", eval_set_tag="eval-set-v1")

    original_tracking_uri = mlflow.get_tracking_uri()  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    try:
        with TestClient(app) as test_client:
            assert app.state.bge_model is not None
            response = test_client.post(
                "/score",
                json={
                    "title": "critical outage",
                    "body": "pod crash cluster panic urgent failure",
                    "author_association": "MEMBER",
                },
            )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
        del app.state.bge_model

    assert response.status_code == 200
    body = response.json()
    valid_classes = {p.value for p in Priority}
    assert body["priority_class"] in valid_classes
    assert set(body["priority_probabilities"]) == valid_classes
    assert abs(sum(body["priority_probabilities"].values()) - 1.0) < 1e-6
    assert body["model_version"] == info.version
