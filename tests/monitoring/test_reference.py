"""Real objects, no mocking, per repo convention -- same idiom as
tests/serving/test_app.py: a real tmp_path-scoped MlflowClient/ModelRegistry,
a real logged+promoted BaselineArtifact, real FeatureRow JSONL files on disk.
"""

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.monitoring.reference import (
    INPUT_DRIFT_COLUMNS,
    load_input_reference,
    load_prediction_reference,
)
from signalscore.registry import DEFAULT_MODEL_NAME, ModelRegistry
from tests.evaluation._helpers import make_rows
from tests.serving.test_app import (
    _register_and_promote_production,  # pyright: ignore[reportPrivateUsage]
)


def _write_jsonl(rows: list[FeatureRow], path: Path) -> None:
    path.write_text("\n".join(row.model_dump_json() for row in rows))


def test_load_input_reference_selects_the_drift_monitoring_columns(tmp_path: Path) -> None:
    rows = make_rows(n_per_class=2)
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(rows, train_path)

    frame = load_input_reference(train_path)

    assert list(frame.columns) == INPUT_DRIFT_COLUMNS
    assert len(frame) == len(rows)
    # label/created_at/issue_number must never leak into the drift-input
    # frame -- live /score traffic never populates them with real signal.
    assert "label" not in frame.columns
    assert "created_at" not in frame.columns
    assert "issue_number" not in frame.columns


def test_load_prediction_reference_scores_eval_set_through_production_model(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    _register_and_promote_production(client, registry, tmp_path)

    eval_rows = make_rows(n_per_class=3, start_issue=1000)
    eval_path = tmp_path / "eval_set_v1.jsonl"
    _write_jsonl(eval_rows, eval_path)

    frame = load_prediction_reference(registry, eval_set_path=eval_path)

    assert len(frame) == len(eval_rows)
    valid_classes = {p.value for p in Priority}
    assert set(frame["predicted_class"]).issubset(valid_classes)
    proba_columns = [c for c in frame.columns if c.startswith("proba_")]
    assert len(proba_columns) == len(valid_classes)
    row_sums = frame[proba_columns].sum(axis=1)
    assert ((row_sums - 1.0).abs() < 1e-6).all()


def test_load_prediction_reference_raises_without_a_production_model(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"  # fresh, empty registry
    registry = ModelRegistry(client=MlflowClient(tracking_uri=tracking_uri))

    with pytest.raises(RuntimeError, match="no production-tagged model"):
        load_prediction_reference(registry, eval_set_path=tmp_path / "unused.jsonl")
