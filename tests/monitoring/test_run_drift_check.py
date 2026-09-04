"""Real objects, no mocking, per repo convention -- same tmp_path-scoped
MlflowClient/ModelRegistry idiom as tests/monitoring/test_reference.py.
"""

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from signalscore.features.schema import FeatureRow
from signalscore.monitoring.buffer import record_live_traffic
from signalscore.monitoring.run_drift_check import run_drift_check
from signalscore.registry import DEFAULT_MODEL_NAME, ModelRegistry
from tests.evaluation._helpers import make_rows
from tests.serving.test_app import (
    _register_and_promote_production,  # pyright: ignore[reportPrivateUsage]
)


def _write_jsonl(rows: list[FeatureRow], path: Path) -> None:
    path.write_text("\n".join(row.model_dump_json() for row in rows))


def _promoted_registry(tmp_path: Path) -> ModelRegistry:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    registry = ModelRegistry(client=client, model_name=DEFAULT_MODEL_NAME)
    _register_and_promote_production(client, registry, tmp_path)
    return registry


def test_run_drift_check_writes_a_report_and_returns_both_results(tmp_path: Path) -> None:
    registry = _promoted_registry(tmp_path)

    train_path = tmp_path / "train.jsonl"
    _write_jsonl(make_rows(n_per_class=5), train_path)
    eval_path = tmp_path / "eval_set_v1.jsonl"
    _write_jsonl(make_rows(n_per_class=5, start_issue=500), eval_path)

    live_traffic_path = tmp_path / "live_traffic.jsonl"
    row = make_rows(n_per_class=1)[0]
    for _ in range(10):
        record_live_traffic(row, predicted_class="P2_backlog", path=live_traffic_path)

    report_dir = tmp_path / "reports"
    input_result, prediction_result = run_drift_check(
        registry,
        train_path=train_path,
        eval_set_path=eval_path,
        live_traffic_path=live_traffic_path,
        report_dir=report_dir,
    )

    assert isinstance(input_result.dataset_drift, bool)
    assert isinstance(prediction_result.dataset_drift, bool)
    assert (report_dir / "latest.json").exists()
    timestamped = list(report_dir.glob("drift_report_*.json"))
    assert len(timestamped) == 1


def test_run_drift_check_raises_without_any_live_traffic(tmp_path: Path) -> None:
    registry = _promoted_registry(tmp_path)

    train_path = tmp_path / "train.jsonl"
    _write_jsonl(make_rows(n_per_class=5), train_path)
    eval_path = tmp_path / "eval_set_v1.jsonl"
    _write_jsonl(make_rows(n_per_class=5, start_issue=500), eval_path)

    with pytest.raises(RuntimeError, match="no live traffic recorded"):
        run_drift_check(
            registry,
            train_path=train_path,
            eval_set_path=eval_path,
            live_traffic_path=tmp_path / "never_written.jsonl",
            report_dir=tmp_path / "reports",
        )
