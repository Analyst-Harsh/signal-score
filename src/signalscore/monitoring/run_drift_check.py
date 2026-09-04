"""CLI entry point: input + prediction drift check against live traffic.

Run via:
    uv run python -m signalscore.monitoring.run_drift_check

Loads both references (`monitoring.reference`), loads the live-traffic
buffer (`monitoring.buffer`), runs both checks
(`monitoring.drift.DriftChecker`), writes a JSON summary under
`data/monitoring/`, and exits non-zero when either check flags dataset-level
drift -- the signal a future CI job (weekly or drift-triggered retrain, per
docs/signalscore-design.md's architecture section) would key off. Wiring
that job itself is out of scope here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow

from signalscore.monitoring.buffer import LIVE_TRAFFIC_PATH, load_live_traffic
from signalscore.monitoring.drift import DriftChecker, DriftResult
from signalscore.monitoring.reference import (
    EVAL_SET_PATH,
    TRAIN_SET_PATH,
    load_input_reference,
    load_prediction_reference,
)
from signalscore.registry import ModelRegistry
from signalscore.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = _REPO_ROOT / "data/monitoring"


def _result_to_dict(result: DriftResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["checked_at"] = result.checked_at.isoformat()
    return payload


def run_drift_check(
    registry: ModelRegistry,
    train_path: Path = TRAIN_SET_PATH,
    eval_set_path: Path = EVAL_SET_PATH,
    live_traffic_path: Path = LIVE_TRAFFIC_PATH,
    report_dir: Path = REPORT_DIR,
) -> tuple[DriftResult, DriftResult]:
    """`registry` is a required parameter (never constructed internally) so
    tests can inject a tmp_path-scoped MlflowClient, matching
    `monitoring.reference.load_prediction_reference`'s own convention.
    """
    # Checked before building either reference: a fresh/low-traffic
    # deployment shouldn't pay for scoring the whole eval set (a real
    # artifact load, and for a BGE model a real embedding pass) only to
    # then discover there's nothing to compare it against.
    live_traffic = load_live_traffic(live_traffic_path)
    if live_traffic.empty:
        raise RuntimeError(
            f"no live traffic recorded yet at {live_traffic_path} -- "
            "score at least one /score request before running a drift check"
        )

    input_reference = load_input_reference(train_path)
    prediction_reference = load_prediction_reference(registry, eval_set_path)
    checker = DriftChecker(input_reference, prediction_reference)

    input_result = checker.check_input_drift(live_traffic)
    prediction_result = checker.check_prediction_drift(live_traffic)

    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "input_drift": _result_to_dict(input_result),
        "prediction_drift": _result_to_dict(prediction_result),
    }
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    (report_dir / f"drift_report_{timestamp}.json").write_text(json.dumps(report, indent=2))
    (report_dir / "latest.json").write_text(json.dumps(report, indent=2))

    return input_result, prediction_result


def main() -> None:
    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
    input_result, prediction_result = run_drift_check(ModelRegistry())

    print(
        json.dumps(
            {
                "input_drift": _result_to_dict(input_result),
                "prediction_drift": _result_to_dict(prediction_result),
            },
            indent=2,
        )
    )
    if input_result.dataset_drift or prediction_result.dataset_drift:
        sys.exit(1)


if __name__ == "__main__":
    main()
