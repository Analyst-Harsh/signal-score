"""Evidently drift checks -- input drift and prediction drift.

Stateful wrapper around Evidently's Report/Dataset API, same tier as
`ModelRegistry` for MLflow: it holds both reference frames
(`monitoring.reference`) across two methods (`check_input_drift`,
`check_prediction_drift`), which is what earns it a class rather than a bare
function per call site (docs/design-patterns-guide.md's Adapter rule) --
today it has one real caller (`run_drift_check.py`), not several, so the
class is justified by the cross-method state it holds, not by
`ModelRegistry`'s separate multi-module-caller reason for being one.

Evidently's own dataset-level `DriftedColumnsCount` metric is the
authoritative `dataset_drift`/`share_drifted_columns` signal -- Evidently
already applies the correct per-column-type drift decision internally to
compute it (verified against a real Evidently report, not assumed from
docs). Per-column `drifted_columns` is derived here from each column's own
`ValueDrift` result using Evidently's two metric families' standard
convention -- see `_is_column_drifted`'s docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Evidently's dataset-level drift alarm fires once this share of columns has
# drifted -- 0.5 is Evidently's own documented default; named here so it's a
# visible, tunable constant rather than an opaque library default. The one
# knob to turn if this repo's class balance ever calls for a
# stricter/looser bound.
DATASET_DRIFT_SHARE_THRESHOLD = 0.5

_TEXT_COLUMN = "text"
_CATEGORICAL_COLUMNS = [
    "is_member_plus",
    "has_code_block",
    "has_stack_trace",
    "has_url",
    "has_logline",
    "has_excl",
    "any_lex",
    "is_ci_flake_shaped",
]
_NUMERICAL_COLUMNS = ["title_chars", "body_chars", "body_words", "code_ratio"]


def _input_data_definition() -> DataDefinition:
    return DataDefinition(
        text_columns=[_TEXT_COLUMN],
        categorical_columns=_CATEGORICAL_COLUMNS,
        numerical_columns=_NUMERICAL_COLUMNS,
    )


def _prediction_data_definition() -> DataDefinition:
    return DataDefinition(categorical_columns=["predicted_class"])


@dataclass(frozen=True)
class DriftResult:
    dataset_drift: bool
    share_drifted_columns: float
    drifted_columns: list[str]
    checked_at: datetime


def _is_column_drifted(value: float, method: str, threshold: float) -> bool:
    """p-value methods (K-S, Z-test/chi-squared -- Evidently's numerical and
    categorical defaults) signal drift when `value` falls BELOW `threshold`,
    rejecting the "same distribution" null hypothesis. Score methods (PSI,
    Jensen-Shannon, the text column's domain-classifier default) signal
    drift when `value` rises AT OR ABOVE `threshold`. Evidently doesn't
    expose a per-column boolean directly in a Report's `.dict()` output
    (checked against a real run's `metric_results`, not assumed) -- this is
    the standard, documented convention for exactly the two metric families
    its default methods fall into.
    """
    if "p_value" in method.lower():
        return value < threshold
    return value >= threshold


def _run_report(
    current: pd.DataFrame, reference: pd.DataFrame, definition: DataDefinition
) -> DriftResult:
    # evidently ships no py.typed marker for Dataset.from_pandas/Report.dict --
    # isolated behind these ignores rather than fighting the variance, same
    # convention as train_baseline.py's own scipy-sparse boundary.
    ds_current = Dataset.from_pandas(current, data_definition=definition)  # pyright: ignore[reportUnknownMemberType]
    ds_reference = Dataset.from_pandas(reference, data_definition=definition)  # pyright: ignore[reportUnknownMemberType]

    report = Report([DataDriftPreset(drift_share=DATASET_DRIFT_SHARE_THRESHOLD)])
    result: dict[str, Any] = report.run(ds_current, ds_reference).dict()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    drifted_columns: list[str] = []
    dataset_drift = False
    share_drifted_columns = 0.0
    for metric in result["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            share_drifted_columns = metric["value"]["share"]
            dataset_drift = share_drifted_columns >= DATASET_DRIFT_SHARE_THRESHOLD
        elif name.startswith("ValueDrift"):
            config = metric["config"]
            if _is_column_drifted(metric["value"], config["method"], config["threshold"]):
                drifted_columns.append(config["column"])

    return DriftResult(
        dataset_drift=dataset_drift,
        share_drifted_columns=share_drifted_columns,
        drifted_columns=sorted(drifted_columns),
        checked_at=datetime.now(UTC),
    )


class DriftChecker:
    """Every Evidently `Report` call in this repo goes through here -- same
    tier as `ModelRegistry` for MLflow.
    """

    def __init__(self, input_reference: pd.DataFrame, prediction_reference: pd.DataFrame) -> None:
        self._input_reference = input_reference
        self._prediction_reference = prediction_reference

    def check_input_drift(self, current: pd.DataFrame) -> DriftResult:
        return _run_report(current, self._input_reference, _input_data_definition())

    def check_prediction_drift(self, current: pd.DataFrame) -> DriftResult:
        return _run_report(
            current.loc[:, ["predicted_class"]],
            self._prediction_reference.loc[:, ["predicted_class"]],
            _prediction_data_definition(),
        )
