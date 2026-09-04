"""Real Evidently reports against small synthetic DataFrames -- no mocking,
per repo convention. Evidently's text-column default method is model-based
(trains a small domain classifier per run), so these stay fast by keeping
row counts small rather than by faking the library out.
"""

import pandas as pd

from signalscore.monitoring.drift import DriftChecker
from signalscore.monitoring.reference import INPUT_DRIFT_COLUMNS

_STABLE_TEXTS = [
    "pod crash cluster panic urgent failure",
    "node not ready kubelet unresponsive",
    "container restart loop backoff error",
    "scheduler failed to bind pod to node",
]


def _stable_input_frame(n: int = 30) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": [_STABLE_TEXTS[i % len(_STABLE_TEXTS)] for i in range(n)],
            "is_member_plus": [True, False] * (n // 2),
            "title_chars": [20] * n,
            "body_chars": [80] * n,
            "body_words": [12] * n,
            "has_code_block": [False] * n,
            "has_stack_trace": [True, False] * (n // 2),
            "has_url": [False] * n,
            "has_logline": [False] * n,
            "has_excl": [True] * n,
            "code_ratio": [0.05] * n,
            "any_lex": [True] * n,
            "is_ci_flake_shaped": [False] * n,
        }
    )
    return frame.loc[:, INPUT_DRIFT_COLUMNS]


_SHIFTED_TEXTS = [
    "totally unrelated cooking recipe pasta",
    "weather forecast sunny skies today",
    "sports scores update final results",
    "movie review new release opinion",
]


def _shifted_input_frame(n: int = 30) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": [_SHIFTED_TEXTS[i % len(_SHIFTED_TEXTS)] for i in range(n)],
            "is_member_plus": [False] * n,
            "title_chars": [400] * n,
            "body_chars": [3000] * n,
            "body_words": [600] * n,
            "has_code_block": [True] * n,
            "has_stack_trace": [False] * n,
            "has_url": [True] * n,
            "has_logline": [True] * n,
            "has_excl": [False] * n,
            "code_ratio": [0.95] * n,
            "any_lex": [False] * n,
            "is_ci_flake_shaped": [True] * n,
        }
    )
    return frame.loc[:, INPUT_DRIFT_COLUMNS]


def _prediction_frame(classes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"predicted_class": classes})


def test_check_input_drift_flags_a_clearly_shifted_distribution() -> None:
    checker = DriftChecker(
        input_reference=_stable_input_frame(),
        prediction_reference=_prediction_frame(["P2_backlog"] * 30),
    )

    result = checker.check_input_drift(_shifted_input_frame())

    assert result.dataset_drift is True
    assert result.share_drifted_columns > 0.5
    assert "code_ratio" in result.drifted_columns
    assert "title_chars" in result.drifted_columns


def test_check_input_drift_does_not_flag_the_same_distribution() -> None:
    reference = _stable_input_frame()
    checker = DriftChecker(
        input_reference=reference,
        prediction_reference=_prediction_frame(["P2_backlog"] * 30),
    )

    result = checker.check_input_drift(_stable_input_frame())

    assert result.dataset_drift is False
    assert result.drifted_columns == []


def test_check_prediction_drift_flags_a_shifted_class_distribution() -> None:
    checker = DriftChecker(
        input_reference=_stable_input_frame(),
        prediction_reference=_prediction_frame(["P2_backlog"] * 30),
    )

    result = checker.check_prediction_drift(_prediction_frame(["P0_critical"] * 30))

    assert result.dataset_drift is True
    assert "predicted_class" in result.drifted_columns


def test_check_prediction_drift_does_not_flag_the_same_class_distribution() -> None:
    checker = DriftChecker(
        input_reference=_stable_input_frame(),
        prediction_reference=_prediction_frame(["P2_backlog"] * 30),
    )

    result = checker.check_prediction_drift(_prediction_frame(["P2_backlog"] * 30))

    assert result.dataset_drift is False
    assert result.drifted_columns == []
