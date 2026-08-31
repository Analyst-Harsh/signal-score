"""Tests for signalscore.evaluation.margin.

Uses duck-typed real objects with hardcoded predict/predict_proba output
(never unittest.mock) so each check -- bootstrap PR-AUC, minority-F1 floor,
Brier cap -- can be exercised deterministically, per repo convention (see
contract.py's `_SlowModel`/`_InvalidProbaModel` for the same pattern).
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from signalscore.evaluation.gate import GateContext
from signalscore.evaluation.margin import (
    MAX_BRIER_REGRESSION,
    _bootstrap_pr_auc_ci_lower,  # pyright: ignore[reportPrivateUsage]
    evaluate_margin,
)
from signalscore.features.labels import Priority
from signalscore.training.train_baseline import BaselineArtifact, fit_vectorizers
from tests.evaluation._helpers import build_artifact, make_rows

CLASSES = [Priority.P0_CRITICAL.value, Priority.P1_SOON.value, Priority.P2_BACKLOG.value]


class _FixedModel:
    """A real object (not a mock) with hardcoded predict/predict_proba output,
    so tests can pin exact scored outcomes instead of depending on real
    training producing a particular metric value.
    """

    def __init__(self, y_pred: NDArray[np.str_], y_proba: NDArray[np.float64]) -> None:
        self._y_pred = y_pred
        self._y_proba = y_proba
        self.classes_ = np.array(CLASSES)

    def predict(self, x: object) -> NDArray[np.str_]:
        del x
        return self._y_pred

    def predict_proba(self, x: object) -> NDArray[np.float64]:
        del x
        return self._y_proba


def _fixed_artifact(
    texts: list[str], y_pred: list[str], y_proba: NDArray[np.float64]
) -> BaselineArtifact:
    vectorizers = fit_vectorizers(texts)
    model = _FixedModel(np.array(y_pred), y_proba)
    return BaselineArtifact(
        word_vectorizer=vectorizers.word,
        char_vectorizer=vectorizers.char,
        model=model,
        classes=CLASSES,
        model_type="logreg",
        extra={"feature_std": np.zeros(1)},
    )


def test_evaluate_margin_returns_none_when_no_production() -> None:
    rows = make_rows(n_per_class=6)
    candidate = build_artifact(rows)
    ctx = GateContext(candidate=candidate, production=None, eval_rows=rows)

    assert evaluate_margin(ctx) is None


def _strong_proba(n: int) -> NDArray[np.float64]:
    """Perfectly ranked -- own class always scores highest -- so PR-AUC is 1.0
    regardless of how a resample mixes rows across classes.
    """
    return np.array([[0.9, 0.05, 0.05]] * n + [[0.05, 0.9, 0.05]] * n + [[0.05, 0.05, 0.9]] * n)


def _confused_proba(n: int) -> NDArray[np.float64]:
    """A genuinely worse ranker: most rows favor the true class, but a real
    minority in every class block score a WRONG class highest -- an actual
    ranking mistake, unlike merely being less confident (which
    average_precision_score doesn't penalize at all, since it only depends on
    rank order, not score magnitude).
    """
    n_wrong = 3
    n_correct = n - n_wrong

    def block(correct: list[float], wrong: list[float]) -> list[list[float]]:
        return [correct] * n_correct + [wrong] * n_wrong

    return np.array(
        block([0.6, 0.25, 0.15], [0.2, 0.6, 0.2])  # P0 block, wrong favors P1
        + block([0.15, 0.6, 0.25], [0.6, 0.2, 0.2])  # P1 block, wrong favors P0
        + block([0.15, 0.25, 0.6], [0.6, 0.2, 0.2])  # P2 block, wrong favors P0
    )


def test_evaluate_margin_passes_when_candidate_clearly_beats_production() -> None:
    n = 7
    rows = make_rows(n_per_class=n)  # 21 rows, ordered P0 block, P1 block, P2 block
    texts = [row.text for row in rows]
    true_labels = [c for c in CLASSES for _ in range(n)]

    strong_proba = _strong_proba(n)
    weak_proba = _confused_proba(n)

    candidate = _fixed_artifact(texts, true_labels, strong_proba)
    production = _fixed_artifact(texts, true_labels, weak_proba)
    ctx = GateContext(candidate=candidate, production=production, eval_rows=rows)

    assert evaluate_margin(ctx, n_bootstrap=200) is None


def test_evaluate_margin_fails_when_candidate_ties_production() -> None:
    n = 7
    rows = make_rows(n_per_class=n)
    texts = [row.text for row in rows]
    true_labels = [c for c in CLASSES for _ in range(n)]
    proba = np.array([[0.6, 0.2, 0.2]] * n + [[0.2, 0.6, 0.2]] * n + [[0.2, 0.2, 0.6]] * n)

    candidate = _fixed_artifact(texts, true_labels, proba)
    production = _fixed_artifact(texts, true_labels, proba)
    ctx = GateContext(candidate=candidate, production=production, eval_rows=rows)

    result = evaluate_margin(ctx, n_bootstrap=200)

    assert result is not None
    assert "not significant" in result


def test_evaluate_margin_fails_on_minority_f1_regression() -> None:
    n = 7
    rows = make_rows(n_per_class=n)
    texts = [row.text for row in rows]
    true_labels = [c for c in CLASSES for _ in range(n)]
    proba = np.array([[0.9, 0.05, 0.05]] * n + [[0.05, 0.9, 0.05]] * n + [[0.05, 0.05, 0.9]] * n)
    # Candidate mispredicts every P0 (minority-class) row as P1 -- P0 f1 drops
    # to 0 -- while production predicts every row correctly.
    candidate_pred = [CLASSES[1]] * n + [CLASSES[1]] * n + [CLASSES[2]] * n
    production_pred = true_labels

    candidate = _fixed_artifact(texts, candidate_pred, proba)
    production = _fixed_artifact(texts, production_pred, proba)
    ctx = GateContext(candidate=candidate, production=production, eval_rows=rows)

    result = evaluate_margin(ctx, n_bootstrap=200)

    assert result is not None
    assert "minority_f1" in result


def test_evaluate_margin_fails_on_brier_regression() -> None:
    n = 7
    rows = make_rows(n_per_class=n)
    texts = [row.text for row in rows]
    true_labels = [c for c in CLASSES for _ in range(n)]

    production_proba = np.array(
        [[0.95, 0.025, 0.025]] * n + [[0.025, 0.95, 0.025]] * n + [[0.025, 0.025, 0.95]] * n
    )
    # Candidate's predicted labels stay correct (minority_f1 tied), but its
    # probabilities for the P1 block are confidently pointed at the WRONG
    # class -- tanks Brier without touching the hardcoded predict() output.
    candidate_proba = np.array(
        [[0.95, 0.025, 0.025]] * n + [[0.9, 0.05, 0.05]] * n + [[0.025, 0.025, 0.95]] * n
    )

    candidate = _fixed_artifact(texts, true_labels, candidate_proba)
    production = _fixed_artifact(texts, true_labels, production_proba)
    ctx = GateContext(candidate=candidate, production=production, eval_rows=rows)

    result = evaluate_margin(ctx, n_bootstrap=200)

    assert result is not None
    assert "brier_macro" in result
    assert f"{MAX_BRIER_REGRESSION}" in result or "exceeding" in result


# --- _bootstrap_pr_auc_ci_lower: direct unit test of the CI logic ----------


def test_bootstrap_pr_auc_ci_lower_is_positive_for_a_clear_separation() -> None:
    y_true = np.array([c for c in CLASSES for _ in range(7)])
    proba_candidate = _strong_proba(7)
    proba_production = _confused_proba(7)

    ci_lower = _bootstrap_pr_auc_ci_lower(
        y_true, proba_candidate, CLASSES, proba_production, CLASSES, n_bootstrap=200, seed=1
    )

    assert ci_lower > 0


def test_bootstrap_pr_auc_ci_lower_is_zero_for_identical_distributions() -> None:
    y_true = np.array([c for c in CLASSES for _ in range(7)])
    proba = _strong_proba(7)

    ci_lower = _bootstrap_pr_auc_ci_lower(
        y_true, proba, CLASSES, proba, CLASSES, n_bootstrap=200, seed=1
    )

    assert ci_lower == pytest.approx(0.0)
