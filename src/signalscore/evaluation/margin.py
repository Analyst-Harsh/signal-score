"""Head-to-head margin check: candidate vs. current production on the frozen
eval set.

Three checks (docs/mlflow-integration-plan.md design decision 2):

- PR-AUC improvement: a percentile bootstrap CI on resampled eval-row
  indices, requiring the 95% CI lower bound of
  `pr_auc_macro(candidate) - pr_auc_macro(production)` to exceed 0 -- the
  improvement must be significant at the eval set's actual noise level, not
  just a positive point estimate.
- Minority F1: a plain non-regression floor (point estimate, no bootstrap) --
  a downside guard, not an upside claim needing a significance test.
- Brier score: a fixed absolute regression cap -- same downside-guard
  reasoning.

Both artifacts are scored with their own fitted vectorizers -- verified
against `train_baseline.py` as exactly how training already works, so the
comparison is apples-to-apples.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)
from sklearn.preprocessing import (
    label_binarize,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)

from signalscore.features.schema import FeatureRow
from signalscore.training.train_baseline import (
    BaselineArtifact,
    FittedVectorizers,
    build_feature_matrix,
    compute_metrics,
    extract_labels,
)

if TYPE_CHECKING:
    from signalscore.evaluation.gate import GateContext

N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42
_CI_PERCENTILE_LOWER = 2.5  # 95% CI lower bound
MAX_BRIER_REGRESSION = 0.01


def _score_artifact(
    artifact: BaselineArtifact, rows: list[FeatureRow]
) -> tuple[NDArray[np.str_], NDArray[np.str_], NDArray[np.float64], list[str]]:
    """Transforms `rows` with `artifact`'s OWN fitted vectorizers (never the
    other artifact's) and returns (y_true, y_pred, y_proba, classes).
    """
    vectorizers = FittedVectorizers(word=artifact.word_vectorizer, char=artifact.char_vectorizer)
    x = build_feature_matrix(rows, vectorizers)
    y_true = extract_labels(rows)
    y_pred: NDArray[np.str_] = artifact.model.predict(x)  # pyright: ignore
    y_proba: NDArray[np.float64] = artifact.model.predict_proba(x)  # pyright: ignore
    return y_true, y_pred, y_proba, artifact.classes  # pyright: ignore[reportUnknownVariableType]


def _pr_auc_macro(
    y_true: NDArray[np.str_], y_proba: NDArray[np.float64], classes: list[str]
) -> float:
    y_true_binarized = label_binarize(y_true, classes=classes)  # pyright: ignore
    return float(
        average_precision_score(y_true_binarized, y_proba, average="macro")  # pyright: ignore
    )


def _bootstrap_pr_auc_ci_lower(
    y_true: NDArray[np.str_],
    proba_candidate: NDArray[np.float64],
    classes_candidate: list[str],
    proba_production: NDArray[np.float64],
    classes_production: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> float:
    """95% CI lower bound of
    `pr_auc_macro(candidate) - pr_auc_macro(production)` via a percentile
    bootstrap over resampled eval-row indices (with replacement). Needs no
    model refitting -- only resampling predictions already computed once.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(y_true)
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_rows, size=n_rows)
        candidate_auc = _pr_auc_macro(y_true[idx], proba_candidate[idx], classes_candidate)
        production_auc = _pr_auc_macro(y_true[idx], proba_production[idx], classes_production)
        deltas[i] = candidate_auc - production_auc
    return float(np.percentile(deltas, _CI_PERCENTILE_LOWER))


def evaluate_margin(ctx: GateContext) -> str | None:
    """None (pass) if `ctx.production` is None (trivial pass -- see
    `gate.MarginStep`) or if all three checks pass; otherwise a failure
    string identifying which check(s) failed and by how much.
    """
    if ctx.production is None:
        return None

    y_true, y_pred_candidate, proba_candidate, classes_candidate = _score_artifact(
        ctx.candidate, ctx.eval_rows
    )
    _, y_pred_production, proba_production, classes_production = _score_artifact(
        ctx.production, ctx.eval_rows
    )

    metrics_candidate = compute_metrics(
        y_true, y_pred_candidate, proba_candidate, classes_candidate
    )
    metrics_production = compute_metrics(
        y_true, y_pred_production, proba_production, classes_production
    )

    failures: list[str] = []

    ci_lower = _bootstrap_pr_auc_ci_lower(
        y_true, proba_candidate, classes_candidate, proba_production, classes_production
    )
    if ci_lower <= 0:
        failures.append(
            f"pr_auc_macro improvement not significant at 95% CI (lower bound {ci_lower:.4f} <= 0)"
        )

    minority_f1_candidate = metrics_candidate["minority_f1"]
    minority_f1_production = metrics_production["minority_f1"]
    if minority_f1_candidate < minority_f1_production:
        failures.append(
            f"minority_f1 regressed: candidate {minority_f1_candidate:.4f} < "
            f"production {minority_f1_production:.4f}"
        )

    brier_regression = metrics_candidate["brier_macro"] - metrics_production["brier_macro"]
    if brier_regression > MAX_BRIER_REGRESSION:
        failures.append(
            f"brier_macro regressed by {brier_regression:.4f}, "
            f"exceeding the {MAX_BRIER_REGRESSION} cap"
        )

    if failures:
        return "; ".join(failures)
    return None
