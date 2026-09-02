"""Tests for signalscore.evaluation.margin.

Uses duck-typed real objects with hardcoded predict/predict_proba output
(never unittest.mock) so each check -- bootstrap PR-AUC, minority-F1 floor,
Brier cap -- can be exercised deterministically, per repo convention (see
contract.py's `_SlowModel`/`_InvalidProbaModel` for the same pattern).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from signalscore.evaluation.gate import GateContext
from signalscore.evaluation.margin import (
    MAX_BRIER_REGRESSION,
    _bootstrap_pr_auc_ci_lower,  # pyright: ignore[reportPrivateUsage]
    evaluate_margin,
    score_artifact,
)
from signalscore.features.labels import Priority
from signalscore.training.train_baseline import BaselineArtifact, fit_vectorizers
from tests.evaluation._helpers import build_artifact, make_rows

if TYPE_CHECKING:
    # Type-only: importing sentence_transformers/torch for real here would risk the
    # xgboost-before-torch import-ordering hazard documented in margin.py -- this
    # block never executes at runtime.
    from sentence_transformers import SentenceTransformer

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


# --- score_artifact: BGE branch ---------------------------------------------


def _bge_artifact(
    y_pred: list[str], y_proba: NDArray[np.float64], revision_sha: str
) -> BaselineArtifact:
    model = _FixedModel(np.array(y_pred), y_proba)
    return BaselineArtifact(
        word_vectorizer=None,
        char_vectorizer=None,
        model=model,
        classes=CLASSES,
        model_type="logreg",
        extra={"revision_sha": revision_sha},
        feature_source="bge",
    )


def test_score_artifact_bge_uses_passed_bge_model_and_never_loads_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = make_rows(n_per_class=2)  # 6 rows
    true_labels = [row.label.value for row in rows]
    proba = np.tile(np.array([0.5, 0.3, 0.2]), (len(rows), 1))
    artifact = _bge_artifact(true_labels, proba, revision_sha="unused-when-model-passed")
    sentinel_model = cast("SentenceTransformer", object())
    embed_calls: list[tuple[list[str], object]] = []

    def fake_embed_texts(texts: list[str], model: object) -> NDArray[np.float64]:
        embed_calls.append((texts, model))
        return np.zeros((len(texts), 384))

    def fake_load_bge_model(revision_sha: str) -> object:
        raise AssertionError(f"load_bge_model must not be called (got {revision_sha!r})")

    monkeypatch.setattr("signalscore.evaluation.margin.embed_texts", fake_embed_texts)
    monkeypatch.setattr("signalscore.evaluation.margin.load_bge_model", fake_load_bge_model)

    y_true, y_pred, y_proba, classes = score_artifact(artifact, rows, bge_model=sentinel_model)

    assert len(embed_calls) == 1
    called_texts, called_model = embed_calls[0]
    assert called_texts == [row.text for row in rows]
    assert called_model is sentinel_model
    assert list(y_true) == true_labels
    assert classes == CLASSES
    assert y_pred.shape == (len(rows),)
    assert y_proba.shape == (len(rows), 3)


def test_score_artifact_bge_lazily_loads_model_when_not_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = make_rows(n_per_class=2)
    true_labels = [row.label.value for row in rows]
    proba = np.tile(np.array([0.5, 0.3, 0.2]), (len(rows), 1))
    artifact = _bge_artifact(true_labels, proba, revision_sha="my-pinned-revision")
    sentinel_model = object()
    load_calls: list[str] = []

    def fake_load_bge_model(revision_sha: str) -> object:
        load_calls.append(revision_sha)
        return sentinel_model

    def fake_embed_texts(texts: list[str], model: object) -> NDArray[np.float64]:
        assert model is sentinel_model
        return np.zeros((len(texts), 384))

    monkeypatch.setattr("signalscore.evaluation.margin.load_bge_model", fake_load_bge_model)
    monkeypatch.setattr("signalscore.evaluation.margin.embed_texts", fake_embed_texts)

    score_artifact(artifact, rows)

    assert load_calls == ["my-pinned-revision"]


@pytest.mark.slow
def test_score_artifact_bge_real_model_end_to_end() -> None:
    """No mocking: a real BGE model embeds real rows and a real LogisticRegression
    predicts on the result -- proves the BGE branch's real wiring (embed_texts's
    actual shape, build_bge_feature_matrix's actual hstack) works end to end, not
    just against the mocked shapes above.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from signalscore.features.embeddings import BGE_REVISION_SHA, embed_texts, load_bge_model
    from signalscore.training.strategies import get_strategy
    from signalscore.training.train_baseline import build_bge_feature_matrix, extract_labels

    rows = make_rows(n_per_class=3)
    model = load_bge_model()
    embeddings = embed_texts([row.text for row in rows], model)
    x = build_bge_feature_matrix(rows, embeddings)
    y = extract_labels(rows)
    trained_model, extra = get_strategy("logreg").train(x, y)
    classes = [str(c) for c in trained_model.classes_]  # pyright: ignore
    artifact = BaselineArtifact(
        word_vectorizer=None,
        char_vectorizer=None,
        model=trained_model,
        classes=classes,
        model_type="logreg",
        extra={**extra, "revision_sha": BGE_REVISION_SHA},
        feature_source="bge",
    )

    y_true, y_pred, y_proba, out_classes = score_artifact(artifact, rows, bge_model=model)

    assert y_true.shape == (len(rows),)
    assert y_pred.shape == (len(rows),)
    assert y_proba.shape == (len(rows), len(out_classes))


@pytest.mark.slow
def test_score_artifact_xgboost_then_bge_in_same_process_does_not_segfault() -> None:
    """Confirmed hazard: importing sentence_transformers/torch before xgboost
    in the same process segfaults the interpreter with no Python traceback on
    this machine. Runs a fresh subprocess that imports `margin` (which pulls in
    both xgboost, via train_baseline's guarded first-import, and
    sentence_transformers/torch, via the embeddings import added below it),
    then exercises score_artifact() against a real XGBoost TF-IDF artifact
    AND a real BGE artifact in that same process -- if the import order in
    margin.py ever regresses, this reliably reproduces the crash instead of
    silently passing.
    """
    script = """
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from signalscore.evaluation.margin import score_artifact
from signalscore.features.embeddings import BGE_REVISION_SHA, embed_texts, load_bge_model
from signalscore.training.strategies import get_strategy
from signalscore.training.train_baseline import (
    BaselineArtifact,
    build_bge_feature_matrix,
    build_feature_matrix,
    extract_labels,
    fit_vectorizers,
)
from tests.evaluation._helpers import make_rows

rows = make_rows(n_per_class=6)
texts = [row.text for row in rows]
y = extract_labels(rows)

vectorizers = fit_vectorizers(texts)
x = build_feature_matrix(rows, vectorizers)
xgb_model, xgb_extra = get_strategy("xgboost").train(x, y, n_components=5)
xgb_classes = [str(c) for c in xgb_model.classes_]
xgb_artifact = BaselineArtifact(
    word_vectorizer=vectorizers.word,
    char_vectorizer=vectorizers.char,
    model=xgb_model,
    classes=xgb_classes,
    model_type="xgboost",
    extra=xgb_extra,
)
score_artifact(xgb_artifact, rows)

bge_model = load_bge_model()
embeddings = embed_texts(texts, bge_model)
x_bge = build_bge_feature_matrix(rows, embeddings)
logreg_model, logreg_extra = get_strategy("logreg").train(x_bge, y)
bge_classes = [str(c) for c in logreg_model.classes_]
bge_artifact = BaselineArtifact(
    word_vectorizer=None,
    char_vectorizer=None,
    model=logreg_model,
    classes=bge_classes,
    model_type="logreg",
    extra={**logreg_extra, "revision_sha": BGE_REVISION_SHA},
    feature_source="bge",
)
score_artifact(bge_artifact, rows, bge_model=bge_model)

print("SMOKE_TEST_OK")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"subprocess crashed (returncode={result.returncode}) -- likely the "
        f"xgboost/torch import-order segfault.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Fatal Python error" not in result.stderr
    assert "SMOKE_TEST_OK" in result.stdout
