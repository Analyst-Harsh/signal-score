"""Conformance tests for signalscore.training.strategies -- both real
TrainingStrategy implementations (LogRegStrategy, XGBoostStrategy) exercised
through the same shape of assertions, plus the BaselineArtifact pickle
backward-compat path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD  # pyright: ignore[reportMissingTypeStubs]

from signalscore.features.schema import FeatureRow
from signalscore.training.strategies import (
    STRATEGIES,
    LogRegStrategy,
    TrainedModel,
    XGBoostStrategy,
    get_strategy,
)
from signalscore.training.train_baseline import (
    BaselineArtifact,
    SparseMatrix,
    build_feature_matrix,
    extract_labels,
    fit_vectorizers,
)
from tests.training.test_train_baseline import make_rows

# Small n_components: this synthetic corpus is far smaller than real data, and
# a fitted TruncatedSVD needs a meaningfully-sized reducer for the conformance
# checks below to be representative rather than degenerate.
XGBOOST_TEST_N_COMPONENTS = 5


def _train(
    strategy_name: str,
    x: SparseMatrix,
    y: NDArray[np.str_],
    *,
    x_val: SparseMatrix | None = None,
    y_val: NDArray[np.str_] | None = None,
) -> tuple[TrainedModel, dict[str, Any]]:
    """Trains via the named strategy with test-sized hyperparameters -- a
    thin, explicitly-branched wrapper (not a **kwargs splat) so every real
    keyword argument stays visible to the type checker.
    """
    strategy = get_strategy(strategy_name)
    if strategy_name == "xgboost":
        return strategy.train(
            x, y, x_val=x_val, y_val=y_val, n_components=XGBOOST_TEST_N_COMPONENTS
        )
    return strategy.train(x, y, x_val=x_val, y_val=y_val)


def _build_artifact(strategy_name: str, rows: list[FeatureRow]) -> BaselineArtifact:
    vectorizers = fit_vectorizers([row.text for row in rows])
    x = build_feature_matrix(rows, vectorizers)
    y = extract_labels(rows)
    model, extra = _train(strategy_name, x, y)
    classes = [str(c) for c in model.classes_]  # pyright: ignore
    return BaselineArtifact(
        word_vectorizer=vectorizers.word,
        char_vectorizer=vectorizers.char,
        model=model,
        classes=classes,
        model_type=strategy_name,
        extra=extra,
    )


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_train_returns_a_trained_model_and_expected_extra_shape(strategy_name: str) -> None:
    rows = make_rows(n_per_class=8)
    x = build_feature_matrix(rows, fit_vectorizers([row.text for row in rows]))
    y = extract_labels(rows)

    model, extra = _train(strategy_name, x, y)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "classes_")

    if strategy_name in ("logreg", "logreg_bge"):
        assert set(extra) == {"feature_std"}
    elif strategy_name == "xgboost":
        assert set(extra) == {"reducer"}
        assert isinstance(extra["reducer"], TruncatedSVD)
    else:
        assert extra == {}


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_transform_for_predict_preserves_row_count(strategy_name: str) -> None:
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    vectorizers = fit_vectorizers([row.text for row in train_rows])
    x_train = build_feature_matrix(train_rows, vectorizers)
    x_val = build_feature_matrix(val_rows, vectorizers)
    y_train = extract_labels(train_rows)

    strategy = get_strategy(strategy_name)
    _, extra = _train(strategy_name, x_train, y_train)

    transformed = strategy.transform_for_predict(x_val, extra)

    assert transformed.shape[0] == len(val_rows)


@pytest.mark.parametrize("strategy_name", sorted(STRATEGIES))
def test_feature_importance_shape(strategy_name: str) -> None:
    rows = make_rows(n_per_class=8)
    artifact = _build_artifact(strategy_name, rows)

    if strategy_name.endswith("_bge"):
        if strategy_name == "logreg_bge":
            # BaselineArtifact doesn't have a real `feature_source` field yet
            # (a sibling in-flight change adds it) -- this artifact was built
            # from TF-IDF vectorizers regardless of strategy_name, so the
            # guard needs the field faked here to exercise the BGE path until
            # that field lands for real.
            artifact.feature_source = "bge"  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(NotImplementedError):
            get_strategy(strategy_name).feature_importance(artifact, top_n=5)
        return

    report = get_strategy(strategy_name).feature_importance(artifact, top_n=5)

    if strategy_name == "logreg":
        assert set(report) == set(artifact.classes)
    else:
        assert set(report) == {"overall"}

    for entries in report.values():
        assert entries


def test_xgboost_bge_strategy_trains_on_dense_array_and_round_trips_predict() -> None:
    """BGE artifacts pass a plain dense ndarray (no scipy sparse), unlike
    TF-IDF's sparse design matrix -- exactly the shape XGBoostStrategy's
    _slice_and_densify would crash on (`.toarray()` has no meaning on a plain
    ndarray). XGBoostBgeStrategy skips slicing/SVD entirely, so training and
    the transform_for_predict -> predict/predict_proba round trip must work
    on this shape with no incident.
    """
    rng = np.random.default_rng(42)
    n_per_class = 8
    labels = np.array(
        [c for c in ("P0_critical", "P1_soon", "P2_backlog") for _ in range(n_per_class)]
    )
    n = len(labels)
    x = np.hstack([rng.normal(size=(n, 384)), rng.integers(0, 2, size=(n, 1)).astype(np.float64)])

    strategy = get_strategy("xgboost_bge")
    model, extra = strategy.train(x, labels)

    assert extra == {}
    transformed = strategy.transform_for_predict(x, extra)
    assert transformed is x  # identity -- no reducer to apply

    predictions = model.predict(transformed)
    assert predictions.shape == (n,)
    assert set(predictions) <= set(labels)

    probabilities = model.predict_proba(transformed)
    assert probabilities.shape == (n, 3)


def test_xgboost_bge_strategy_feature_importance_raises_not_implemented() -> None:
    rows = make_rows(n_per_class=8)
    artifact = _build_artifact("xgboost_bge", rows)

    with pytest.raises(NotImplementedError):
        get_strategy("xgboost_bge").feature_importance(artifact, top_n=5)


def test_get_strategy_bogus_raises_value_error_naming_valid_choices() -> None:
    with pytest.raises(ValueError, match="logreg") as exc_info:
        get_strategy("bogus")

    assert "xgboost" in str(exc_info.value)


def test_strategy_name_class_vars_match_registry_keys() -> None:
    assert LogRegStrategy.name == "logreg"
    assert XGBoostStrategy.name == "xgboost"
    assert STRATEGIES["logreg"].name == "logreg"
    assert STRATEGIES["xgboost"].name == "xgboost"


def test_baseline_artifact_setstate_backfills_model_type_and_extra_for_old_pickles() -> None:
    """Regression guard for the currently-registered production baseline_v0
    pickle: an old-shaped state dict has NO model_type/extra key at all (not
    "missing with the default applied"), since pickle restores dataclass
    instances via __dict__.update(), bypassing __init__ defaults entirely.
    __setstate__ is the real code path pickle invokes on unpickling -- called
    directly here, no need to round-trip through actual pickle bytes.
    """
    old_feature_std = np.array([1.0, 2.0, 3.0])
    old_state: dict[str, Any] = {
        "word_vectorizer": object(),
        "char_vectorizer": object(),
        "model": object(),
        "classes": ["P0_critical", "P1_soon", "P2_backlog"],
        "feature_std": old_feature_std,
    }
    assert "model_type" not in old_state
    assert "extra" not in old_state

    obj = BaselineArtifact.__new__(BaselineArtifact)
    obj.__setstate__(dict(old_state))

    assert obj.model_type == "logreg"
    assert obj.extra["feature_std"] is old_feature_std
    assert set(obj.extra) == {"feature_std"}


def test_baseline_artifact_setstate_handles_a_pickle_with_no_feature_std_either() -> None:
    """An even older/degenerate state with no feature_std at all must still
    backfill to an empty extra dict, not raise a KeyError.
    """
    old_state: dict[str, Any] = {
        "word_vectorizer": object(),
        "char_vectorizer": object(),
        "model": object(),
        "classes": ["P0_critical", "P1_soon", "P2_backlog"],
        "feature_std": None,
    }

    obj = BaselineArtifact.__new__(BaselineArtifact)
    obj.__setstate__(dict(old_state))

    assert obj.model_type == "logreg"
    assert obj.extra == {}
