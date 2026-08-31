"""TrainingStrategy seam -- one model family per class, dispatched by a plain
dict (not a Factory class: docs/design-patterns-guide.md sets the Factory
Method trigger at >=3 real variants; there are exactly 2 today).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD  # pyright: ignore[reportMissingTypeStubs]
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import (  # pyright: ignore[reportMissingTypeStubs]
    compute_class_weight,  # pyright: ignore[reportUnknownVariableType]
    compute_sample_weight,  # pyright: ignore[reportUnknownVariableType]
)
from xgboost import XGBClassifier  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from signalscore.training.train_baseline import BaselineArtifact

SparseMatrix = Any

DEFAULT_C = 1.0
DEFAULT_PENALTY = "l2"
DEFAULT_SVD_COMPONENTS = 100
DEFAULT_N_ESTIMATORS = 1000
DEFAULT_EARLY_STOPPING_ROUNDS = 30


class TrainedModel(Protocol):
    """Structural subset of the sklearn classifier API actually called
    downstream (contract.py/margin.py/serving all only ever call predict/
    predict_proba on artifact.model; classes_ is read exactly once, right
    after fit, inside train_and_evaluate). LogisticRegression and
    XGBClassifier both already satisfy this with zero wrapper code."""

    classes_: NDArray[np.str_]

    def predict(self, x: SparseMatrix) -> NDArray[np.str_]: ...
    def predict_proba(self, x: SparseMatrix) -> NDArray[np.float64]: ...


class TrainingStrategy(Protocol):
    name: ClassVar[str]

    def train(
        self,
        x_train: SparseMatrix,
        y_train: NDArray[np.str_],
        *,
        x_val: SparseMatrix | None = None,
        y_val: NDArray[np.str_] | None = None,
        **hyperparams: Any,
    ) -> tuple[TrainedModel, dict[str, Any]]:
        """Returns the fitted model + a strategy-owned `extra` payload persisted
        on the artifact (LogReg: {"feature_std": ...}; XGBoost: {"reducer": fitted_svd}).
        x_val/y_val are optional -- LogReg ignores them; XGBoost uses them as an
        early-stopping eval set."""
        ...

    def transform_for_predict(self, x: SparseMatrix, extra: dict[str, Any]) -> SparseMatrix:
        """Identity for LogReg. The ONE choke point between build_feature_matrix()
        and model.predict()/predict_proba() -- exists so a dense/SVD representation
        change never requires touching contract.py/margin.py/serving again."""
        ...

    def feature_importance(
        self, artifact: BaselineArtifact, top_n: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Per-class breakdown for LogReg (real class names as keys); a single
        "overall" key for XGBoost (multi:softprob has no native per-class
        importance -- do not fake one)."""
        ...


def combined_feature_names(artifact: BaselineArtifact) -> list[str]:
    """Order matches build_feature_matrix: word cols, char cols, then is_member_plus.
    Only meaningful for LogReg (XGBoost's dense columns are SVD components, not
    vocabulary tokens -- see XGBoostStrategy.feature_importance)."""
    return [
        *artifact.word_vectorizer.get_feature_names_out(),  # pyright: ignore
        *artifact.char_vectorizer.get_feature_names_out(),  # pyright: ignore
        "is_member_plus",
    ]


def train_classifier(
    x_train: SparseMatrix,
    y_train: NDArray[np.str_],
    *,
    C: float = DEFAULT_C,  # noqa: N803 -- matches sklearn's own LogisticRegression(C=...) name
    penalty: str = DEFAULT_PENALTY,
) -> LogisticRegression:
    # max_iter raised from sklearn's default 100: with word+char TF-IDF (tens of
    # thousands of columns) on ~7.7k rows, the default will very likely fail to
    # converge. class_weight stays exactly as the locked spec states.
    #
    # lbfgs (sklearn's default solver) only supports the "l2" penalty; "l1"
    # needs a different solver. liblinear is the usual first reach for l1, but
    # this repo's installed sklearn (1.9.0) raises "the 'liblinear' solver does
    # not support multiclass classification (n_classes >= 3)" for our 3-class
    # problem -- saga supports l1 with native multinomial multiclass instead.
    #
    # sklearn 1.9 also deprecates the `penalty` kwarg itself (FutureWarning:
    # removed in 1.10) in favor of always passing an explicit `l1_ratio` and
    # leaving `penalty` unset -- l1_ratio=0 means what penalty="l2" used to
    # mean, l1_ratio=1 means penalty="l1". Solver compatibility is unaffected:
    # sklearn still derives the same internal penalty from l1_ratio and
    # applies the same solver checks, so the l2/l1 <-> lbfgs/saga pairing
    # above still has to hold. This project's own `penalty` parameter name
    # (here, the CLI flag, tune_baseline.py's GRID) is unrelated to sklearn's
    # deprecated kwarg and is untouched by this.
    #
    # random_state is fixed (not a tuning target, like class_weight): saga
    # shuffles data stochastically, so two fits of the *same* config without a
    # fixed seed can diverge -- a real problem when Experiment 2's sweep
    # (tune_baseline.py) evaluates a config and a human then re-runs this CLI
    # with the winning flags to produce the actual gated candidate. Both fits
    # must produce the same model given the same config and data.
    solver = "saga" if penalty == "l1" else "lbfgs"
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        random_state=42,
    )
    model.fit(x_train, y_train)  # pyright: ignore
    return model


def compute_feature_std(x: SparseMatrix) -> NDArray[np.float64]:
    """Per-column std of the design matrix -- the scale correction that makes
    coef_ magnitudes comparable across columns on different scales (idf-weighted,
    row-L2-normalized TF-IDF terms vs. the raw {0,1} is_member_plus column).
    Computed via E[x^2] - E[x]^2 so it works directly on the sparse matrix.
    """
    mean = np.asarray(x.mean(axis=0)).ravel()  # pyright: ignore
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()  # pyright: ignore
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


class LogRegStrategy:
    name: ClassVar[str] = "logreg"

    def train(
        self,
        x_train: SparseMatrix,
        y_train: NDArray[np.str_],
        *,
        x_val: SparseMatrix | None = None,
        y_val: NDArray[np.str_] | None = None,
        C: float = DEFAULT_C,  # noqa: N803
        penalty: str = DEFAULT_PENALTY,
    ) -> tuple[TrainedModel, dict[str, Any]]:
        del x_val, y_val  # LogReg has no eval-set concept
        model = train_classifier(x_train, y_train, C=C, penalty=penalty)
        return model, {"feature_std": compute_feature_std(x_train)}  # pyright: ignore[reportReturnType]

    def transform_for_predict(self, x: SparseMatrix, extra: dict[str, Any]) -> SparseMatrix:
        del extra
        return x

    def feature_importance(
        self, artifact: BaselineArtifact, top_n: int
    ) -> dict[str, list[dict[str, Any]]]:
        """Per class: top_n features ranked by |std_weight|, each with weight,
        odds_ratio, std_weight, and importance_ratio.

        - weight: the raw coefficient (log-odds per unit of that feature).
        - odds_ratio: exp(weight) -- the standard interpretable form of a
          logistic coefficient ("this token roughly Nx's the odds of this class").
        - std_weight: weight * feature_std -- the standardized coefficient,
          correcting for the fact that raw weight is NOT comparable across
          columns on different scales.
        - importance_ratio: |std_weight| / sum(|std_weight|) for that class.
        """
        names = combined_feature_names(artifact)
        coef = np.asarray(artifact.model.coef_, dtype=np.float64)  # pyright: ignore
        feature_std = artifact.extra["feature_std"]
        std_coef = coef * feature_std
        result: dict[str, list[dict[str, Any]]] = {}
        for class_idx, class_name in enumerate(artifact.classes):
            weights: NDArray[np.float64] = coef[class_idx]
            std_weights: NDArray[np.float64] = std_coef[class_idx]
            total_abs = float(np.abs(std_weights).sum())
            ranked_idx = np.argsort(-np.abs(std_weights))[:top_n]
            result[class_name] = [
                {
                    "feature": names[i],
                    "weight": float(weights[i]),
                    "odds_ratio": float(np.exp(weights[i])),
                    "std_weight": float(std_weights[i]),
                    "importance_ratio": float(np.abs(std_weights[i])) / total_abs,
                }
                for i in ranked_idx
            ]
        return result


def _slice_and_densify(x: SparseMatrix, reducer: TruncatedSVD) -> NDArray[np.float64]:
    """Splits x into [text columns | is_member_plus column] using the fitted
    reducer's own recorded input width as the boundary (not a bare -1 index --
    this is the shape-assertion fix from independent review), runs the SVD
    transform on the text columns, and reassembles a dense array with
    is_member_plus appended as the last column."""
    n_text_cols = int(
        reducer.n_features_in_  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
    )
    expected_total = n_text_cols + 1
    if x.shape[1] != expected_total:
        raise ValueError(
            f"expected {expected_total} columns (text={n_text_cols} + is_member_plus=1) "
            f"from build_feature_matrix, got {x.shape[1]}"
        )
    text_cols = x[:, :n_text_cols]
    member_col = x[:, n_text_cols:]
    dense_text: NDArray[np.float64] = reducer.transform(text_cols)  # pyright: ignore
    return np.hstack([dense_text, member_col.toarray()])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


class _XGBoostModel:
    """Wraps a fitted XGBClassifier to speak the same string-label surface as
    LogisticRegression (the TrainedModel Protocol). xgboost's sklearn API no
    longer label-encodes string targets internally (fitting XGBClassifier
    directly on string labels raises "Invalid classes inferred from unique
    values of `y`" on this repo's installed xgboost 3.x) -- this wrapper does
    that encoding once at fit time and decodes predict()'s output back to the
    original labels, so contract.py/margin.py/serving/feature_importance.py
    never need to know XGBoost's classes are internally integers.
    """

    def __init__(self, model: XGBClassifier, classes: NDArray[np.str_]) -> None:
        self._model = model
        self.classes_ = classes

    def predict(self, x: SparseMatrix) -> NDArray[np.str_]:
        encoded: NDArray[np.intp] = self._model.predict(x)  # pyright: ignore
        return self.classes_[encoded.astype(np.intp)]

    def predict_proba(self, x: SparseMatrix) -> NDArray[np.float64]:
        return self._model.predict_proba(x)  # pyright: ignore

    @property
    def feature_importances_(self) -> NDArray[np.float64]:
        return self._model.feature_importances_  # pyright: ignore


class XGBoostStrategy:
    name: ClassVar[str] = "xgboost"

    def train(
        self,
        x_train: SparseMatrix,
        y_train: NDArray[np.str_],
        *,
        x_val: SparseMatrix | None = None,
        y_val: NDArray[np.str_] | None = None,
        n_components: int = DEFAULT_SVD_COMPONENTS,
        **hyperparams: Any,
    ) -> tuple[TrainedModel, dict[str, Any]]:
        n_text_cols = x_train.shape[1] - 1  # word+char cols; is_member_plus is last
        text_train = x_train[:, :n_text_cols]
        member_train = x_train[:, n_text_cols:]

        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        dense_text_train: NDArray[np.float64] = reducer.fit_transform(text_train)  # pyright: ignore
        dense_train = np.hstack([dense_text_train, member_train.toarray()])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        # ONE class-weight mapping, derived from TRAIN's distribution only, applied
        # to both splits. This mirrors exactly how class_weight="balanced" already
        # works for LogisticRegression (computed once from whatever y is passed to
        # .fit() -- train only; LR's .fit() has no eval-set concept at all).
        # Computing "balanced" independently on val would give val a DIFFERENT
        # mapping than train used (val's class proportions can differ from train's
        # by sampling noise even under a stratified split), so the weighted
        # mlogloss early stopping watches would no longer match what training is
        # actually minimizing. Do NOT compute_sample_weight("balanced", y_val)
        # independently -- this was flagged as a real bug in review.
        classes_arr = np.unique(y_train)
        class_weights = compute_class_weight(  # pyright: ignore[reportUnknownVariableType]
            "balanced", classes=classes_arr, y=y_train
        )
        class_weight_map: dict[str, float] = dict(
            zip(
                classes_arr.tolist(),
                class_weights.tolist(),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                strict=True,
            )
        )
        sample_weight = compute_sample_weight(class_weight=class_weight_map, y=y_train)

        # xgboost's sklearn API requires integer class labels (0..n_classes-1) --
        # see _XGBoostModel's docstring. label_to_idx is built from classes_arr's
        # sorted order so the encoded labels, and therefore predict_proba's
        # column order, line up with classes_arr / artifact.classes exactly.
        label_to_idx = {label: idx for idx, label in enumerate(classes_arr.tolist())}
        y_train_encoded = np.array([label_to_idx[label] for label in y_train])

        fit_kwargs: dict[str, Any] = {"sample_weight": sample_weight}
        early_stopping_rounds: int | None = None
        if x_val is not None and y_val is not None:
            dense_val = _slice_and_densify(x_val, reducer)
            sample_weight_val = compute_sample_weight(class_weight=class_weight_map, y=y_val)
            y_val_encoded = np.array([label_to_idx[label] for label in y_val])
            fit_kwargs["eval_set"] = [(dense_val, y_val_encoded)]
            fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
            fit_kwargs["verbose"] = False
            early_stopping_rounds = DEFAULT_EARLY_STOPPING_ROUNDS

        model = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=hyperparams.pop("n_estimators", DEFAULT_N_ESTIMATORS),
            early_stopping_rounds=early_stopping_rounds,
            importance_type="gain",
            tree_method="hist",
            random_state=42,
            **hyperparams,
        )
        model.fit(dense_train, y_train_encoded, **fit_kwargs)  # pyright: ignore
        return _XGBoostModel(model, classes_arr), {"reducer": reducer}

    def transform_for_predict(self, x: SparseMatrix, extra: dict[str, Any]) -> SparseMatrix:
        return _slice_and_densify(x, extra["reducer"])

    def feature_importance(
        self, artifact: BaselineArtifact, top_n: int
    ) -> dict[str, list[dict[str, Any]]]:
        reducer: TruncatedSVD = artifact.extra["reducer"]
        importances = np.asarray(artifact.model.feature_importances_, dtype=np.float64)  # pyright: ignore
        names = [f"svd_component_{i}" for i in range(reducer.n_components)] + ["is_member_plus"]  # pyright: ignore[reportUnknownMemberType]
        ranked_idx = np.argsort(-importances)[:top_n]  # pyright: ignore[reportUnknownArgumentType]
        return {
            "overall": [
                {"feature": names[i], "importance": float(importances[i])} for i in ranked_idx
            ]
        }


# LogRegStrategy/XGBoostStrategy's `train` each narrow the Protocol's
# **hyperparams: Any catch-all to their own real keyword args (C/penalty;
# n_components) -- a safe narrowing for how this module actually calls them,
# but not one pyright's structural Protocol matching accepts.
STRATEGIES: dict[str, TrainingStrategy] = {
    "logreg": LogRegStrategy(),  # pyright: ignore[reportAssignmentType]
    "xgboost": XGBoostStrategy(),
}


def get_strategy(model_type: str) -> TrainingStrategy:
    try:
        return STRATEGIES[model_type]
    except KeyError:
        raise ValueError(
            f"unknown model_type {model_type!r}; choose one of {sorted(STRATEGIES)}"
        ) from None
