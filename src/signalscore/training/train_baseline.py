"""Baseline: TF-IDF(word 1-2 + char_wb 3-5) + is_member_plus -> LogisticRegression.

Run via:
    uv run python -m signalscore.training.train_baseline \\
        --train data/processed/kubernetes-kubernetes/train.jsonl \\
        --val data/processed/kubernetes-kubernetes/val.jsonl

Locked spec: docs/v0-feature-selection.md. Fits vectorizers + classifier on
--train only; --val is transformed with the already-fit vectorizers, never
refit. Never reads eval_set_v1.jsonl or any test-split row.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from scipy import sparse
from sklearn.feature_extraction.text import (  # pyright: ignore[reportMissingTypeStubs]
    TfidfVectorizer,
)
from sklearn.metrics import (  # pyright: ignore[reportMissingTypeStubs]
    accuracy_score,  # pyright: ignore[reportUnknownVariableType]
    average_precision_score,  # pyright: ignore[reportUnknownVariableType]
    brier_score_loss,  # pyright: ignore[reportUnknownVariableType]
    f1_score,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.preprocessing import (
    label_binarize,  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
)

from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.registry import ModelRegistry
from signalscore.settings import Settings
from signalscore.training.strategies import (
    DEFAULT_C,
    DEFAULT_PENALTY,
    TrainedModel,
    get_strategy,
)

MODEL_VERSION = "baseline_v0"
# Reused by registry/evaluation/serving later so the artifact filename lives
# in one place instead of as a repeated inline literal.
MODEL_ARTIFACT_FILENAME = "model.joblib"

# Locked v0 spec hyperparameter defaults (docs/v0-feature-selection.md). Named
# here, not just inlined as keyword defaults below, so main()'s mlflow.log_params
# call can log the values actually passed to training instead of a separately
# maintained (and driftable) set of literals -- a single source of truth for
# what "the default config" means, tunable per-call by tune_baseline.py.
DEFAULT_WORD_NGRAM_RANGE: tuple[int, int] = (1, 2)
DEFAULT_WORD_MIN_DF = 3
DEFAULT_WORD_SUBLINEAR_TF = True

# scipy ships no py.typed marker, so every sparse-matrix boundary is Unknown to
# pyright -- isolate that behind one alias rather than fighting each variance.
SparseMatrix = Any


def load_feature_rows(path: Path) -> list[FeatureRow]:
    with path.open() as f:
        return [FeatureRow.model_validate_json(line) for line in f if line.strip()]


@dataclass
class FittedVectorizers:
    word: TfidfVectorizer
    char: TfidfVectorizer


def fit_vectorizers(
    train_texts: list[str],
    *,
    word_ngram_range: tuple[int, int] = DEFAULT_WORD_NGRAM_RANGE,
    word_min_df: int = DEFAULT_WORD_MIN_DF,
    word_sublinear_tf: bool = DEFAULT_WORD_SUBLINEAR_TF,
) -> FittedVectorizers:
    """Fits two fresh vectorizers on train-split text only, every call --
    never a module-global, per docs/design-patterns-guide.md's Singleton ban
    on fitted transformers.

    Only the word vectorizer is tunable (Experiment 2, docs/candidate_models/
    overview.md) -- the char_wb vectorizer stays at its locked spec values.
    """
    word = TfidfVectorizer(
        ngram_range=word_ngram_range,
        min_df=word_min_df,
        sublinear_tf=word_sublinear_tf,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000, sublinear_tf=True
    )
    word.fit(train_texts)  # pyright: ignore
    char.fit(train_texts)  # pyright: ignore
    return FittedVectorizers(word=word, char=char)


def build_feature_matrix(rows: list[FeatureRow], vectorizers: FittedVectorizers) -> SparseMatrix:
    """Always `.transform()`, never `.fit_transform()` -- the vectorizers were
    already fit in `fit_vectorizers`. This is what makes fit-on-train /
    transform-on-val structural rather than a naming convention.
    """
    texts = [row.text for row in rows]
    word_matrix = vectorizers.word.transform(texts)  # pyright: ignore
    char_matrix = vectorizers.char.transform(texts)  # pyright: ignore
    member_column = sparse.csr_matrix(  # pyright: ignore
        np.array([[1.0 if row.is_member_plus else 0.0] for row in rows])
    )
    return sparse.hstack(  # pyright: ignore
        [word_matrix, char_matrix, member_column], format="csr"
    )


def extract_labels(rows: list[FeatureRow]) -> NDArray[np.str_]:
    return np.array([row.label.value for row in rows])


@dataclass
class BaselineArtifact:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    model: TrainedModel
    classes: list[str]
    model_type: str = "logreg"
    extra: dict[str, Any] = field(default_factory=lambda: {})
    # legacy field, backward-compat only -- see __setstate__
    feature_std: NDArray[np.float64] | None = None

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Pickle restores dataclass instances via __dict__.update(), bypassing
        # __init__ defaults entirely -- an already-pickled pre-refactor artifact's
        # __dict__ has NO model_type/extra key at all, not "missing with the
        # default applied". Backfill explicitly, or the currently-registered
        # production baseline_v0 pickle becomes unloadable (AttributeError) the
        # moment any code reads artifact.model_type.
        state.setdefault("model_type", "logreg")
        state.setdefault(
            "extra",
            {"feature_std": state["feature_std"]} if state.get("feature_std") is not None else {},
        )
        self.__dict__.update(state)


# Training is always invoked as `python -m signalscore.training.train_baseline`
# (see this module's own docstring), which makes Python set this module's
# __name__ -- and therefore BaselineArtifact.__module__ -- to "__main__" at
# joblib.dump() time. Any other process (the gate CLI, a fresh serving
# process, tests) unpickling that file looks for BaselineArtifact on ITS OWN
# __main__ and fails with AttributeError.
#
# Renaming __module__ alone isn't enough: pickle's save_global() re-imports
# whatever module __module__ names to verify it holds the exact same class
# object being pickled. Under a `-m` run this module was never registered in
# sys.modules under its real dotted path, so that re-import would execute
# the file a second time and mint a second, distinct BaselineArtifact class
# -- pickle then (correctly) refuses, raising "not the same object as ...".
# Aliasing sys.modules to the already-executing module object (not a fresh
# import) keeps both the name AND the object identity correct.
if BaselineArtifact.__module__ == "__main__":
    BaselineArtifact.__module__ = "signalscore.training.train_baseline"
    sys.modules.setdefault("signalscore.training.train_baseline", sys.modules["__main__"])


def _per_class_f1(
    y_true: NDArray[np.str_], y_pred: NDArray[np.str_], labels: list[str]
) -> NDArray[np.float64]:
    """Isolates the one f1_score call whose `average=None` return type sklearn's
    stubs can't express -- everything downstream works with a typed array.
    """
    return f1_score(y_true, y_pred, average=None, labels=labels)  # pyright: ignore


def compute_metrics(
    y_true: NDArray[np.str_],
    y_pred: NDArray[np.str_],
    y_proba: NDArray[np.float64],
    classes: list[str],
) -> dict[str, Any]:
    """`classes` MUST be `list(model.classes_)` from the fitted model, threaded in
    unchanged from `run_training_pipeline` -- never a separately hand-authored
    list. `y_proba`'s columns are in `model.classes_` order; if the one-hot
    encoding used for PR-AUC/Brier used a different ordering, both metrics would
    be silently wrong. Priority labels sort alphabetically in P0/P1/P2 order
    today, so a hardcoded list would coincidentally work -- that coincidence is
    closed off deliberately here, not relied on.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=classes))
    per_class_scores = _per_class_f1(y_true, y_pred, classes)
    per_class_f1 = {cls: float(score) for cls, score in zip(classes, per_class_scores, strict=True)}
    minority_f1 = per_class_f1[Priority.P0_CRITICAL.value]

    y_true_binarized = label_binarize(y_true, classes=classes)  # pyright: ignore
    pr_auc_macro = float(
        average_precision_score(y_true_binarized, y_proba, average="macro")  # pyright: ignore
    )
    brier_scores = [
        brier_score_loss((y_true == cls).astype(int), y_proba[:, i])
        for i, cls in enumerate(classes)
    ]
    brier_macro = float(np.mean(brier_scores))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "minority_f1": minority_f1,
        "pr_auc_macro": pr_auc_macro,
        "brier_macro": brier_macro,
    }


def save_artifact(artifact: BaselineArtifact, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)  # pyright: ignore[reportUnknownMemberType]


def save_metrics(metrics: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))


def format_experiments_row(
    date: str,
    model_config: str,
    dataset_snapshot: str,
    metrics: dict[str, Any],
    gate_result: str,
    notes: str,
) -> str:
    """Column order matches EXPERIMENTS.md's log table exactly: Date, Model config,
    Dataset snapshot, PR-AUC, Minority F1, Brier/ECE, Gate result, Notes.
    """
    return (
        f"| {date} | {model_config} | {dataset_snapshot} | {metrics['pr_auc_macro']:.4f} | "
        f"{metrics['minority_f1']:.4f} | {metrics['brier_macro']:.4f} | {gate_result} | {notes} |"
    )


def train_and_evaluate(
    train_rows: list[FeatureRow],
    val_rows: list[FeatureRow],
    *,
    word_ngram_range: tuple[int, int] = DEFAULT_WORD_NGRAM_RANGE,
    word_min_df: int = DEFAULT_WORD_MIN_DF,
    word_sublinear_tf: bool = DEFAULT_WORD_SUBLINEAR_TF,
    model_type: str = "logreg",
    model_hyperparams: dict[str, Any] | None = None,
) -> tuple[BaselineArtifact, dict[str, Any]]:
    """Fit-on-train / evaluate-on-val core, with no disk or MLflow I/O -- the
    reusable entry point for both a single training run (`run_training_pipeline`)
    and a hyperparameter sweep (`tune_baseline.py`), which calls this in a loop
    without reloading rows from disk each time.
    """
    vectorizers = fit_vectorizers(
        [row.text for row in train_rows],
        word_ngram_range=word_ngram_range,
        word_min_df=word_min_df,
        word_sublinear_tf=word_sublinear_tf,
    )
    x_train = build_feature_matrix(train_rows, vectorizers)
    x_val = build_feature_matrix(val_rows, vectorizers)
    y_train = extract_labels(train_rows)
    y_val = extract_labels(val_rows)

    strategy = get_strategy(model_type)
    model, extra = strategy.train(
        x_train, y_train, x_val=x_val, y_val=y_val, **(model_hyperparams or {})
    )
    classes = [str(c) for c in model.classes_]  # pyright: ignore

    x_val_for_predict = strategy.transform_for_predict(x_val, extra)
    y_pred: NDArray[np.str_] = model.predict(x_val_for_predict)  # pyright: ignore
    y_proba: NDArray[np.float64] = model.predict_proba(x_val_for_predict)  # pyright: ignore

    metrics = compute_metrics(y_val, y_pred, y_proba, classes)  # pyright: ignore[reportUnknownArgumentType]

    artifact = BaselineArtifact(
        word_vectorizer=vectorizers.word,
        char_vectorizer=vectorizers.char,
        model=model,
        classes=classes,
        model_type=model_type,
        extra=extra,
    )
    return artifact, metrics


def run_training_pipeline(
    train_path: Path,
    val_path: Path,
    model_out: Path,
    metrics_out: Path,
    *,
    word_ngram_range: tuple[int, int] = DEFAULT_WORD_NGRAM_RANGE,
    word_min_df: int = DEFAULT_WORD_MIN_DF,
    word_sublinear_tf: bool = DEFAULT_WORD_SUBLINEAR_TF,
    model_type: str = "logreg",
    model_hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_rows = load_feature_rows(train_path)
    val_rows = load_feature_rows(val_path)

    artifact, metrics = train_and_evaluate(
        train_rows,
        val_rows,
        word_ngram_range=word_ngram_range,
        word_min_df=word_min_df,
        word_sublinear_tf=word_sublinear_tf,
        model_type=model_type,
        model_hyperparams=model_hyperparams,
    )

    save_artifact(artifact, model_out)
    save_metrics(metrics, metrics_out)
    return metrics


def format_config_label(config: dict[str, Any]) -> str:
    """Renders a hyperparameter config (the kwargs accepted by
    `train_and_evaluate`) as a human-readable model description -- for the
    EXPERIMENTS.md row and, in `tune_baseline.py`, the sweep's trial table
    and recommended next command. Built from the actual config every time,
    never a static string, since hyperparameters are CLI-tunable.
    """
    lo, hi = config["word_ngram_range"]
    model_type = config.get("model_type", "logreg")
    hyperparams = config.get("model_hyperparams", {})
    return (
        f"{MODEL_VERSION}: tfidf(word {lo}-{hi} min_df={config['word_min_df']} "
        f"sublinear_tf={config['word_sublinear_tf']} + char_wb 3-5) + is_member_plus "
        f"-> {model_type}({hyperparams})"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    # Hyperparameter overrides for Experiment 2 (docs/candidate_models/overview.md):
    # tune_baseline.py sweeps these locally and recommends a winning combination,
    # then this same CLI is re-run with that combination to produce the one real,
    # gated candidate -- tune_baseline.py itself never saves or registers a model.
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--penalty", choices=["l2", "l1"], default=DEFAULT_PENALTY)
    parser.add_argument("--word-ngram-max", type=int, default=DEFAULT_WORD_NGRAM_RANGE[1])
    parser.add_argument("--word-min-df", type=int, default=DEFAULT_WORD_MIN_DF)
    parser.add_argument(
        "--word-sublinear-tf",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WORD_SUBLINEAR_TF,
    )
    parser.add_argument("--model-type", default="logreg", choices=["logreg", "xgboost"])
    parser.add_argument("--model-hyperparams", type=json.loads, default="{}")
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_dir_name = args.train.parent.name
    default_dir = Path("data/models") / repo_dir_name / MODEL_VERSION
    model_out = args.model_out or default_dir / MODEL_ARTIFACT_FILENAME
    metrics_out = args.metrics_out or default_dir / "metrics.json"

    if args.model_type == "logreg":
        model_hyperparams: dict[str, Any] = {"C": args.C, "penalty": args.penalty}
    else:
        model_hyperparams = args.model_hyperparams

    config: dict[str, Any] = {
        "word_ngram_range": (1, args.word_ngram_max),
        "word_min_df": args.word_min_df,
        "word_sublinear_tf": args.word_sublinear_tf,
        "model_type": args.model_type,
        "model_hyperparams": model_hyperparams,
    }

    train_rows = load_feature_rows(args.train)
    val_rows = load_feature_rows(args.val)

    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
    mlflow.set_experiment(MODEL_VERSION)  # pyright: ignore[reportUnknownMemberType]
    with mlflow.start_run(  # pyright: ignore[reportUnknownMemberType]
        run_name=f"{MODEL_VERSION}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    ) as run:
        mlflow.log_params(  # pyright: ignore[reportUnknownMemberType]
            {
                "model_version": MODEL_VERSION,
                # Logs whatever config was actually used -- the CLI defaults
                # (DEFAULT_*) when no flags are passed, or the caller's
                # overrides otherwise. Never a separately hardcoded, driftable
                # set of literals.
                "word_ngram_range": "{}-{}".format(*config["word_ngram_range"]),
                "word_min_df": config["word_min_df"],
                "word_sublinear_tf": config["word_sublinear_tf"],
                "char_ngram_range": "3-5",
                "char_min_df": 5,
                "char_max_features": 300_000,
                "classifier": config["model_type"],
                # class_weight="balanced"/max_iter=1000 are LogisticRegression-specific
                # fixed literals, not something XGBoost shares -- only logged when
                # that's actually the family being trained.
                **(
                    {"class_weight": "balanced", "max_iter": 1000}
                    if config["model_type"] == "logreg"
                    else {}
                ),
                # Flattened, not the nested dict itself -- mlflow.log_params
                # stringifies a nested dict into one opaque blob rather than
                # per-key queryable params.
                **{f"model_hyperparams.{k}": v for k, v in config["model_hyperparams"].items()},
                # reproducibility: which library versions produced this artifact --
                # joblib.load() is sensitive to exact minor-version mismatches, and
                # mlflow.log_artifact() on a raw joblib dump (not mlflow.sklearn.log_model)
                # skips MLflow's automatic environment capture, so this is the only record.
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                # data snapshot identity: repo_dir_name alone (already in EXPERIMENTS.md)
                # doesn't say WHICH pull of that repo produced this candidate.
                "train_row_count": len(train_rows),
                "val_row_count": len(val_rows),
            }
        )
        metrics = run_training_pipeline(args.train, args.val, model_out, metrics_out, **config)
        mlflow.log_metrics(  # pyright: ignore[reportUnknownMemberType]
            {k: v for k, v in metrics.items() if k != "per_class_f1"}
        )
        mlflow.log_dict(metrics["per_class_f1"], "per_class_f1.json")  # pyright: ignore[reportUnknownMemberType]
        mlflow.log_artifact(str(model_out), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
        ModelRegistry().register_candidate(run_id=run.info.run_id)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    print(json.dumps(metrics, indent=2))

    row = format_experiments_row(
        date=datetime.now(UTC).date().isoformat(),
        model_config=format_config_label(config),
        dataset_snapshot=repo_dir_name,
        metrics=metrics,
        gate_result="n/a -- no promotion gate yet",
        notes=args.notes,
    )
    print(row)


if __name__ == "__main__":
    main()
