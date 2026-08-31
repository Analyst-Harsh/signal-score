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
from dataclasses import dataclass
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
from sklearn.linear_model import LogisticRegression
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

MODEL_VERSION = "baseline_v0"
# Reused by registry/evaluation/serving later so the artifact filename lives
# in one place instead of as a repeated inline literal.
MODEL_ARTIFACT_FILENAME = "model.joblib"

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


def fit_vectorizers(train_texts: list[str]) -> FittedVectorizers:
    """Fits two fresh vectorizers on train-split text only, every call --
    never a module-global, per docs/design-patterns-guide.md's Singleton ban
    on fitted transformers.
    """
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents="unicode")
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


def compute_feature_std(x: SparseMatrix) -> NDArray[np.float64]:
    """Per-column std of the design matrix -- the scale correction that makes
    coef_ magnitudes comparable across columns on different scales (idf-weighted,
    row-L2-normalized TF-IDF terms vs. the raw {0,1} is_member_plus column).
    Computed via E[x^2] - E[x]^2 so it works directly on the sparse matrix.
    """
    mean = np.asarray(x.mean(axis=0)).ravel()  # pyright: ignore
    mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()  # pyright: ignore
    return np.sqrt(np.maximum(mean_sq - mean**2, 0.0))


def train_classifier(x_train: SparseMatrix, y_train: NDArray[np.str_]) -> LogisticRegression:
    # max_iter raised from sklearn's default 100: with word+char TF-IDF (tens of
    # thousands of columns) on ~7.7k rows, the default will very likely fail to
    # converge. class_weight stays exactly as the locked spec states.
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(x_train, y_train)  # pyright: ignore
    return model


@dataclass
class BaselineArtifact:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    model: LogisticRegression
    classes: list[str]
    feature_std: NDArray[np.float64]


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


def run_training_pipeline(
    train_path: Path, val_path: Path, model_out: Path, metrics_out: Path
) -> dict[str, Any]:
    train_rows = load_feature_rows(train_path)
    val_rows = load_feature_rows(val_path)

    vectorizers = fit_vectorizers([row.text for row in train_rows])
    x_train = build_feature_matrix(train_rows, vectorizers)
    x_val = build_feature_matrix(val_rows, vectorizers)
    y_train = extract_labels(train_rows)
    y_val = extract_labels(val_rows)

    model = train_classifier(x_train, y_train)
    classes = [str(c) for c in model.classes_]  # pyright: ignore

    y_pred: NDArray[np.str_] = model.predict(x_val)  # pyright: ignore
    y_proba: NDArray[np.float64] = model.predict_proba(x_val)  # pyright: ignore

    metrics = compute_metrics(y_val, y_pred, y_proba, classes)  # pyright: ignore[reportUnknownArgumentType]

    artifact = BaselineArtifact(
        word_vectorizer=vectorizers.word,
        char_vectorizer=vectorizers.char,
        model=model,
        classes=classes,
        feature_std=compute_feature_std(x_train),
    )
    save_artifact(artifact, model_out)
    save_metrics(metrics, metrics_out)
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, default=None)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--notes", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_dir_name = args.train.parent.name
    default_dir = Path("data/models") / repo_dir_name / MODEL_VERSION
    model_out = args.model_out or default_dir / MODEL_ARTIFACT_FILENAME
    metrics_out = args.metrics_out or default_dir / "metrics.json"

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
                "word_ngram_range": "1-2",
                "word_min_df": 3,
                "char_ngram_range": "3-5",
                "char_min_df": 5,
                "char_max_features": 300_000,
                "classifier": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 1000,
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
        metrics = run_training_pipeline(args.train, args.val, model_out, metrics_out)
        mlflow.log_metrics(  # pyright: ignore[reportUnknownMemberType]
            {k: v for k, v in metrics.items() if k != "per_class_f1"}
        )
        mlflow.log_dict(metrics["per_class_f1"], "per_class_f1.json")  # pyright: ignore[reportUnknownMemberType]
        mlflow.log_artifact(str(model_out), artifact_path="model")  # pyright: ignore[reportUnknownMemberType]
        ModelRegistry().register_candidate(run_id=run.info.run_id)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    print(json.dumps(metrics, indent=2))

    row = format_experiments_row(
        date=datetime.now(UTC).date().isoformat(),
        model_config=(
            f"{MODEL_VERSION}: tfidf(word 1-2 + char_wb 3-5) + is_member_plus -> LogisticRegression"
        ),
        dataset_snapshot=repo_dir_name,
        metrics=metrics,
        gate_result="n/a -- no promotion gate yet",
        notes=args.notes,
    )
    print(row)


if __name__ == "__main__":
    main()
