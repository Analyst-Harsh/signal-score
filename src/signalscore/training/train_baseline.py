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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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

MODEL_VERSION = "baseline_v0"

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
    model_out = args.model_out or default_dir / "model.joblib"
    metrics_out = args.metrics_out or default_dir / "metrics.json"

    metrics = run_training_pipeline(args.train, args.val, model_out, metrics_out)
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
