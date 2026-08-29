"""Feature weights and importance ratios for the baseline LogisticRegression.

Run via:
    uv run python -m signalscore.training.feature_importance \\
        --model data/models/kubernetes-kubernetes/baseline_v0/model.joblib

`LogisticRegression.coef_` (shape n_classes x n_features) already *is* the
weight of every feature for every class; this module just names the columns
and ranks them, reading the same coefficients three ways:

- weight: the raw coefficient (log-odds per unit of that feature).
- odds_ratio: exp(weight) -- the standard interpretable form of a logistic
  coefficient ("this token roughly Nx's the odds of this class").
- importance_ratio: |weight| / sum(|weight|) for that class -- a 0..1 figure
  that sums to 1, comparable to a tree model's feature_importances_.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from signalscore.training.train_baseline import BaselineArtifact


def combined_feature_names(artifact: BaselineArtifact) -> list[str]:
    """Order matches build_feature_matrix: word cols, char cols, then is_member_plus."""
    return [
        *artifact.word_vectorizer.get_feature_names_out(),  # pyright: ignore
        *artifact.char_vectorizer.get_feature_names_out(),  # pyright: ignore
        "is_member_plus",
    ]


def top_features_by_class(
    artifact: BaselineArtifact, top_n: int = 25
) -> dict[str, list[dict[str, Any]]]:
    """Per class: top_n features ranked by |weight|, each with weight, odds_ratio,
    and importance_ratio (see module docstring).
    """
    names = combined_feature_names(artifact)
    coef = np.asarray(artifact.model.coef_, dtype=np.float64)  # pyright: ignore
    result: dict[str, list[dict[str, Any]]] = {}
    for class_idx, class_name in enumerate(artifact.classes):
        weights: NDArray[np.float64] = coef[class_idx]
        total_abs = float(np.abs(weights).sum())
        ranked_idx = np.argsort(-np.abs(weights))[:top_n]
        result[class_name] = [
            {
                "feature": names[i],
                "weight": float(weights[i]),
                "odds_ratio": float(np.exp(weights[i])),
                "importance_ratio": float(np.abs(weights[i])) / total_abs,
            }
            for i in ranked_idx
        ]
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact: BaselineArtifact = joblib.load(args.model)  # pyright: ignore
    report = top_features_by_class(artifact, top_n=args.top_n)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
