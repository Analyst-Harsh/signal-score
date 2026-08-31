"""Feature weights and importance ratios for the baseline model -- multi-family:
dispatches to whichever `TrainingStrategy` produced the artifact
(signalscore.training.strategies).

Run via:
    uv run python -m signalscore.training.feature_importance \\
        --model data/models/kubernetes-kubernetes/baseline_v0/model.joblib

For LogisticRegression, `coef_` (shape n_classes x n_features) already *is*
the weight of every feature for every class; LogRegStrategy.feature_importance
names the columns and ranks them, reading the same coefficients four ways:

- weight: the raw coefficient (log-odds per unit of that feature).
- odds_ratio: exp(weight) -- the standard interpretable form of a logistic
  coefficient ("this token roughly Nx's the odds of this class").
- std_weight: weight * artifact.extra["feature_std"] -- the standardized
  coefficient. Raw weight is NOT comparable across columns: idf-weighted,
  row-L2-normalized TF-IDF terms and the raw {0,1} is_member_plus column are
  on very different scales, so a bigger raw weight doesn't mean a bigger real
  effect.
- importance_ratio: |std_weight| / sum(|std_weight|) for that class -- a 0..1
  figure that sums to 1, comparable to a tree model's feature_importances_.

For XGBoost, there's no per-class native importance (multi:softprob doesn't
expose one) -- XGBoostStrategy.feature_importance returns a single "overall"
key instead of faking a per-class breakdown.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib

# Re-exported for backward compat -- some callers still do
# `from signalscore.training.feature_importance import combined_feature_names`.
# The `as combined_feature_names` (not a bare import) is the standard
# explicit-re-export idiom type checkers recognize as intentional.
from signalscore.training.strategies import combined_feature_names as combined_feature_names
from signalscore.training.strategies import get_strategy
from signalscore.training.train_baseline import BaselineArtifact


def top_features_by_class(
    artifact: BaselineArtifact, top_n: int = 25
) -> dict[str, list[dict[str, Any]]]:
    return get_strategy(artifact.model_type).feature_importance(artifact, top_n)


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
