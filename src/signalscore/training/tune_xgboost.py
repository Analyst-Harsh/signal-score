"""Experiment 3: XGBoost on a dense SVD-compressed representation of the
TF-IDF text signal + is_member_plus. See docs/candidate_models/overview.md.

Pure search + report, same discipline as tune_baseline.py's Experiment 2:
fits each config on --train only, evaluates once on --val, prints a ranked
trial table and the winning config as a ready-to-run train_baseline.py
command. Never saves a model, logs to MLflow, or touches the registry.

Full factorial over these 7 knobs is ~3,888 cells -- always budgeted
(random search via ParameterSampler), never the full grid.

Run via:
    uv run python -m signalscore.training.tune_xgboost \\
        --train data/processed/kubernetes-kubernetes/train.jsonl \\
        --val data/processed/kubernetes-kubernetes/val.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from signalscore.training.train_baseline import load_feature_rows, train_and_evaluate
from signalscore.training.tune_baseline import Trial, iter_configs, select_winner

GRID_XGB: dict[str, list[Any]] = {
    "max_depth": [2, 3, 4, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0],
}


def format_next_command(config: dict[str, Any], train_path: Path, val_path: Path) -> str:
    hyperparams_json = json.dumps(config)
    return (
        "uv run python -m signalscore.training.train_baseline "
        f"--train {train_path} --val {val_path} "
        f"--model-type xgboost --model-hyperparams '{hyperparams_json}'"
    )


def _print_trial_table(trials: list[Trial]) -> None:
    ranked = sorted(trials, key=lambda t: t.metrics["pr_auc_macro"], reverse=True)
    print(f"{'pr_auc_macro':>13}  {'minority_f1':>11}  {'brier_macro':>11}  config")
    for t in ranked:
        m = t.metrics
        print(
            f"{m['pr_auc_macro']:>13.4f}  {m['minority_f1']:>11.4f}  "
            f"{m['brier_macro']:>11.4f}  {t.config}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=80,
        help="Random-sample this many configs -- full factorial (~3,888 cells) is never used.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_rows = load_feature_rows(args.train)
    val_rows = load_feature_rows(args.val)

    trials = [
        Trial(
            config=config,
            metrics=train_and_evaluate(
                train_rows, val_rows, model_type="xgboost", model_hyperparams=config
            )[1],
        )
        for config in iter_configs(GRID_XGB, budget=args.budget, seed=args.seed)
    ]

    _print_trial_table(trials)
    winner = select_winner(trials)
    m = winner.metrics
    print()
    print(f"Winner: xgboost({winner.config})")
    print(
        f"  pr_auc_macro={m['pr_auc_macro']:.4f} minority_f1={m['minority_f1']:.4f} "
        f"brier_macro={m['brier_macro']:.4f}"
    )
    print()
    print("Run this to produce the real, gated candidate:")
    print(f"  {format_next_command(winner.config, args.train, args.val)}")


if __name__ == "__main__":
    main()
