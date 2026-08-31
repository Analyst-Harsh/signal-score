"""Experiment 2: hyperparameter sweep on baseline_v0's own feature set --
TF-IDF(word) + TF-IDF(char) + is_member_plus -> LogisticRegression. See
docs/candidate_models/overview.md.

Diagnostics (Experiment 1) are NOT re-included here in any form -- Experiment
1 already rejected them via the real gate (EXPERIMENTS.md, 2026-08-31).

Pure search + report: fits each hyperparameter configuration on --train only,
evaluates once on --val (no k-fold CV, matching train_baseline.py's existing
fit-on-train/transform-on-val discipline), and prints a ranked trial table
plus the winning config as a ready-to-run `train_baseline.py` command.

This script never saves a model artifact, logs to MLflow, or touches the
model registry -- that's train_baseline.py's job. Re-run it (the exact
command is printed below the winner) with the winning hyperparameter flags
to produce the one real training run that gets an MLflow candidate and,
after `run_gate.py`, an EXPERIMENTS.md row. A sweep trial is not itself
"a real training run" in that sense; only the config someone chooses to
actually commit to is.

Run via:
    uv run python -m signalscore.training.tune_baseline \\
        --train data/processed/kubernetes-kubernetes/train.jsonl \\
        --val data/processed/kubernetes-kubernetes/val.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import (  # pyright: ignore[reportMissingTypeStubs]
    ParameterGrid,  # pyright: ignore[reportUnknownVariableType]
    ParameterSampler,  # pyright: ignore[reportUnknownVariableType]
)

from signalscore.training.train_baseline import (
    format_config_label,
    load_feature_rows,
    train_and_evaluate,
)

# Experiment 2 search space (docs/candidate_models/overview.md step 2) --
# word-vectorizer + regularization knobs only. The char_wb vectorizer and
# class_weight="balanced" stay fixed (locked spec / grid-tractability, see
# the plan) -- not swept here.
GRID: dict[str, list[Any]] = {
    "C": [0.01, 0.1, 1.0, 10.0],
    "penalty": ["l2", "l1"],
    "word_ngram_range": [(1, 1), (1, 2)],
    "word_min_df": [3, 5],
    "word_sublinear_tf": [True, False],
}


@dataclass
class Trial:
    config: dict[str, Any]
    metrics: dict[str, Any]


def iter_configs(*, budget: int | None = None, seed: int = 42) -> Iterator[dict[str, Any]]:
    """Enumerates the search space -- the full 64-cell grid by default, or a
    fixed-budget random sample (sampling without replacement, since every
    GRID value is a list) as a faster fallback if the full factorial proves
    too slow in practice.
    """
    if budget is None:
        yield from ParameterGrid(GRID)  # pyright: ignore[reportUnknownVariableType]
    else:
        yield from ParameterSampler(  # pyright: ignore[reportUnknownVariableType]
            GRID, n_iter=budget, random_state=seed
        )


def select_winner(trials: list[Trial]) -> Trial:
    """Primary metric matches the gate's own primary criterion (pr_auc_macro);
    minority_f1 breaks ties. brier_macro is gate-checked too but not a sort
    key here -- it's a non-regression cap, not something to optimize for.
    """
    return max(trials, key=lambda t: (t.metrics["pr_auc_macro"], t.metrics["minority_f1"]))


def format_next_command(config: dict[str, Any], train_path: Path, val_path: Path) -> str:
    """The exact train_baseline.py invocation that reproduces this config as
    the one real, gated candidate. train_classifier's fixed random_state
    makes this reproduction deterministic, not just "close" -- see
    train_baseline.py's train_classifier docstring comment.
    """
    _, hi = config["word_ngram_range"]
    sublinear_flag = (
        "--word-sublinear-tf" if config["word_sublinear_tf"] else "--no-word-sublinear-tf"
    )
    return (
        "uv run python -m signalscore.training.train_baseline "
        f"--train {train_path} --val {val_path} "
        f"--C {config['C']} --penalty {config['penalty']} "
        f"--word-ngram-max {hi} --word-min-df {config['word_min_df']} {sublinear_flag}"
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
        default=None,
        help="Random-sample this many configs instead of the full 64-cell grid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Only used with --budget.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    train_rows = load_feature_rows(args.train)
    val_rows = load_feature_rows(args.val)

    trials: list[Trial] = [
        Trial(config=config, metrics=train_and_evaluate(train_rows, val_rows, **config)[1])
        for config in iter_configs(budget=args.budget, seed=args.seed)
    ]

    _print_trial_table(trials)

    winner = select_winner(trials)
    m = winner.metrics
    print()
    print(f"Winner: {format_config_label(winner.config)}")
    print(
        f"  pr_auc_macro={m['pr_auc_macro']:.4f} minority_f1={m['minority_f1']:.4f} "
        f"brier_macro={m['brier_macro']:.4f}"
    )
    print()
    print("Run this to produce the real, gated candidate:")
    print(f"  {format_next_command(winner.config, args.train, args.val)}")


if __name__ == "__main__":
    main()
