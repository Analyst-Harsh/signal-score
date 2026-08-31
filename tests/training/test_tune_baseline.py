"""Tests for signalscore.training.tune_baseline (Experiment 2 sweep)."""

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.training.tune_baseline import (
    GRID,
    Trial,
    format_next_command,
    iter_configs,
    main,
    select_winner,
)

CLASS_TEXTS = {
    Priority.P0_CRITICAL: "critical outage pod crash cluster panic urgent failure",
    Priority.P1_SOON: "important bug slow response needs investigation soon fix",
    Priority.P2_BACKLOG: "minor cosmetic typo backlog low priority cleanup task",
}


def make_feature_row(
    issue_number: int,
    text: str,
    label: Priority = Priority.P2_BACKLOG,
    is_member_plus: bool = False,
) -> FeatureRow:
    return FeatureRow(
        issue_number=issue_number,
        created_at=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=issue_number),
        label=label,
        text=text,
        is_member_plus=is_member_plus,
        title_chars=1,
        body_chars=1,
        body_words=1,
        has_code_block=False,
        has_stack_trace=False,
        has_url=False,
        has_logline=False,
        has_excl=False,
        code_ratio=0.0,
        any_lex=False,
        is_ci_flake_shaped=False,
    )


def make_rows(n_per_class: int, start_issue: int = 0) -> list[FeatureRow]:
    """Enough repeated shared vocabulary per class to clear the locked spec's
    min_df=3 (word) / min_df=5 (char_wb) thresholds -- mirrors
    tests/training/test_train_baseline.py's helper of the same name.
    """
    rows: list[FeatureRow] = []
    issue = start_issue
    for label, text in CLASS_TEXTS.items():
        for i in range(n_per_class):
            rows.append(
                make_feature_row(
                    issue, f"{text} variant {i}", label=label, is_member_plus=(i % 2 == 0)
                )
            )
            issue += 1
    return rows


def write_rows(rows: list[FeatureRow], path: Path) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def test_iter_configs_full_grid_covers_every_combination() -> None:
    configs = list(iter_configs(budget=None))

    assert len(configs) == math.prod(len(values) for values in GRID.values())
    assert len(configs) == 64


def test_iter_configs_budget_is_reproducible_with_same_seed() -> None:
    first = list(iter_configs(budget=10, seed=42))
    second = list(iter_configs(budget=10, seed=42))

    assert first == second
    assert len(first) == 10


def test_select_winner_picks_highest_pr_auc_macro() -> None:
    trials = [
        Trial(config={"C": 1.0}, metrics={"pr_auc_macro": 0.5, "minority_f1": 0.4}),
        Trial(config={"C": 0.1}, metrics={"pr_auc_macro": 0.6, "minority_f1": 0.3}),
        Trial(config={"C": 10.0}, metrics={"pr_auc_macro": 0.55, "minority_f1": 0.9}),
    ]

    winner = select_winner(trials)

    assert winner.config == {"C": 0.1}


def test_select_winner_breaks_ties_with_minority_f1() -> None:
    trials = [
        Trial(config={"tag": "a"}, metrics={"pr_auc_macro": 0.5, "minority_f1": 0.4}),
        Trial(config={"tag": "b"}, metrics={"pr_auc_macro": 0.5, "minority_f1": 0.7}),
    ]

    winner = select_winner(trials)

    assert winner.config == {"tag": "b"}


def test_format_next_command_reproduces_the_config_as_cli_flags() -> None:
    config = {
        "C": 0.1,
        "penalty": "l1",
        "word_ngram_range": (1, 1),
        "word_min_df": 5,
        "word_sublinear_tf": False,
    }

    command = format_next_command(config, Path("train.jsonl"), Path("val.jsonl"))

    assert "signalscore.training.train_baseline" in command
    assert "--train train.jsonl" in command
    assert "--val val.jsonl" in command
    assert "--C 0.1" in command
    assert "--penalty l1" in command
    assert "--word-ngram-max 1" in command
    assert "--word-min-df 5" in command
    assert "--no-word-sublinear-tf" in command


def test_format_next_command_uses_the_positive_sublinear_flag_when_true() -> None:
    config = {
        "C": 1.0,
        "penalty": "l2",
        "word_ngram_range": (1, 2),
        "word_min_df": 3,
        "word_sublinear_tf": True,
    }

    command = format_next_command(config, Path("train.jsonl"), Path("val.jsonl"))

    assert "--word-sublinear-tf" in command
    assert "--no-word-sublinear-tf" not in command


def test_main_is_pure_search_and_report_no_persistence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression guard for the tuner/trainer split: sweeping must never save
    a model artifact, write metrics, or touch MLflow -- it only prints a
    ranked trial table and the train_baseline.py command to run next.
    """
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    files_before = set(tmp_path.iterdir())

    main(
        [
            "--train",
            str(train_path),
            "--val",
            str(val_path),
            "--budget",
            "3",
            "--seed",
            "42",
        ]
    )

    assert set(tmp_path.iterdir()) == files_before

    out = capsys.readouterr().out
    assert "pr_auc_macro" in out
    assert "Winner:" in out
    assert "Run this to produce the real, gated candidate:" in out
    assert "signalscore.training.train_baseline" in out
