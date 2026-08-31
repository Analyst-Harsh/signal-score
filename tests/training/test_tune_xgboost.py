"""Tests for signalscore.training.tune_xgboost (Experiment 3 sweep)."""

from pathlib import Path

import pytest

from signalscore.training.tune_xgboost import format_next_command, main
from tests.training.test_train_baseline import make_rows, write_rows


def test_format_next_command_reproduces_the_config_as_a_train_baseline_invocation() -> None:
    config = {"max_depth": 3, "learning_rate": 0.1}

    command = format_next_command(config, Path("train.jsonl"), Path("val.jsonl"))

    assert "signalscore.training.train_baseline" in command
    assert "--train train.jsonl" in command
    assert "--val val.jsonl" in command
    assert "--model-type xgboost" in command
    assert '{"max_depth": 3, "learning_rate": 0.1}' in command


def test_main_is_pure_search_and_report_no_persistence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression guard mirroring tune_baseline.py's own version of this test:
    sweeping must never save a model artifact, write metrics, or touch
    MLflow -- it only prints a ranked trial table and the train_baseline.py
    command to run next.

    TruncatedSVD's default n_components=100 exceeds this tiny synthetic
    corpus's row/feature counts, but sklearn silently caps the effective
    rank rather than raising (verified empirically) -- so this stays a fast,
    real (non-mocked) exercise of the search/report mechanics with a small
    --budget, not a real hyperparameter sweep.
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
    assert "--model-type xgboost" in out
