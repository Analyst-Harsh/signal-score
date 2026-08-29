"""Tests for signalscore.training.split."""

from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from signalscore.features.labels import Priority
from signalscore.features.schema import FeatureRow
from signalscore.training.split import (
    assemble_splits,
    determine_eval_cutoff,
    freeze_eval_set,
    hash_split,
    issue_bucket,
    load_frozen_eval_set,
)


def _row(issue_number: int, created_at: datetime) -> FeatureRow:
    return FeatureRow(
        issue_number=issue_number,
        created_at=created_at,
        label=Priority.P2_BACKLOG,
        text="something",
        is_member_plus=False,
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


def test_issue_bucket_is_deterministic() -> None:
    assert issue_bucket(12345) == issue_bucket(12345)


def test_issue_bucket_is_roughly_uniform_over_many_issue_numbers() -> None:
    buckets = [issue_bucket(n) for n in range(20_000)]
    below_cutoff = sum(1 for b in buckets if b < 19)

    # Expect ~19% of issue numbers to land below the val cutoff bucket.
    assert 0.16 < below_cutoff / len(buckets) < 0.22


def test_determine_eval_cutoff_selects_the_most_recent_20_percent_boundary() -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [_row(n, base + timedelta(days=n)) for n in range(100)]

    cutoff = determine_eval_cutoff(rows, eval_fraction=0.20)

    at_or_after_cutoff = sum(1 for row in rows if row.created_at >= cutoff)
    assert at_or_after_cutoff == 20


def test_freeze_and_load_eval_set_round_trips(tmp_path: Path) -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [_row(n, base + timedelta(days=n)) for n in range(10)]
    cutoff = base + timedelta(days=8)
    out_path = tmp_path / "eval.jsonl"

    freeze_eval_set(rows, cutoff, out_path)
    loaded = load_frozen_eval_set(out_path)

    assert {row.issue_number for row in loaded} == {8, 9}


def test_hash_split_partitions_with_no_overlap_and_lands_near_target_val_fraction() -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [_row(n, base) for n in range(5000)]

    train, val = hash_split(rows)

    train_ids = {r.issue_number for r in train}
    val_ids = {r.issue_number for r in val}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {r.issue_number for r in rows}
    assert 0.15 < len(val_ids) / len(rows) < 0.23


def test_assemble_splits_partitions_all_rows_with_no_overlap_and_no_drop(tmp_path: Path) -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    rows = [_row(n, base + timedelta(days=n)) for n in range(100)]
    cutoff = determine_eval_cutoff(rows, eval_fraction=0.20)
    frozen_path = tmp_path / "eval.jsonl"
    freeze_eval_set(rows, cutoff, frozen_path)

    assignments = assemble_splits(rows, frozen_path)

    assert set(assignments.keys()) == {r.issue_number for r in rows}
    counts = Counter(assignments.values())
    assert set(counts.keys()) == {"train", "val", "test"}
    assert counts["test"] == 20
