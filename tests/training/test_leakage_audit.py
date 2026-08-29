"""Tests for signalscore.training.leakage_audit."""

from datetime import UTC, datetime
from typing import Any

from signalscore.features.labels import LabelMappingConfig, Priority
from signalscore.features.schema import FeatureRow
from signalscore.training.leakage_audit import (
    audit_author_concentration,
    audit_label_drift,
    find_near_duplicates,
)

TRAIN_CONFIG = LabelMappingConfig.model_validate(
    {
        "version": 1,
        "repos": {
            "kubernetes/kubernetes": {
                "role": "train",
                "mapping": {
                    "priority/critical-urgent": "P0_critical",
                    "priority/important-soon": "P1_soon",
                    "priority/backlog": "P2_backlog",
                },
                "excluded_labels": [],
            }
        },
    }
)


def _row(issue_number: int, text: str) -> FeatureRow:
    return FeatureRow(
        issue_number=issue_number,
        created_at=datetime(2020, 1, 1, tzinfo=UTC),
        label=Priority.P2_BACKLOG,
        text=text,
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


def test_find_near_duplicates_flags_identical_text_pair_and_marks_cross_split() -> None:
    rows = [
        _row(1, "the pod crashes on startup every time"),
        _row(2, "the pod crashes on startup every time"),
        _row(3, "completely unrelated flaky test in e2e suite"),
    ]
    split_of = {1: "train", 2: "test", 3: "train"}

    pairs = find_near_duplicates(rows, split_of, threshold=0.85)

    assert len(pairs) == 1
    assert {pairs[0].issue_a, pairs[0].issue_b} == {1, 2}
    assert pairs[0].cross_split is True


def test_find_near_duplicates_does_not_flag_dissimilar_pair() -> None:
    rows = [
        _row(1, "the pod crashes on startup every time"),
        _row(2, "completely unrelated flaky test in e2e suite"),
    ]
    split_of = {1: "train", 2: "train"}

    pairs = find_near_duplicates(rows, split_of, threshold=0.85)

    assert pairs == []


def test_audit_author_concentration_flags_dominant_author_split() -> None:
    author_by_number = dict.fromkeys(range(1, 21), "same-author")
    split_of = dict.fromkeys(range(1, 21), "train")

    reports = audit_author_concentration(author_by_number, split_of)

    assert len(reports) == 1
    assert reports[0].flagged is True
    assert reports[0].top1_author_share == 1.0


def test_audit_author_concentration_does_not_flag_diverse_split() -> None:
    # 100 distinct authors, one issue each: HHI = 1/100 = 0.01, below the default
    # 0.02 threshold. A smaller sample's HHI floor (1/n_authors) would flag as
    # "concentrated" purely from small-sample arithmetic, not real dominance.
    author_by_number = {n: f"author-{n}" for n in range(1, 101)}
    split_of = dict.fromkeys(range(1, 101), "train")

    reports = audit_author_concentration(author_by_number, split_of)

    assert len(reports) == 1
    assert reports[0].flagged is False


def test_audit_label_drift_flags_disagreement() -> None:
    # Live labels now resolve to P0, but the issue was originally queried as backlog.
    kept_rows: list[dict[str, Any]] = [
        {
            "number": 1,
            "_queried_labels": {"priority/backlog"},
            "_priority": Priority.P0_CRITICAL,
        }
    ]

    drifted = audit_label_drift(kept_rows, "kubernetes/kubernetes", TRAIN_CONFIG)

    assert len(drifted) == 1
    assert drifted[0].issue_number == 1


def test_audit_label_drift_clean_when_consistent() -> None:
    kept_rows: list[dict[str, Any]] = [
        {
            "number": 1,
            "_queried_labels": {"priority/backlog"},
            "_priority": Priority.P2_BACKLOG,
        }
    ]

    drifted = audit_label_drift(kept_rows, "kubernetes/kubernetes", TRAIN_CONFIG)

    assert drifted == []
