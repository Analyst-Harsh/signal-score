"""Tests for signalscore.features.loading."""

import json
from pathlib import Path
from typing import Any

import pytest

from signalscore.features.labels import LabelMappingConfig
from signalscore.features.loading import (
    dedupe_by_issue_number,
    load_raw_jsonl,
    resolve_and_filter,
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
                "excluded_labels": [
                    "priority/important-longterm",
                    "priority/awaiting-more-evidence",
                ],
            }
        },
    }
)

HOLDOUT_CONFIG = LabelMappingConfig.model_validate(
    {
        "version": 1,
        "repos": {"rust-lang/rust": {"role": "holdout", "mapping": {}, "excluded_labels": []}},
    }
)


def _label(name: str) -> dict[str, Any]:
    return {"name": name}


def _issue(
    number: int, queried_label: str, live_labels: list[str], **overrides: Any
) -> dict[str, Any]:
    row = {
        "number": number,
        "title": "something broke",
        "body": "steps to reproduce",
        "author_association": "CONTRIBUTOR",
        "created_at": "2020-01-01T00:00:00Z",
        "labels": [_label(name) for name in live_labels],
        "_queried_label": queried_label,
    }
    row.update(overrides)
    return row


def test_load_raw_jsonl_reads_multiple_files(tmp_path: Path) -> None:
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    file_a.write_text(json.dumps({"number": 1}) + "\n")
    file_b.write_text(json.dumps({"number": 2}) + "\n")

    result = list(load_raw_jsonl([file_a, file_b]))

    assert result == [{"number": 1}, {"number": 2}]


def test_dedupe_by_issue_number_collapses_duplicates_and_merges_queried_labels() -> None:
    rows = [
        _issue(1, "priority/critical-urgent", ["priority/critical-urgent", "priority/backlog"]),
        _issue(1, "priority/backlog", ["priority/critical-urgent", "priority/backlog"]),
        _issue(2, "priority/backlog", ["priority/backlog"]),
    ]

    result = dedupe_by_issue_number(rows)

    assert set(result.keys()) == {1, 2}
    assert result[1]["_queried_labels"] == {"priority/critical-urgent", "priority/backlog"}
    assert "_queried_label" not in result[1]
    assert result[2]["_queried_labels"] == {"priority/backlog"}


def test_resolve_and_filter_drops_multi_priority_label_rows() -> None:
    rows_by_number = {
        1: _issue(1, "priority/critical-urgent", ["priority/critical-urgent", "priority/backlog"]),
    }

    kept, report = resolve_and_filter(
        rows_by_number, "kubernetes/kubernetes", TRAIN_CONFIG, total_input_rows=1
    )

    assert kept == []
    assert report.dropped_multi_priority_label == 1
    assert report.kept == 0


def test_resolve_and_filter_drops_excluded_label_rows() -> None:
    rows_by_number = {
        1: _issue(
            1,
            "priority/important-longterm",
            ["priority/important-longterm", "priority/backlog"],
        ),
    }

    kept, report = resolve_and_filter(
        rows_by_number, "kubernetes/kubernetes", TRAIN_CONFIG, total_input_rows=1
    )

    assert kept == []
    assert report.dropped_excluded_label == {"priority/important-longterm": 1}


def test_resolve_and_filter_keeps_valid_rows_and_coerces_none_body() -> None:
    rows_by_number = {
        1: _issue(1, "priority/backlog", ["priority/backlog"], body=None),
    }

    kept, report = resolve_and_filter(
        rows_by_number, "kubernetes/kubernetes", TRAIN_CONFIG, total_input_rows=1
    )

    assert len(kept) == 1
    assert kept[0]["body"] == ""
    assert report.kept == 1


def test_resolve_and_filter_raises_for_holdout_repo() -> None:
    rows_by_number = {1: _issue(1, "P-high", ["P-high"])}

    with pytest.raises(ValueError, match="holdout"):
        resolve_and_filter(rows_by_number, "rust-lang/rust", HOLDOUT_CONFIG, total_input_rows=1)
