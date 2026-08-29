"""Tests for signalscore.features.labels."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from signalscore.features.labels import Priority, load_label_mapping, resolve_priority

VALID_YAML = """
version: 1
repos:
  kubernetes/kubernetes:
    role: train
    mapping:
      priority/critical-urgent: P0_critical
      priority/important-soon: P1_soon
      priority/backlog: P2_backlog
    excluded_labels:
      - priority/important-longterm
      - priority/awaiting-more-evidence
"""


def test_load_label_mapping_round_trips_via_yaml(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text(VALID_YAML)

    config = load_label_mapping(path)

    assert config.version == 1
    repo = config.repos["kubernetes/kubernetes"]
    assert repo.role == "train"
    assert repo.mapping["priority/critical-urgent"] == Priority.P0_CRITICAL
    assert repo.excluded_labels == [
        "priority/important-longterm",
        "priority/awaiting-more-evidence",
    ]


def test_load_label_mapping_rejects_unknown_canonical_value(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text("""
version: 1
repos:
  kubernetes/kubernetes:
    role: train
    mapping:
      priority/critical-urgent: P9_bogus
""")

    with pytest.raises(ValidationError):
        load_label_mapping(path)


def test_resolve_priority_returns_priority_for_single_known_label(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text(VALID_YAML)
    config = load_label_mapping(path)

    result = resolve_priority(
        ["priority/critical-urgent", "kind/bug"], "kubernetes/kubernetes", config
    )

    assert result == Priority.P0_CRITICAL


def test_resolve_priority_returns_none_for_zero_priority_labels(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text(VALID_YAML)
    config = load_label_mapping(path)

    result = resolve_priority(["kind/bug"], "kubernetes/kubernetes", config)

    assert result is None


def test_resolve_priority_returns_none_for_multiple_priority_labels(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text(VALID_YAML)
    config = load_label_mapping(path)

    result = resolve_priority(
        ["priority/critical-urgent", "priority/backlog"], "kubernetes/kubernetes", config
    )

    assert result is None


def test_resolve_priority_returns_none_for_excluded_label(tmp_path: Path) -> None:
    path = tmp_path / "label_mapping.yaml"
    path.write_text(VALID_YAML)
    config = load_label_mapping(path)

    result = resolve_priority(["priority/important-longterm"], "kubernetes/kubernetes", config)

    assert result is None
