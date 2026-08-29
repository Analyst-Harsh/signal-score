"""Integration test for signalscore.training.build_dataset's run_pipeline.

Exercises the full wiring — load -> dedupe -> filter -> featurize -> freeze ->
split -> audit -> write — against tiny synthetic fixtures, since this module
is orchestration glue with no independent logic of its own (mirroring
collection/fetch_issues.py's untested main()).
"""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from signalscore.training.build_dataset import run_pipeline

LABEL_MAPPING_YAML = """
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


def _raw_issue(number: int, label: str, created_at: datetime) -> dict[str, object]:
    return {
        "number": number,
        "title": f"issue number {number}",
        "body": f"body text for issue {number}",
        "author_association": "CONTRIBUTOR",
        "created_at": created_at.isoformat(),
        "user": {"login": f"user-{number}"},
        "labels": [{"name": label}],
        "_queried_label": label,
    }


def test_run_pipeline_end_to_end_produces_consistent_outputs(tmp_path: Path) -> None:
    base = datetime(2020, 1, 1, tzinfo=UTC)
    labels = [
        "priority/critical-urgent",
        "priority/important-soon",
        "priority/backlog",
    ]
    raw_rows = [_raw_issue(n, labels[n % 3], base + timedelta(days=n)) for n in range(1, 51)]

    raw_path = tmp_path / "raw.jsonl"
    with raw_path.open("w") as f:
        for row in raw_rows:
            f.write(json.dumps(row) + "\n")

    label_mapping_path = tmp_path / "label_mapping.yaml"
    label_mapping_path.write_text(LABEL_MAPPING_YAML)

    features_out = tmp_path / "features_v1.jsonl"
    frozen_eval_path = tmp_path / "eval_set_v1.jsonl"
    split_out = tmp_path / "split_assignments_v1.csv"
    audit_dir = tmp_path / "audit"

    run_pipeline(
        raw_paths=[raw_path],
        repo="kubernetes/kubernetes",
        label_mapping_path=label_mapping_path,
        features_out=features_out,
        frozen_eval_path=frozen_eval_path,
        split_out=split_out,
        audit_dir=audit_dir,
    )

    feature_lines = features_out.read_text().splitlines()
    assert len(feature_lines) == 50

    with split_out.open() as f:
        rows = list(csv.DictReader(f))
    split_issue_numbers = {int(row["issue_number"]) for row in rows}
    assert split_issue_numbers == set(range(1, 51))
    assert {row["split"] for row in rows} <= {"train", "val", "test"}

    assert (audit_dir / "near_duplicates_v1.csv").exists()
    assert (audit_dir / "author_concentration_v1.json").exists()
    assert (audit_dir / "label_drift_v1.csv").exists()

    # No re-triage between pulls in this fixture, so the drift report is empty
    # beyond its header row.
    drift_lines = (audit_dir / "label_drift_v1.csv").read_text().splitlines()
    assert len(drift_lines) == 1
