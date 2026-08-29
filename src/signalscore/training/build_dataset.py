"""CLI orchestrator: raw JSONL -> assembled feature matrix -> frozen split -> audit reports.

Run via:
    uv run python -m signalscore.training.build_dataset

Mirrors collection/fetch_issues.py's `--repo` CLI shape. The manual-review gate
and `dvc add`/`git tag eval-set-v1` steps stay outside this script — the frozen
eval set is meant to be created once and never silently regenerated.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from signalscore.features.labels import LabelMappingConfig, load_label_mapping
from signalscore.features.loading import (
    dedupe_by_issue_number,
    load_raw_jsonl,
    resolve_and_filter,
)
from signalscore.features.pipeline import build_feature_frame
from signalscore.features.schema import FeatureRow
from signalscore.training.leakage_audit import (
    audit_author_concentration,
    audit_label_drift,
    find_near_duplicates,
)
from signalscore.training.split import (
    assemble_splits,
    determine_eval_cutoff,
    freeze_eval_set,
    materialize_train_val,
)

DEFAULT_REPO = "kubernetes/kubernetes"
# Last known-good count from docs/v0-feature-selection.md. A refetch is expected
# to drift this slightly; only warn, never hard-fail, on drift.
EXPECTED_KEPT_ROWS = 11_899


def run_pipeline(
    raw_paths: list[Path],
    repo: str,
    label_mapping_path: Path,
    features_out: Path,
    frozen_eval_path: Path,
    split_out: Path,
    audit_dir: Path,
    train_out: Path,
    val_out: Path,
) -> None:
    config = load_label_mapping(label_mapping_path)
    raw_rows = list(load_raw_jsonl(raw_paths))
    deduped = dedupe_by_issue_number(raw_rows)
    kept_rows, report = resolve_and_filter(deduped, repo, config, total_input_rows=len(raw_rows))
    print(report.model_dump_json(indent=2))
    if abs(report.kept - EXPECTED_KEPT_ROWS) > EXPECTED_KEPT_ROWS * 0.05:
        print(f"WARNING: kept row count {report.kept} drifted from expected {EXPECTED_KEPT_ROWS}")

    frame = build_feature_frame(kept_rows)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    with features_out.open("w") as f:
        for row in frame:
            f.write(row.model_dump_json() + "\n")

    cutoff = determine_eval_cutoff(frame)
    frozen_eval_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_eval_set(frame, cutoff, frozen_eval_path)

    materialize_train_val(features_out, frozen_eval_path, train_out, val_out)

    assignments = assemble_splits(frame, frozen_eval_path)
    split_out.parent.mkdir(parents=True, exist_ok=True)
    with split_out.open("w") as f:
        f.write("issue_number,split\n")
        for issue_number, split in sorted(assignments.items()):
            f.write(f"{issue_number},{split}\n")

    audit_dir.mkdir(parents=True, exist_ok=True)
    _write_near_duplicates_report(frame, assignments, audit_dir / "near_duplicates_v1.csv")
    _write_author_concentration_report(
        kept_rows, assignments, audit_dir / "author_concentration_v1.json"
    )
    drifted = _write_label_drift_report(kept_rows, repo, config, audit_dir / "label_drift_v1.csv")
    if drifted:
        raise SystemExit(
            f"{drifted} issue(s) show label drift between _queried_label and live "
            f"labels — see {audit_dir / 'label_drift_v1.csv'}. Investigate before "
            "freezing this split."
        )


def _write_near_duplicates_report(
    frame: list[FeatureRow], assignments: dict[int, str], out_path: Path
) -> None:
    pairs = find_near_duplicates(frame, assignments)
    with out_path.open("w") as f:
        f.write("issue_a,issue_b,split_a,split_b,cosine_similarity,cross_split,decision\n")
        for pair in pairs:
            f.write(
                f"{pair.issue_a},{pair.issue_b},{pair.split_a},{pair.split_b},"
                f"{pair.cosine_similarity:.4f},{pair.cross_split},\n"
            )


def _write_author_concentration_report(
    kept_rows: list[dict[str, Any]], assignments: dict[int, str], out_path: Path
) -> None:
    author_by_number = {row["number"]: row["user"]["login"] for row in kept_rows if row.get("user")}
    reports = audit_author_concentration(author_by_number, assignments)
    with out_path.open("w") as f:
        json.dump([vars(report) for report in reports], f, indent=2)


def _write_label_drift_report(
    kept_rows: list[dict[str, Any]], repo: str, config: LabelMappingConfig, out_path: Path
) -> int:
    drifted = audit_label_drift(kept_rows, repo, config)
    with out_path.open("w") as f:
        f.write("issue_number,queried_labels,resolved_label\n")
        for row in drifted:
            queried = "|".join(sorted(row.queried_labels))
            f.write(f"{row.issue_number},{queried},{row.resolved_label.value}\n")
    return len(drifted)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--label-mapping", type=Path, default=Path("config/label_mapping.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audit"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    repo_dir_name = args.repo.replace("/", "-")
    raw_dir = args.raw_dir / repo_dir_name
    raw_paths = sorted(raw_dir.glob("*.jsonl"))
    if not raw_paths:
        raise SystemExit(f"no raw JSONL files found under {raw_dir}")

    run_pipeline(
        raw_paths=raw_paths,
        repo=args.repo,
        label_mapping_path=args.label_mapping,
        features_out=args.processed_dir / repo_dir_name / "features_v1.jsonl",
        frozen_eval_path=args.processed_dir / repo_dir_name / "eval_set_v1.jsonl",
        split_out=args.splits_dir / repo_dir_name / "split_assignments_v1.csv",
        audit_dir=args.audit_dir / repo_dir_name,
        train_out=args.processed_dir / repo_dir_name / "train.jsonl",
        val_out=args.processed_dir / repo_dir_name / "val.jsonl",
    )


if __name__ == "__main__":
    main()
