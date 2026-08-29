"""Frozen time-based eval split, and a deterministic hash split for train/val.

A random/stratified split was measured (docs/priority-signal-decision.md) to
inflate the majority-class baseline from 20.8% to 40.9% by letting the model
exploit temporal drift in label prevalence — so eval/test is a frozen, most-
recent-20%-by-`created_at` slice, computed once and never regenerated. Only
the remaining 80% candidate pool is hash-split, mirroring
docs/signalscore-design.md's exclusion pattern: frozen ids are always derived
from the frozen file itself, never a separately maintained list.
"""

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from signalscore.features.schema import FeatureRow

EVAL_FRACTION = 0.20
# Target overall split ~65/15/20 train/val/test. Test is 20% exactly (frozen).
# Val needs to be ~15/80 = 18.75% of the 80% candidate pool -> nearest bucket cutoff 19,
# giving overall test 20.0% / val 15.2% / train 64.8%.
VAL_BUCKET_CUTOFF = 19


def issue_bucket(issue_number: int) -> int:
    digest = hashlib.sha256(str(issue_number).encode()).hexdigest()
    return int(digest, 16) % 100


def determine_eval_cutoff(
    rows: Sequence[FeatureRow], eval_fraction: float = EVAL_FRACTION
) -> datetime:
    """The `created_at` threshold at/after which the most recent `eval_fraction` of
    rows fall. Computed once over the full cleaned pool; never recomputed on refetch.
    """
    sorted_dates = sorted(row.created_at for row in rows)
    cutoff_index = int(len(sorted_dates) * (1 - eval_fraction))
    return sorted_dates[cutoff_index]


def freeze_eval_set(rows: Sequence[FeatureRow], cutoff: datetime, out_path: Path) -> None:
    eval_rows = [row for row in rows if row.created_at >= cutoff]
    with out_path.open("w") as f:
        for row in eval_rows:
            f.write(row.model_dump_json() + "\n")


def _load_feature_rows(path: Path) -> list[FeatureRow]:
    with path.open() as f:
        return [FeatureRow.model_validate_json(line) for line in f if line.strip()]


def load_frozen_eval_set(path: Path) -> list[FeatureRow]:
    return _load_feature_rows(path)


def hash_split(
    candidate_pool: Sequence[FeatureRow], val_bucket_cutoff: int = VAL_BUCKET_CUTOFF
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    """candidate_pool must already exclude frozen eval issue numbers."""
    train: list[FeatureRow] = []
    val: list[FeatureRow] = []
    for row in candidate_pool:
        bucket = issue_bucket(row.issue_number)
        (val if bucket < val_bucket_cutoff else train).append(row)
    return train, val


def assemble_splits(rows: Sequence[FeatureRow], frozen_eval_path: Path) -> dict[int, str]:
    frozen_eval_ids = {row.issue_number for row in load_frozen_eval_set(frozen_eval_path)}
    candidate_pool = [row for row in rows if row.issue_number not in frozen_eval_ids]
    train, val = hash_split(candidate_pool)

    assignments: dict[int, str] = {row.issue_number: "train" for row in train}
    assignments.update({row.issue_number: "val" for row in val})
    assignments.update(dict.fromkeys(frozen_eval_ids, "test"))
    return assignments


def write_split_jsonl(
    rows: Sequence[FeatureRow], assignments: Mapping[int, str], split: str, out_path: Path
) -> None:
    """Writes rows whose assignment matches `split`, one FeatureRow JSON per line —
    mirrors freeze_eval_set's write pattern.
    """
    if split == "test":
        raise ValueError(
            "write_split_jsonl must never be called with split='test'; "
            "test rows come only from freeze_eval_set"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            if assignments.get(row.issue_number) == split:
                f.write(row.model_dump_json() + "\n")


def materialize_train_val(
    features_path: Path, frozen_eval_path: Path, train_out: Path, val_out: Path
) -> None:
    """Rebuilds train.jsonl/val.jsonl from the already-materialized features_v1.jsonl +
    the frozen eval set alone -- no raw data or full raw-pipeline rerun needed. Both
    inputs are the exact two files DVC already tracks, so this works right after
    `dvc pull`. Reuses assemble_splits, the single source of truth for split assignment.
    """
    frame = _load_feature_rows(features_path)
    assignments = assemble_splits(frame, frozen_eval_path)
    write_split_jsonl(frame, assignments, "train", train_out)
    write_split_jsonl(frame, assignments, "val", val_out)
