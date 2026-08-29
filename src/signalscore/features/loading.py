"""Raw JSONL loading, cross-file dedup, and label resolution/row-dropping.

Rows returned by `resolve_and_filter` retain audit-time-only fields (`labels`,
`_queried_labels`) alongside the whitelisted serving fields — the whitelist is
enforced structurally downstream, at the `FeatureRow`/`build_features` boundary
in `signalscore.features.pipeline`, not here.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from signalscore.features.labels import LabelMappingConfig, resolve_priority


class DropReport(BaseModel):
    total_input_rows: int
    total_unique_issues: int
    dropped_multi_priority_label: int
    dropped_excluded_label: dict[str, int]
    kept: int


def load_raw_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open() as f:
            for line in f:
                yield json.loads(line)


def dedupe_by_issue_number(rows: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Collapse rows sharing `number`, merging every `_queried_label` seen into a set.

    An issue queried under two labels shows up once per source file with the same
    current `labels` snapshot but a different `_queried_label` — losing either value
    would weaken the label-drift audit, which needs every label an issue was ever
    fetched under.
    """
    queried_labels_by_number: dict[int, set[str]] = {}
    latest_by_number: dict[int, dict[str, Any]] = {}
    for row in rows:
        number = row["number"]
        queried_labels_by_number.setdefault(number, set()).add(row["_queried_label"])
        latest_by_number[number] = row

    result: dict[int, dict[str, Any]] = {}
    for number, row in latest_by_number.items():
        merged = {**row}
        del merged["_queried_label"]
        merged["_queried_labels"] = queried_labels_by_number[number]
        result[number] = merged
    return result


def resolve_and_filter(
    rows_by_number: dict[int, dict[str, Any]],
    repo: str,
    config: LabelMappingConfig,
    total_input_rows: int,
) -> tuple[list[dict[str, Any]], DropReport]:
    """Drop rows whose live labels don't resolve to exactly one canonical Priority.

    Raises ValueError if `repo` is configured as `role: holdout` — the one
    enforcement point for CLAUDE.md's hard guardrail against training on a
    holdout-taxonomy repo.
    """
    repo_config = config.repos[repo]
    if repo_config.role != "train":
        raise ValueError(
            f"repo {repo!r} is role={repo_config.role!r}, not 'train' — "
            "refusing to build a training matrix from a holdout-taxonomy repo"
        )

    kept: list[dict[str, Any]] = []
    dropped_multi_priority_label = 0
    dropped_excluded_label: dict[str, int] = {}

    for row in rows_by_number.values():
        live_labels = [label["name"] for label in row["labels"]]
        priority = resolve_priority(live_labels, repo, config)
        if priority is None:
            excluded_hit = next(
                (label for label in live_labels if label in repo_config.excluded_labels), None
            )
            if excluded_hit is not None:
                dropped_excluded_label[excluded_hit] = (
                    dropped_excluded_label.get(excluded_hit, 0) + 1
                )
            else:
                dropped_multi_priority_label += 1
            continue
        kept_row = {**row}
        kept_row["title"] = kept_row.get("title") or ""
        kept_row["body"] = kept_row.get("body") or ""
        kept_row["_priority"] = priority
        kept.append(kept_row)

    report = DropReport(
        total_input_rows=total_input_rows,
        total_unique_issues=len(rows_by_number),
        dropped_multi_priority_label=dropped_multi_priority_label,
        dropped_excluded_label=dropped_excluded_label,
        kept=len(kept),
    )
    return kept, report
