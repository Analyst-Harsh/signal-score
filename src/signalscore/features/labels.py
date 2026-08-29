"""Canonical priority taxonomy and the repo-parametrized label mapping.

The mapping is a hand-authored, versioned YAML file (`config/label_mapping.yaml`),
not inferred or embedding-matched. Repos marked `role: holdout` in that file carry
a different label taxonomy and must never flow into training — see
`signalscore.features.loading.resolve_and_filter`, which enforces this.
"""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class Priority(StrEnum):
    P0_CRITICAL = "P0_critical"
    P1_SOON = "P1_soon"
    P2_BACKLOG = "P2_backlog"


class RepoLabelMapping(BaseModel):
    role: Literal["train", "holdout"]
    mapping: dict[str, Priority]
    excluded_labels: list[str] = []


class LabelMappingConfig(BaseModel):
    version: int
    repos: dict[str, RepoLabelMapping]


def load_label_mapping(path: Path = Path("config/label_mapping.yaml")) -> LabelMappingConfig:
    raw = yaml.safe_load(path.read_text())
    return LabelMappingConfig.model_validate(raw)


def resolve_priority(
    live_labels: list[str], repo: str, config: LabelMappingConfig
) -> Priority | None:
    """Return the canonical Priority for one issue's live label list.

    None means "drop this row": zero priority/* labels present, more than one
    priority/* label present (ambiguous), or the single label found is excluded
    from the taxonomy (e.g. important-longterm).
    """
    repo_config = config.repos[repo]
    if any(label in repo_config.excluded_labels for label in live_labels):
        return None
    matched = [repo_config.mapping[label] for label in live_labels if label in repo_config.mapping]
    if len(matched) != 1:
        return None
    return matched[0]
