"""Whitelist-only GitHub metadata features.

Everything except `author_association` is banned as leakage or repo-specific
(see docs/priority-signal-decision.md's serving-contract allow list) — this
function returns exactly one bit, nothing else.
"""

from typing import TypedDict

_MEMBER_PLUS = {"MEMBER", "OWNER", "COLLABORATOR"}


class MetadataFeatures(TypedDict):
    is_member_plus: bool


def extract_metadata_features(author_association: str | None) -> MetadataFeatures:
    return {"is_member_plus": author_association in _MEMBER_PLUS}
