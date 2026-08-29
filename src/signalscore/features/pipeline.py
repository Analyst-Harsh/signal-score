"""The shared train/serve feature-extraction entry point.

`build_features` is the frozen serving contract: training, evaluation,
serving, and monitoring must all import and call it directly, never a private
re-implementation, so train/serve skew is structurally impossible.
"""

from collections.abc import Iterable
from typing import Any, TypedDict

from signalscore.features.metadata import extract_metadata_features
from signalscore.features.schema import FeatureRow
from signalscore.features.text import extract_text_features


class TextDiagnostics(TypedDict):
    title_chars: int
    body_chars: int
    body_words: int
    has_code_block: bool
    has_stack_trace: bool
    has_url: bool
    has_logline: bool
    has_excl: bool
    code_ratio: float
    any_lex: bool
    is_ci_flake_shaped: bool


class FeatureBundle(TypedDict):
    text: str  # -> TF-IDF input, fit only on the train split (training's job)
    is_member_plus: bool  # -> model feature
    diagnostics: TextDiagnostics  # never concatenated into the model matrix


def build_features(title: str, body: str, author_association: str | None) -> FeatureBundle:
    text_features = extract_text_features(title, body)
    metadata = extract_metadata_features(author_association)
    diagnostics: TextDiagnostics = {
        "title_chars": text_features["title_chars"],
        "body_chars": text_features["body_chars"],
        "body_words": text_features["body_words"],
        "has_code_block": text_features["has_code_block"],
        "has_stack_trace": text_features["has_stack_trace"],
        "has_url": text_features["has_url"],
        "has_logline": text_features["has_logline"],
        "has_excl": text_features["has_excl"],
        "code_ratio": text_features["code_ratio"],
        "any_lex": text_features["any_lex"],
        "is_ci_flake_shaped": text_features["is_ci_flake_shaped"],
    }
    return {
        "text": text_features["text"],
        "is_member_plus": metadata["is_member_plus"],
        "diagnostics": diagnostics,
    }


def build_feature_frame(issues: Iterable[dict[str, Any]]) -> list[FeatureRow]:
    """The only caller of build_features() during dataset assembly.

    `issues` are kept rows from `resolve_and_filter` — each must carry
    `number`, `title`, `body`, `author_association`, `created_at`, and the
    resolved `_priority`.
    """
    rows: list[FeatureRow] = []
    for issue in issues:
        bundle = build_features(issue["title"], issue["body"], issue.get("author_association"))
        rows.append(
            FeatureRow(
                issue_number=issue["number"],
                created_at=issue["created_at"],
                label=issue["_priority"],
                text=bundle["text"],
                is_member_plus=bundle["is_member_plus"],
                **bundle["diagnostics"],
            )
        )
    return rows
