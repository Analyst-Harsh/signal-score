"""Tests for signalscore.features.pipeline — the frozen train/serve contract."""

import inspect
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from signalscore.features.labels import Priority
from signalscore.features.pipeline import build_feature_frame, build_features
from signalscore.features.schema import FeatureRow

VALID_FEATURE_ROW_KWARGS: dict[str, Any] = {
    "issue_number": 1,
    "created_at": datetime(2020, 1, 1, tzinfo=UTC),
    "label": Priority.P0_CRITICAL,
    "text": "something",
    "is_member_plus": True,
    "title_chars": 1,
    "body_chars": 1,
    "body_words": 1,
    "has_code_block": False,
    "has_stack_trace": False,
    "has_url": False,
    "has_logline": False,
    "has_excl": False,
    "code_ratio": 0.0,
    "any_lex": False,
    "is_ci_flake_shaped": False,
}


def test_build_features_signature_is_frozen() -> None:
    assert set(inspect.signature(build_features).parameters) == {
        "title",
        "body",
        "author_association",
    }


def test_build_features_separates_model_features_from_diagnostics() -> None:
    bundle = build_features("panic in kubelet", "goroutine dump here", "MEMBER")

    assert set(bundle.keys()) == {"text", "is_member_plus", "diagnostics"}
    assert "has_code_block" not in bundle


def test_build_features_training_and_serving_calls_are_byte_identical() -> None:
    issue: dict[str, Any] = {
        "number": 1,
        "title": "panic in kubelet",
        "body": "goroutine dump here",
        "author_association": "MEMBER",
        "created_at": "2020-01-01T00:00:00Z",
        "_priority": Priority.P0_CRITICAL,
    }

    serving_call = build_features(issue["title"], issue["body"], issue["author_association"])
    training_call = build_feature_frame([issue])[0]

    assert training_call.text == serving_call["text"]
    assert training_call.is_member_plus == serving_call["is_member_plus"]
    assert training_call.has_code_block == serving_call["diagnostics"]["has_code_block"]


def test_build_feature_frame_attaches_issue_metadata() -> None:
    issue = {
        "number": 42,
        "title": "title",
        "body": "body",
        "author_association": "NONE",
        "created_at": "2021-06-15T00:00:00Z",
        "_priority": Priority.P2_BACKLOG,
    }

    rows = build_feature_frame([issue])

    assert len(rows) == 1
    assert rows[0].issue_number == 42
    assert rows[0].label == Priority.P2_BACKLOG
    assert rows[0].created_at == datetime(2021, 6, 15, tzinfo=UTC)


def test_feature_row_accepts_only_whitelisted_fields() -> None:
    FeatureRow(**VALID_FEATURE_ROW_KWARGS)


def test_feature_row_rejects_unwhitelisted_fields() -> None:
    with pytest.raises(ValidationError):
        FeatureRow.model_validate(
            {**VALID_FEATURE_ROW_KWARGS, "updated_at": "2020-01-02T00:00:00Z"}
        )
