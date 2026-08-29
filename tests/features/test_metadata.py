"""Tests for signalscore.features.metadata."""

from signalscore.features.metadata import extract_metadata_features


def test_is_member_plus_true_for_member_owner_collaborator() -> None:
    for association in ("MEMBER", "OWNER", "COLLABORATOR"):
        assert extract_metadata_features(association)["is_member_plus"] is True


def test_is_member_plus_false_for_contributor_or_none_association() -> None:
    for association in ("CONTRIBUTOR", "NONE"):
        assert extract_metadata_features(association)["is_member_plus"] is False


def test_is_member_plus_false_when_author_association_is_none() -> None:
    assert extract_metadata_features(None)["is_member_plus"] is False
