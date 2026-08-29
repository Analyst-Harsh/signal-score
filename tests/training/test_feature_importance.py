"""Tests for signalscore.training.feature_importance."""

from collections import Counter
from pathlib import Path

import joblib
import pytest

from signalscore.features.labels import Priority
from signalscore.training.feature_importance import (
    combined_feature_names,
    top_features_by_class,
)
from signalscore.training.train_baseline import BaselineArtifact, run_training_pipeline
from tests.training.test_train_baseline import make_rows, write_rows


def _fit_artifact(tmp_path: Path) -> BaselineArtifact:
    train_rows = make_rows(n_per_class=8)
    val_rows = make_rows(n_per_class=3, start_issue=1000)
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_rows(train_rows, train_path)
    write_rows(val_rows, val_path)

    model_out = tmp_path / "model.joblib"
    run_training_pipeline(train_path, val_path, model_out, tmp_path / "metrics.json")
    return joblib.load(model_out)  # pyright: ignore


def test_combined_feature_names_appends_is_member_plus_last(tmp_path: Path) -> None:
    artifact = _fit_artifact(tmp_path)

    names = combined_feature_names(artifact)

    word_vocab_size = len(artifact.word_vectorizer.vocabulary_)  # pyright: ignore
    char_vocab_size = len(artifact.char_vectorizer.vocabulary_)  # pyright: ignore
    assert len(names) == word_vocab_size + char_vocab_size + 1
    assert names[-1] == "is_member_plus"


def test_top_features_by_class_ranks_obvious_separating_word_first(tmp_path: Path) -> None:
    artifact = _fit_artifact(tmp_path)

    report = top_features_by_class(artifact, top_n=10)

    p0_features = [f["feature"] for f in report[Priority.P0_CRITICAL.value]]
    # CLASS_TEXTS for P0_critical is "critical outage pod crash cluster panic urgent failure".
    assert any(word in p0_features for word in ("critical", "crash", "panic", "outage"))


def test_top_features_by_class_importance_ratio_sums_to_one(tmp_path: Path) -> None:
    artifact = _fit_artifact(tmp_path)

    report = top_features_by_class(artifact, top_n=len(combined_feature_names(artifact)))

    for class_name in artifact.classes:
        ratios = [f["importance_ratio"] for f in report[class_name]]
        assert sum(ratios) == pytest.approx(1.0)


def test_top_features_by_class_ranks_by_standardized_not_raw_weight(tmp_path: Path) -> None:
    """Raw coef_ magnitude is not comparable across columns on different scales
    (idf-weighted TF-IDF terms vs. the raw {0,1} is_member_plus column) -- ranking
    and importance_ratio must be driven by weight * feature_std, not weight alone.
    """
    artifact = _fit_artifact(tmp_path)
    names = combined_feature_names(artifact)
    all_features = top_features_by_class(artifact, top_n=len(names))

    # word and char_wb vocabularies can coincidentally share a literal string
    # (e.g. a char 3-gram equal to a short word token) -- only check names that
    # unambiguously map back to a single column's std.
    unique_names = {name for name, count in Counter(names).items() if count == 1}
    std_by_name = dict(zip(names, artifact.feature_std, strict=True))

    for class_name in artifact.classes:
        rows = all_features[class_name]
        for row in rows:
            if row["feature"] not in unique_names:
                continue
            expected_std_weight = row["weight"] * std_by_name[row["feature"]]
            assert row["std_weight"] == pytest.approx(expected_std_weight)
        # Ranked by |std_weight| descending -- not necessarily |weight| descending.
        std_weights = [abs(row["std_weight"]) for row in rows]
        assert std_weights == sorted(std_weights, reverse=True)
