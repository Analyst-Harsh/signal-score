"""Tests for signalscore.registry."""

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from signalscore.registry import ModelRegistry

MODEL_NAME = "test-model"


def _make_registry(tmp_path: Path) -> ModelRegistry:
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    return ModelRegistry(client=client, model_name=MODEL_NAME)


def test_register_candidate_tags_new_version_as_staging(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)

    info = registry.register_candidate(run_id="run-1")

    assert info.stage == "staging"
    assert info.run_id == "run-1"


def test_get_production_returns_none_before_any_promotion(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    assert registry.get_production() is None

    registry.register_candidate(run_id="run-1")
    assert registry.get_production() is None


def test_promote_sets_production_alias_and_structured_tags(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    candidate = registry.register_candidate(run_id="run-1")

    registry.promote(candidate.version, gate_result="pass", eval_set_tag="eval-set-v1")

    production = registry.get_production()
    assert production is not None
    assert production.version == candidate.version
    assert production.stage == "production"


def test_promote_demotes_prior_production_to_archived(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    v1 = registry.register_candidate(run_id="run-1")
    v2 = registry.register_candidate(run_id="run-2")

    registry.promote(v1.version)
    registry.promote(v2.version)

    production = registry.get_production()
    assert production is not None
    assert production.version == v2.version
    assert registry.get_version(v1.version).stage == "archived"


def test_reject_tags_rejected_with_reason(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    candidate = registry.register_candidate(run_id="run-1")

    registry.reject(candidate.version, reason="failed margin check", eval_set_tag="eval-set-v1")

    rejected = registry.get_version(candidate.version)
    assert rejected.stage == "rejected"


def test_rollback_raises_when_nothing_to_roll_back_to(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    candidate = registry.register_candidate(run_id="run-1")
    registry.promote(candidate.version)  # first-ever promotion: no prior version to record

    with pytest.raises(RuntimeError, match="no prior production version"):
        registry.rollback()


def test_promote_three_times_keeps_prior_production_unique_and_rollback_restores_v2(
    tmp_path: Path,
) -> None:
    """Regression test for the bug caught in review: a naive prior_production
    tag with no uniqueness guarantee could end up set on two versions at once
    after a third promotion, making rollback ambiguous. Two promotions
    wouldn't have caught this -- this test promotes three.
    """
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    registry = ModelRegistry(client=client, model_name=MODEL_NAME)
    v1 = registry.register_candidate(run_id="run-1")
    v2 = registry.register_candidate(run_id="run-2")
    v3 = registry.register_candidate(run_id="run-3")

    registry.promote(v1.version)
    registry.promote(v2.version)
    registry.promote(v3.version)

    prior_production_versions = [
        mv.version
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
        if mv.tags.get("prior_production") == "true"
    ]
    assert prior_production_versions == [int(v2.version)]

    restored = registry.rollback()

    assert restored == v2.version
    production = registry.get_production()
    assert production is not None
    assert production.version == v2.version
    assert registry.get_version(v3.version).stage == "archived"


def test_resolve_artifact_path_downloads_real_logged_artifact(tmp_path: Path) -> None:
    """Fully client-scoped (no mlflow.set_tracking_uri()/mlflow.start_run()) --
    those go through global module state (mlflow.set_tracking_uri() writes the
    MLFLOW_TRACKING_URI env var as a side effect, confirmed live) and would
    leak across tests instead of staying tmp_path-isolated like every other
    test in this file.
    """
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    registry = ModelRegistry(client=client, model_name=MODEL_NAME)

    artifact_file = tmp_path / "model.joblib"
    artifact_file.write_text("fake artifact contents")
    experiment_id = client.create_experiment("test-experiment")
    run = client.create_run(experiment_id)
    run_id: str = run.info.run_id
    client.log_artifact(  # pyright: ignore[reportUnknownMemberType]
        run_id, str(artifact_file), artifact_path="model"
    )

    candidate = registry.register_candidate(run_id=run_id)
    local_dir = registry.resolve_artifact_path(candidate)

    assert (local_dir / "model.joblib").read_text() == "fake artifact contents"
