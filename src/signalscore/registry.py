"""Thin MLflow client shared across pipeline stages.

Provides the single point of contact with the MLflow model registry: `training`
registers a newly trained candidate under the "staging" tag, `evaluation` flips
a model's tag between "staging", "production", and "rejected" once it has run
the promotion gate, and `serving` reads whichever model currently carries the
"production" tag to answer `/score` requests. See docs/signalscore-design.md,
the architecture section and the promotion-gate rules, for the full tag
lifecycle this module is expected to support.

The current production version is tracked with MLflow's `@production` model
alias (a structurally-enforced single pointer: only one version can ever hold
a given alias), not a plain tag -- a plain `stage` tag stays alongside it
purely as a human-readable label for the MLflow UI / audit trail. See
docs/mlflow-integration-plan.md for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow.artifacts
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "signalscore-priority"
PRODUCTION_ALIAS = "production"

STAGE_TAG = "stage"
PRIOR_PRODUCTION_TAG = "prior_production"
GATE_RESULT_TAG = "gate_result"
REJECTION_REASON_TAG = "rejection_reason"
EVAL_SET_TAG = "eval_set_tag"

# error_codes get_model_version_by_alias raises for "nothing to find yet" --
# RESOURCE_DOES_NOT_EXIST when the registered model itself doesn't exist,
# INVALID_PARAMETER_VALUE when the model exists but the alias was never set.
# Both mean the same thing to a caller: no production version yet.
_NO_PRODUCTION_ERROR_CODES = frozenset({"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"})


@dataclass(frozen=True)
class ModelVersionInfo:
    version: str
    run_id: str
    stage: str
    source: str


def _to_info(mv: ModelVersion) -> ModelVersionInfo:
    return ModelVersionInfo(
        version=str(mv.version),
        run_id=mv.run_id or "",
        stage=mv.tags.get(STAGE_TAG, ""),
        source=mv.source or "",
    )


class ModelRegistry:
    """Every mlflow call in the repo goes through here."""

    def __init__(
        self, client: MlflowClient | None = None, model_name: str = DEFAULT_MODEL_NAME
    ) -> None:
        self._client = client or MlflowClient()
        self._model_name = model_name

    def _set_tag(self, version: str, key: str, value: str) -> None:
        self._client.set_model_version_tag(self._model_name, version, key=key, value=value)

    def register_candidate(self, run_id: str, artifact_path: str = "model") -> ModelVersionInfo:
        try:
            self._client.create_registered_model(self._model_name)
        except MlflowException as e:
            if e.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        source = f"runs:/{run_id}/{artifact_path}"
        mv = self._client.create_model_version(name=self._model_name, source=source, run_id=run_id)
        self._set_tag(str(mv.version), STAGE_TAG, "staging")
        return _to_info(self._client.get_model_version(self._model_name, str(mv.version)))

    def get_production(self) -> ModelVersionInfo | None:
        try:
            mv = self._client.get_model_version_by_alias(self._model_name, PRODUCTION_ALIAS)
        except MlflowException as e:
            if e.error_code in _NO_PRODUCTION_ERROR_CODES:
                return None
            raise
        return _to_info(mv)

    def get_version(self, version: str) -> ModelVersionInfo:
        return _to_info(self._client.get_model_version(self._model_name, version))

    def resolve_artifact_path(self, info: ModelVersionInfo) -> Path:
        """Passes tracking_uri explicitly -- mlflow.artifacts.download_artifacts()
        is a module-level function unaware of self._client's configured store,
        so without this it silently falls back to whatever the process-global
        default tracking URI happens to be (usually wrong, and a "run not
        found" error rather than a loud misconfiguration error)."""
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=info.source,
            tracking_uri=str(self._client.tracking_uri),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        return Path(local_path)

    def _clear_prior_production_tag(self) -> None:
        for mv in self._client.search_model_versions(f"name='{self._model_name}'"):
            if mv.tags.get(PRIOR_PRODUCTION_TAG) == "true":
                self._client.delete_model_version_tag(
                    self._model_name, str(mv.version), key=PRIOR_PRODUCTION_TAG
                )

    def promote(
        self, version: str, gate_result: str = "pass", eval_set_tag: str | None = None
    ) -> None:
        self._clear_prior_production_tag()
        old = self.get_production()
        if old is not None and old.version != version:
            self._set_tag(old.version, STAGE_TAG, "archived")
            self._set_tag(old.version, PRIOR_PRODUCTION_TAG, "true")
        self._client.set_registered_model_alias(self._model_name, PRODUCTION_ALIAS, version)
        self._set_tag(version, STAGE_TAG, "production")
        self._set_tag(version, GATE_RESULT_TAG, gate_result)
        if eval_set_tag is not None:
            self._set_tag(version, EVAL_SET_TAG, eval_set_tag)

    def reject(self, version: str, reason: str, eval_set_tag: str | None = None) -> None:
        self._set_tag(version, STAGE_TAG, "rejected")
        self._set_tag(version, REJECTION_REASON_TAG, reason)
        self._set_tag(version, GATE_RESULT_TAG, "fail")
        if eval_set_tag is not None:
            self._set_tag(version, EVAL_SET_TAG, eval_set_tag)

    def rollback(self) -> str:
        for mv in self._client.search_model_versions(f"name='{self._model_name}'"):
            if mv.tags.get(PRIOR_PRODUCTION_TAG) == "true":
                version = str(mv.version)
                self.promote(version, gate_result="rollback")
                return version
        raise RuntimeError("no prior production version recorded -- nothing to roll back to")
