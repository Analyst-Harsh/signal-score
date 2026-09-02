"""FastAPI serving app -- the `/score` MVP.

Startup-time load, not per-request (docs/mlflow-integration-plan.md section
5): `ModelRegistry().get_production()` is read exactly once, at process
boot, via the `lifespan` context manager below. If no production-tagged
model exists yet, startup raises `RuntimeError` and the process never comes
up -- a fresh clone/CI/dev env with an empty registry is a hard failure, not
a degraded-mode fallback. Because the model is loaded once at boot, `/score`
never touches MLflow again -- no `/healthz`, no "MLflow died mid-request"
handling, both explicitly out of scope per the plan.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import joblib
import mlflow
import structlog
from fastapi import FastAPI
from pydantic import BaseModel

from signalscore.evaluation.margin import score_artifact

# Must come after the margin import above: margin.py's own module-level
# imports already force xgboost to load before sentence_transformers/torch
# (see margin.py's comment) -- this line only re-finds the already-imported
# module, but keeping it below margin's import (rather than letting isort
# hoist it, which it wouldn't here since "evaluation" < "features"
# alphabetically anyway) keeps that ordering guarantee visible and intact.
from signalscore.features.embeddings import load_bge_model
from signalscore.features.labels import Priority
from signalscore.features.pipeline import build_features
from signalscore.features.schema import FeatureRow
from signalscore.registry import ModelRegistry
from signalscore.settings import Settings
from signalscore.training.train_baseline import MODEL_ARTIFACT_FILENAME, BaselineArtifact

logger = structlog.get_logger(__name__)

# FeatureRow's `label` is the training target, not a real scoring input --
# this placeholder fills it so a single-row FeatureRow can still be built to
# reuse score_artifact() unchanged, rather than reimplementing its
# transform-and-predict logic here.
_PLACEHOLDER_LABEL = Priority.P2_BACKLOG


class ScoreRequest(BaseModel):
    """Raw GitHub issue fields, not the assembled FeatureRow shape -- title/
    body/author_association go through the same `build_features()` frozen
    serving contract training used, so train/serve skew (e.g. a caller
    hand-concatenating title+body differently than `extract_text_features`
    does) is structurally impossible rather than left to the caller.
    """

    title: str
    body: str
    author_association: str | None = None


class ScoreResponse(BaseModel):
    priority_class: str
    priority_probabilities: dict[str, float]
    model_version: str


def _to_feature_row(payload: ScoreRequest) -> FeatureRow:
    bundle = build_features(payload.title, payload.body, payload.author_association)
    return FeatureRow(
        issue_number=0,
        created_at=datetime.now(UTC),
        label=_PLACEHOLDER_LABEL,
        text=bundle["text"],
        is_member_plus=bundle["is_member_plus"],
        **bundle["diagnostics"],
    )


def _load_production_artifact() -> tuple[BaselineArtifact, str]:
    """Loads the current production artifact the same way
    run_gate.py._load_artifact does: resolve the registry version, download
    its artifact dir, joblib.load() the model file.
    """
    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
    registry = ModelRegistry()
    info = registry.get_production()
    if info is None:
        raise RuntimeError(
            "no production-tagged model exists yet -- cannot start serving. "
            "Run the training + gate pipeline to promote a candidate first."
        )
    local_dir = registry.resolve_artifact_path(info)
    artifact: BaselineArtifact = joblib.load(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        local_dir / MODEL_ARTIFACT_FILENAME
    )
    return artifact, info.version


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    artifact, model_version = _load_production_artifact()
    app.state.artifact = artifact
    app.state.model_version = model_version
    if artifact.feature_source == "bge":
        app.state.bge_model = load_bge_model(artifact.extra["revision_sha"])
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/score")
def score(payload: ScoreRequest) -> ScoreResponse:
    started = time.perf_counter()
    row = _to_feature_row(payload)
    artifact: BaselineArtifact = app.state.artifact
    _, y_pred, y_proba, classes = score_artifact(
        artifact, [row], bge_model=getattr(app.state, "bge_model", None)
    )
    priority_class = str(y_pred[0])
    probabilities = {cls: float(p) for cls, p in zip(classes, y_proba[0], strict=True)}
    latency_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "score_request",
        model_version=app.state.model_version,
        latency_ms=latency_ms,
        priority_class=priority_class,
    )
    return ScoreResponse(
        priority_class=priority_class,
        priority_probabilities=probabilities,
        model_version=app.state.model_version,
    )
