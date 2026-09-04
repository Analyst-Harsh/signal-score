"""Reference (baseline) datasets for Evidently drift comparison.

Input-drift reference = the production model's actual training distribution
(`train.jsonl`) -- what the model learned from, so a shift in live traffic
away from that distribution is exactly what "input drift" means. Prediction-
drift reference = the CURRENT production model's own predictions on
`eval_set_v1.jsonl` (frozen, held out, never trained on -- see
`evaluation/contract.py`'s `FROZEN_EVAL_TAG`), scored fresh on every call
rather than reused from training time, so the baseline always reflects
what's actually serving right now. See docs/signalscore-design.md,
architecture section, for why monitoring sits downstream of `/score`.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from signalscore.evaluation.margin import score_artifact
from signalscore.registry import ModelRegistry
from signalscore.training.train_baseline import (
    MODEL_ARTIFACT_FILENAME,
    BaselineArtifact,
    load_feature_rows,
)

# Anchored to the repo root, matching evaluation/contract.py's own
# _REPO_ROOT convention -- a bare relative path would resolve against
# whatever the invoking process's cwd happens to be.
_REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SET_PATH = _REPO_ROOT / "data/processed/kubernetes-kubernetes/train.jsonl"
EVAL_SET_PATH = _REPO_ROOT / "data/processed/kubernetes-kubernetes/eval_set_v1.jsonl"

# The 10 diagnostic columns docs/v0-feature-selection.md names as "the
# drift-monitoring vector for Evidently", plus is_member_plus and text -- the
# full input-drift column set. issue_number/created_at/label are excluded:
# live /score requests never populate them with real signal (see
# serving/app.py::_to_feature_row's placeholder label/issue_number/created_at).
INPUT_DRIFT_COLUMNS = [
    "text",
    "is_member_plus",
    "title_chars",
    "body_chars",
    "body_words",
    "has_code_block",
    "has_stack_trace",
    "has_url",
    "has_logline",
    "has_excl",
    "code_ratio",
    "any_lex",
    "is_ci_flake_shaped",
]


def load_input_reference(train_path: Path = TRAIN_SET_PATH) -> pd.DataFrame:
    """The `build_features()` output distribution for the rows the current
    production model was actually fit on -- the input-drift baseline.
    """
    rows = load_feature_rows(train_path)
    frame = pd.DataFrame([row.model_dump() for row in rows])
    return frame.loc[:, INPUT_DRIFT_COLUMNS]


def _load_production_artifact(registry: ModelRegistry) -> BaselineArtifact:
    """Resolves the registry's production version, downloads its artifact
    dir, joblib.load()s the model file -- mirrors
    serving/app.py::_load_production_artifact and
    evaluation/run_gate.py::_load_artifact exactly. Kept as its own small
    copy rather than a shared import, matching this repo's existing pattern
    (those two already cross-reference each other the same way in their own
    docstrings rather than factoring out a shared helper).
    """
    info = registry.get_production()
    if info is None:
        raise RuntimeError(
            "no production-tagged model exists yet -- cannot build a "
            "prediction reference without one"
        )
    local_dir = registry.resolve_artifact_path(info)
    artifact: BaselineArtifact = joblib.load(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        local_dir / MODEL_ARTIFACT_FILENAME
    )
    return artifact


def load_prediction_reference(
    registry: ModelRegistry, eval_set_path: Path = EVAL_SET_PATH
) -> pd.DataFrame:
    """Scores `eval_set_path` through the CURRENT production model, once per
    call -- the prediction-drift baseline. `registry` is a required parameter
    (never constructed internally) so tests can inject a tmp_path-scoped
    MlflowClient the same way tests/serving/test_app.py already does.
    """
    artifact = _load_production_artifact(registry)
    rows = load_feature_rows(eval_set_path)
    _, y_pred, y_proba, classes = score_artifact(artifact, rows)
    return pd.DataFrame(
        {
            "predicted_class": y_pred,
            **{f"proba_{cls}": y_proba[:, i] for i, cls in enumerate(classes)},
        }
    )
