"""Model I/O contract check -- schema, valid probabilities, no NaNs, latency
bound -- plus the DVC-based frozen eval-set integrity check.

The eval set is DVC-tracked and frozen at the git tag `eval-set-v1`
(`data/processed/kubernetes-kubernetes/eval_set_v1.jsonl`). `dvc diff
<frozen_tag> --targets <path> --json` reports an empty diff when the working
file exactly matches what was frozen at that tag, or names what changed
otherwise -- this checks DVC's actual content hash instead of a weaker proxy
like a row-count assertion (see docs/mlflow-integration-plan.md design
decision 2).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from signalscore.training.train_baseline import FittedVectorizers, build_feature_matrix

if TYPE_CHECKING:
    from signalscore.evaluation.gate import GateContext

FROZEN_EVAL_TAG = "eval-set-v1"
# The real frozen eval set this check verifies against in production. Kept as
# a default parameter (not hardcoded inline) so tests can point
# `check_model_contract`/`verify_eval_set_integrity` at a throwaway tmp_path
# git+dvc repo instead -- same "thread the path as a parameter" convention
# `train_baseline.py`/`registry.py` already use for exactly this reason.
EVAL_SET_PATH = Path("data/processed/kubernetes-kubernetes/eval_set_v1.jsonl")

_DVC_DIFF_TIMEOUT_SECONDS = 30
_MAX_PREDICT_LATENCY_SECONDS = 2.0
# Scoring the full eval set on every gate run is unnecessary for a contract
# check (schema/NaN/probability-validity/latency) -- a sample is enough and
# keeps the gate fast. Margin (margin.py) scores the full set.
_CONTRACT_SAMPLE_SIZE = 200


def _parse_dvc_diff_output(raw_json: str) -> str | None:
    """None = clean (matches the frozen tag exactly); otherwise a message
    naming what changed. Pure function -- trivial to unit-test with literal
    JSON fixtures, no subprocess involved.
    """
    diff = json.loads(raw_json)
    if not diff:
        return None
    changed = [k for k in ("added", "modified", "deleted", "renamed") if diff.get(k)]
    return f"eval set does not match frozen tag {FROZEN_EVAL_TAG}: {', '.join(changed)}"


def verify_eval_set_integrity(eval_set_path: Path, frozen_tag: str = FROZEN_EVAL_TAG) -> str | None:
    """Thin subprocess wrapper around `dvc diff`. Catches CalledProcessError/
    FileNotFoundError/TimeoutExpired and returns a failure string rather than
    raising -- a missing `dvc` binary or a missing tag must hard-fail the
    gate, never pass silently.

    Judgment call: `dvc diff` resolves `--targets` relative to the invoked
    process's cwd, and a bare absolute path there was verified (empirically,
    against a real tmp_path git+dvc repo) to make dvc misreport an unchanged
    file as "added". Running with `cwd=eval_set_path.parent` and a
    filename-only target sidesteps that and works correctly regardless of
    where the gate itself is invoked from.
    """
    try:
        result = subprocess.run(  # noqa: S603
            # `dvc` is a repo-level dev dependency resolved via PATH, not an
            # attacker-controlled partial path.
            ["dvc", "diff", frozen_tag, "--targets", eval_set_path.name, "--json"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=_DVC_DIFF_TIMEOUT_SECONDS,
            check=True,
            cwd=eval_set_path.parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"could not verify eval set against frozen tag {frozen_tag}: {e}"
    return _parse_dvc_diff_output(result.stdout)


def check_model_contract(ctx: GateContext, eval_set_path: Path = EVAL_SET_PATH) -> str | None:
    """Verifies the frozen eval set's DVC integrity, then scores
    `ctx.candidate` on a sample of `ctx.eval_rows` and checks: output shape
    matches (n_rows, n_classes), no NaNs, every row is a valid probability
    distribution (in [0, 1], sums to ~1), and prediction latency stays under
    `_MAX_PREDICT_LATENCY_SECONDS`.

    `eval_set_path` is a parameter (not the hardcoded `EVAL_SET_PATH`
    constant) specifically so tests can inject a throwaway tmp_path git+dvc
    repo instead of depending on the real multi-megabyte eval set file or a
    real git tag existing in whatever environment runs the tests.
    """
    integrity_failure = verify_eval_set_integrity(eval_set_path)
    if integrity_failure is not None:
        return integrity_failure

    rows = ctx.eval_rows[:_CONTRACT_SAMPLE_SIZE]
    if not rows:
        return "no eval rows available to check the model contract against"

    candidate = ctx.candidate
    vectorizers = FittedVectorizers(word=candidate.word_vectorizer, char=candidate.char_vectorizer)
    x = build_feature_matrix(rows, vectorizers)

    started = time.perf_counter()
    proba = candidate.model.predict_proba(x)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    elapsed = time.perf_counter() - started

    proba_arr: NDArray[np.float64] = np.asarray(  # pyright: ignore[reportUnknownVariableType]
        proba,  # pyright: ignore[reportUnknownArgumentType]
        dtype=np.float64,
    )
    actual_shape: tuple[int, int] = proba_arr.shape  # pyright: ignore[reportUnknownMemberType]
    expected_shape = (len(rows), len(candidate.classes))
    if actual_shape != expected_shape:
        return f"predict_proba shape {actual_shape} does not match expected {expected_shape}"
    if np.isnan(proba_arr).any():
        return "predict_proba produced NaN values"
    if (proba_arr < 0.0).any() or (proba_arr > 1.0).any():
        return "predict_proba produced values outside the valid [0, 1] probability range"
    row_sums = proba_arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        return "predict_proba rows do not sum to ~1 (not a valid probability distribution)"
    if elapsed > _MAX_PREDICT_LATENCY_SECONDS:
        return (
            f"predict_proba took {elapsed:.3f}s on {len(rows)} rows, "
            f"exceeding the {_MAX_PREDICT_LATENCY_SECONDS}s bound"
        )

    return None
