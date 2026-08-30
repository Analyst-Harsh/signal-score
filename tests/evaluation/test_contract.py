"""Tests for signalscore.evaluation.contract."""

import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from signalscore.evaluation.contract import (
    FROZEN_EVAL_TAG,
    _parse_dvc_diff_output,  # pyright: ignore[reportPrivateUsage]
    check_model_contract,
    verify_eval_set_integrity,
)
from signalscore.evaluation.gate import GateContext
from signalscore.training.train_baseline import SparseMatrix
from tests.evaluation._helpers import build_artifact, init_frozen_git_dvc_repo, make_rows

# --- _parse_dvc_diff_output: literal JSON fixtures, no subprocess ---------


def test_parse_dvc_diff_output_returns_none_for_empty_diff() -> None:
    assert _parse_dvc_diff_output("{}") is None


def test_parse_dvc_diff_output_names_modified_key() -> None:
    raw = json.dumps(
        {"added": [], "deleted": [], "modified": [{"path": "x"}], "renamed": [], "not in cache": []}
    )
    result = _parse_dvc_diff_output(raw)
    assert result is not None
    assert FROZEN_EVAL_TAG in result
    assert "modified" in result


def test_parse_dvc_diff_output_names_multiple_changed_keys() -> None:
    raw = json.dumps(
        {
            "added": [{"path": "y"}],
            "deleted": [{"path": "z"}],
            "modified": [],
            "renamed": [],
            "not in cache": [],
        }
    )
    result = _parse_dvc_diff_output(raw)
    assert result is not None
    assert "added" in result
    assert "deleted" in result


# --- verify_eval_set_integrity: real subprocess against a tmp_path repo ---


def test_verify_eval_set_integrity_returns_none_when_clean(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)

    assert verify_eval_set_integrity(eval_file) is None


def test_verify_eval_set_integrity_reports_modification(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    eval_file.write_text("hello v2 -- modified after freezing")

    result = verify_eval_set_integrity(eval_file)

    assert result is not None
    assert FROZEN_EVAL_TAG in result
    assert "modified" in result


def test_verify_eval_set_integrity_reports_missing_tag(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)

    result = verify_eval_set_integrity(eval_file, frozen_tag="no-such-tag")

    assert result is not None
    assert "no-such-tag" in result


def test_verify_eval_set_integrity_reports_missing_dvc_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # a dir with no `dvc` executable

    result = verify_eval_set_integrity(eval_file)

    assert result is not None
    assert FROZEN_EVAL_TAG in result


# --- check_model_contract ---------------------------------------------------


class _SlowModel:
    """Real time.sleep()-based latency injection, not mocking, per repo
    convention -- duck-types LogisticRegression's predict_proba surface.
    """

    def predict_proba(self, x: SparseMatrix) -> NDArray[np.float64]:
        n_rows: int = x.shape[0]
        # Just above _MAX_PREDICT_LATENCY_SECONDS (2.0s) with enough margin to
        # avoid flakiness on a loaded CI runner, without wasting extra time.
        time.sleep(2.2)
        return np.tile(np.array([0.34, 0.33, 0.33]), (n_rows, 1))


class _InvalidProbaModel:
    """Returns out-of-range, non-normalized probabilities -- real object, no
    mocking -- to exercise check_model_contract's validity checks.
    """

    def predict_proba(self, x: SparseMatrix) -> NDArray[np.float64]:
        n_rows: int = x.shape[0]
        return np.tile(np.array([1.5, -0.5, 0.5]), (n_rows, 1))


def test_check_model_contract_passes_for_a_healthy_artifact(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    ctx = GateContext(candidate=artifact, production=None, eval_rows=rows)

    assert check_model_contract(ctx, eval_set_path=eval_file) is None


def test_check_model_contract_fails_when_eval_set_does_not_match_frozen_tag(
    tmp_path: Path,
) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    eval_file.write_text("modified after freezing")
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    ctx = GateContext(candidate=artifact, production=None, eval_rows=rows)

    result = check_model_contract(ctx, eval_set_path=eval_file)

    assert result is not None
    assert FROZEN_EVAL_TAG in result


def test_check_model_contract_fails_on_latency_bound(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    artifact.model = cast(LogisticRegression, _SlowModel())
    ctx = GateContext(candidate=artifact, production=None, eval_rows=rows)

    result = check_model_contract(ctx, eval_set_path=eval_file)

    assert result is not None
    assert "latency" in result.lower() or "exceeding" in result.lower()


def test_check_model_contract_fails_on_invalid_probabilities(tmp_path: Path) -> None:
    eval_file = init_frozen_git_dvc_repo(tmp_path)
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    artifact.model = cast(LogisticRegression, _InvalidProbaModel())
    ctx = GateContext(candidate=artifact, production=None, eval_rows=rows)

    result = check_model_contract(ctx, eval_set_path=eval_file)

    assert result is not None
    assert "probability" in result.lower() or "range" in result.lower()
