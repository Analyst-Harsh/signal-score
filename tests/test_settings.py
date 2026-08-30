"""Tests for signalscore.settings."""

import pytest

from signalscore.settings import Settings


def test_defaults_to_local_sqlite_tracking_uri() -> None:
    assert Settings().mlflow_tracking_uri == "sqlite:///mlflow.db"


def test_reads_mlflow_tracking_uri_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/other.db")
    assert Settings().mlflow_tracking_uri == "sqlite:////tmp/other.db"
