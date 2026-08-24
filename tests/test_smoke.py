"""Smoke test: the package must be importable."""

import importlib


def test_import_signalscore() -> None:
    importlib.import_module("signalscore")
