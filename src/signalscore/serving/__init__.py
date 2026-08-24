"""Serving stage.

A FastAPI app exposing a `/score` endpoint that loads the current
"production"-tagged model via `registry.py` and scores incoming GitHub issue
data. Future response schema (not implemented yet):
`{"priority_score": float, "priority_class": str, "model_version": str,
"confidence_note": str}`. See docs/signalscore-design.md, the architecture
section, for the full response payload example.
"""
