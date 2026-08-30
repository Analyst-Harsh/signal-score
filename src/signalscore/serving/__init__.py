"""Serving stage.

A FastAPI app exposing a `/score` endpoint that loads the current
"production"-tagged model via `registry.py` and scores incoming GitHub issue
data. See docs/signalscore-design.md, the architecture section, for the full
response payload example.

Modules:
    app.py  FastAPI app: startup-time production-model load (fails fast if
            none exists yet) + `POST /score`, per
            docs/mlflow-integration-plan.md section 5.
"""
