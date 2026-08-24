"""Thin MLflow client shared across pipeline stages.

Provides the single point of contact with the MLflow model registry: `training`
registers a newly trained candidate under the "staging" tag, `evaluation` flips
a model's tag between "staging", "production", and "rejected" once it has run
the promotion gate, and `serving` reads whichever model currently carries the
"production" tag to answer `/score` requests. See docs/signalscore-design.md,
the architecture section and the promotion-gate rules, for the full tag
lifecycle this module is expected to support.
"""
