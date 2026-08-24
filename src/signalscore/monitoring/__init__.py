"""Monitoring stage.

Uses Evidently AI to monitor input and prediction drift on live `/score`
traffic. Drift results feed the (separate-repo) Next.js dashboard's drift
charts and trigger the drift-triggered retraining GitHub Actions workflow, in
addition to the regular scheduled retrain. See docs/signalscore-design.md,
the architecture section, for how drift monitoring loops back into
retraining.
"""
