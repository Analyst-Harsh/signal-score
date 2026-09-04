"""Monitoring stage.

Uses Evidently AI to monitor input and prediction drift on live `/score`
traffic. Drift results feed the (separate-repo) Next.js dashboard's drift
charts and trigger the drift-triggered retraining GitHub Actions workflow, in
addition to the regular scheduled retrain. See docs/signalscore-design.md,
the architecture section, for how drift monitoring loops back into
retraining.

Modules:
    reference.py       Input-drift reference (train.jsonl) and
                        prediction-drift reference (production model scored
                        on eval_set_v1.jsonl).
    buffer.py           Live-traffic capture -- Evidently's "current"
                        dataset, appended to by `/score`.
    drift.py            `DriftChecker` -- the Evidently Report/Dataset
                         wrapper, same tier as `registry.ModelRegistry`.
    run_drift_check.py  CLI entry point tying the above together into one
                         runnable command producing a JSON report.
"""
