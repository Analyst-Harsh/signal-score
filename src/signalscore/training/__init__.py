"""Training stage.

Splits collected, featurized data into train/val/test sets, trains multiple
scikit-learn and XGBoost model configs, and tracks each run in Weights &
Biases. The best candidate is registered in the MLflow registry under the
"staging" tag via `registry.py`, quarantined until it passes the promotion
gate in `evaluation`. See docs/signalscore-design.md, the architecture and
tech stack sections, for the full training-to-registry flow.
"""
