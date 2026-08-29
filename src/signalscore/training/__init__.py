"""Training stage.

`build_dataset.py` assembles the feature matrix (via `features.pipeline`) and
splits it: `split.py` freezes a time-based eval/test set (most recent 20% by
`created_at` — a random/stratified split was measured to inflate the
majority-class baseline, see docs/priority-signal-decision.md) and
hash-splits the remaining 80% into train/val. `leakage_audit.py` runs the
near-duplicate, author-concentration, and label-drift checks before that
split is DVC-frozen.

Model training itself (sklearn/XGBoost configs, W&B tracking, registering
"staging" via `registry.py`) is not yet implemented — see
docs/signalscore-design.md for that planned flow.
"""
