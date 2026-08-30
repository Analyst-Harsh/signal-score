"""Evaluation stage — the promotion gate.

Gates whether a "staging"-tagged candidate model becomes the new
"production" model. Runs, in order: feature-pipeline unit tests, a model I/O
contract test (schema, valid probabilities, no NaNs, latency bound), and
then — only if both pass — a head-to-head evaluation against the current
production model on a frozen held-out set, comparing PR-AUC / minority-class
F1 and calibration (Brier score / ECE). A candidate must beat production by a
real margin (a minimum delta or non-overlapping confidence intervals), not
just any point-estimate win, to be promoted. Losses are logged, not
discarded. On a pass, flips the candidate's MLflow tag to "production" (and
the prior production model to a non-production tag) via `registry.py`; on a
fail, the candidate stays "staging" or is marked "rejected". See
docs/signalscore-design.md, the promotion gate section, for the full rule
set.

Modules:
    gate.py      Chain-of-Responsibility GateStep/GateContext/GateResult +
                 build_promotion_gate() -- the fixed unit_tests -> contract ->
                 margin chain.
    contract.py  Step 2 check logic: schema/probability/NaN/latency, plus the
                 DVC eval-set integrity check against the frozen `eval-set-v1`
                 tag.
    margin.py    Step 3 check logic: bootstrap-CI PR-AUC win + minority-F1/
                 Brier non-regression floors, head-to-head on the frozen eval
                 set.

This package is pure gate-decision logic, decoupled from `registry.py` --
loading candidate/production models and flipping MLflow tags is `run_gate.py`
(not yet built).
"""
