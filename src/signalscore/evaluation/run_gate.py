"""Gate CLI: loads a candidate (and current production, if any) from the
MLflow registry, runs the fixed promotion gate (`gate.build_promotion_gate()`),
and promotes/rejects the candidate accordingly.

Run via:
    uv run python -m signalscore.evaluation.run_gate --candidate-version 1

Pass/fail handling (docs/mlflow-integration-plan.md design decision 2):
- Pre-check failure (`unit_tests`/`contract`): stays "staging", retry-worthy --
  no registry.promote()/reject() call. Still gets an EXPERIMENTS.md row
  (printed, never auto-written -- see `train_baseline.py::main()`'s own
  convention, matched here).
- Margin failure: `registry.reject()`, tagged `gate_result=fail`.
- Pass, no current production (first-ever promotion): an absolute PR-AUC
  quality floor applies instead of a pure pass-through -- see
  `FIRST_PROMOTION_MIN_PR_AUC`.
- Pass, production exists (genuine head-to-head win): `registry.promote()`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow

from signalscore.evaluation import contract
from signalscore.evaluation.gate import GateContext, build_promotion_gate
from signalscore.evaluation.margin import score_artifact
from signalscore.registry import ModelRegistry
from signalscore.settings import Settings
from signalscore.training.train_baseline import (
    MODEL_ARTIFACT_FILENAME,
    BaselineArtifact,
    compute_metrics,
    format_experiments_row,
    load_feature_rows,
)

# Absolute quality floor for the first-ever promotion (no production exists
# yet to beat head-to-head). Derived from the one real baseline run already
# logged in EXPERIMENTS.md (PR-AUC 0.5653, dated 2026-08-29) -- not invented.
FIRST_PROMOTION_MIN_PR_AUC = 0.5653

_PRE_CHECK_STAGES = ("unit_tests", "contract")


def _load_artifact(registry: ModelRegistry, version: str) -> BaselineArtifact:
    info = registry.get_version(version)
    local_dir = registry.resolve_artifact_path(info)
    artifact: BaselineArtifact = joblib.load(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        local_dir / MODEL_ARTIFACT_FILENAME
    )
    return artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-version", required=True, help="Registered model version to gate."
    )
    parser.add_argument(
        "--dataset-snapshot",
        default="kubernetes-kubernetes",
        help="Dataset snapshot name for the EXPERIMENTS.md row.",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=contract.EVAL_SET_PATH,
        help=(
            "Frozen eval set to gate against. Defaults to the real "
            f"{contract.FROZEN_EVAL_TAG} set; override for a future eval-set "
            "version or (in tests) a throwaway DVC-tracked fixture."
        ),
    )
    parser.add_argument("--notes", default="", help="Extra notes for the EXPERIMENTS.md row.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    candidate_version: str = args.candidate_version

    mlflow.set_tracking_uri(Settings().mlflow_tracking_uri)  # pyright: ignore[reportUnknownMemberType]
    registry = ModelRegistry()

    candidate_info = registry.get_version(candidate_version)
    candidate = _load_artifact(registry, candidate_version)

    production_info = registry.get_production()
    production = (
        _load_artifact(registry, production_info.version) if production_info is not None else None
    )

    eval_rows = load_feature_rows(args.eval_set)

    y_true, y_pred, y_proba, classes = score_artifact(candidate, eval_rows)
    metrics = compute_metrics(y_true, y_pred, y_proba, classes)

    ctx = GateContext(candidate=candidate, production=production, eval_rows=eval_rows)
    result = build_promotion_gate(eval_set_path=args.eval_set).handle(ctx)

    if not result.passed and result.failed_stage in _PRE_CHECK_STAGES:
        # Pre-check failure: stays "staging", retry-worthy -- no promote()/
        # reject() call per the plan.
        gate_result = f"pre-check failed: {result.failed_stage}"
        notes = f"{args.notes} {result.detail}".strip()
    elif not result.passed:
        reason = result.detail or "margin check failed"
        registry.reject(candidate_version, reason=reason, eval_set_tag=contract.FROZEN_EVAL_TAG)
        gate_result = "fail"
        notes = f"{args.notes} {reason}".strip()
    elif production_info is None:
        pr_auc = metrics["pr_auc_macro"]
        if pr_auc > FIRST_PROMOTION_MIN_PR_AUC:
            registry.promote(
                candidate_version,
                gate_result="pass (first promotion)",
                eval_set_tag=contract.FROZEN_EVAL_TAG,
            )
            gate_result = "pass (first promotion)"
            notes = args.notes
        else:
            reason = (
                f"first-promotion floor not cleared: pr_auc_macro {pr_auc:.4f} "
                f"<= {FIRST_PROMOTION_MIN_PR_AUC}"
            )
            registry.reject(candidate_version, reason=reason, eval_set_tag=contract.FROZEN_EVAL_TAG)
            gate_result = "fail: first-promotion floor not cleared"
            notes = f"{args.notes} {reason}".strip()
    else:
        registry.promote(
            candidate_version, gate_result="pass", eval_set_tag=contract.FROZEN_EVAL_TAG
        )
        gate_result = "pass"
        notes = args.notes

    print(json.dumps(metrics, indent=2))

    row = format_experiments_row(
        date=datetime.now(UTC).date().isoformat(),
        model_config=f"candidate version {candidate_version} (run {candidate_info.run_id})",
        dataset_snapshot=args.dataset_snapshot,
        metrics=metrics,
        gate_result=gate_result,
        notes=notes,
    )
    print(row)


if __name__ == "__main__":
    main()
