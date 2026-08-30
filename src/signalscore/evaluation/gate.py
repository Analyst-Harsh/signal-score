"""Promotion gate -- Chain of Responsibility, fixed non-configurable chain.

docs/design-patterns-guide.md's Hard Warning #4 rules out Template Method
here specifically because a step must hard-fail rather than be skippable.
The resolution (docs/mlflow-integration-plan.md design decision 2) uses CoR's
*mechanism* -- each step decides pass-and-delegate vs. hard-fail -- without
its usual *configurability*: the chain is wired exactly once, in
`build_promotion_gate()`, and is never a caller-supplied/reorderable list.

This module is pure gate-decision logic: given a candidate artifact, an
optional production artifact, and eval rows, decide pass/fail and why. No
`ModelRegistry` calls, no `mlflow` calls, no CLI -- that wiring is PR4's job
(`run_gate.py`).
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from signalscore.evaluation import contract, margin
from signalscore.features.schema import FeatureRow
from signalscore.training.train_baseline import BaselineArtifact

# Generous enough for a slow CI box, short enough to fail the gate fast rather
# than hang it -- this is the repo's first use of subprocess, so an unscoped/
# unbounded call is exactly the risk this timeout exists to cut off.
_UNIT_TEST_TIMEOUT_SECONDS = 120
_DEFAULT_UNIT_TEST_PATH = "tests/features/"
# Tail of subprocess output kept on failure -- enough to show the real
# assertion, not so much that a hung/verbose run floods the gate result.
_FAILURE_OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class GateContext:
    """Everything a GateStep needs.

    `production` is None only in the first-ever-promotion case. MarginStep
    auto-passes then (see its docstring) -- the absolute quality floor for
    that case belongs to the PR4 caller (`run_gate.py`), which knows it's
    making a promotion decision; this module only knows pass/fail per stage.
    """

    candidate: BaselineArtifact
    production: BaselineArtifact | None
    eval_rows: list[FeatureRow]


@dataclass(frozen=True)
class GateResult:
    """Outcome of running the gate (or one step of it).

    Referenced but not fully spelled out in the plan -- shape settled here to
    match its own usage (`GateResult(passed=False, failed_stage=self.name,
    detail=reason)` on failure; `passed=True, failed_stage=None, detail=None`
    on a clean pass through the whole chain).
    """

    passed: bool
    failed_stage: str | None
    detail: str | None


class GateStep(ABC):
    """One link in the fixed promotion-gate chain."""

    name: ClassVar[str]

    def __init__(self, next_step: GateStep | None = None) -> None:
        self._next = next_step

    @abstractmethod
    def _check(self, ctx: GateContext) -> str | None:
        """None = pass, else the failure reason."""

    def handle(self, ctx: GateContext) -> GateResult:
        reason = self._check(ctx)
        if reason is not None:
            return GateResult(passed=False, failed_stage=self.name, detail=reason)
        if self._next is None:
            return GateResult(passed=True, failed_stage=None, detail=None)
        return self._next.handle(ctx)


class UnitTestStep(GateStep):
    """Runs the feature-pipeline unit tests as a subprocess.

    Scoped to `tests/features/` specifically, not the whole suite, per the
    plan's explicit reasoning: this is the repo's first use of subprocess,
    and an unscoped call risks self-recursion (running the gate re-invokes
    pytest, which could re-invoke the gate) or a slow-suite hang.

    `test_path` is a constructor default, not chain configurability -- it
    exists purely so tests can point this step at a tiny throwaway pass/fail
    file under `tmp_path` instead of the real suite (see
    tests/evaluation/test_gate.py). `build_promotion_gate()` never overrides
    it, so production always runs the real `tests/features/` suite.
    """

    name: ClassVar[str] = "unit_tests"

    def __init__(
        self,
        next_step: GateStep | None = None,
        test_path: str = _DEFAULT_UNIT_TEST_PATH,
    ) -> None:
        super().__init__(next_step)
        self._test_path = test_path

    def _check(self, ctx: GateContext) -> str | None:
        del ctx  # unit tests don't depend on the candidate/production artifacts
        try:
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pytest", self._test_path],
                capture_output=True,
                text=True,
                timeout=_UNIT_TEST_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"unit tests timed out after {_UNIT_TEST_TIMEOUT_SECONDS}s"
        except OSError as e:
            return f"could not run unit tests: {e}"
        if result.returncode == 0:
            return None
        tail = (result.stdout + result.stderr)[-_FAILURE_OUTPUT_TAIL_CHARS:]
        return f"unit tests failed (exit code {result.returncode}): {tail}"


class ContractStep(GateStep):
    """Model I/O contract: schema, valid probabilities, no NaNs, latency
    bound, plus the DVC eval-set integrity check. See `contract.py`.
    """

    name: ClassVar[str] = "contract"

    def _check(self, ctx: GateContext) -> str | None:
        return contract.check_model_contract(ctx)


class MarginStep(GateStep):
    """Head-to-head margin: bootstrap-CI PR-AUC win + non-regression floors
    on minority F1 and Brier score. See `margin.py`.

    Auto-passes (returns None) when `ctx.production` is None -- the
    first-ever-promotion case is a separate absolute quality floor applied by
    the PR4 caller, not by this step (this module has no notion of "this is
    a promotion decision", only pass/fail per stage).
    """

    name: ClassVar[str] = "margin"

    def _check(self, ctx: GateContext) -> str | None:
        if ctx.production is None:
            return None
        return margin.evaluate_margin(ctx)


def build_promotion_gate() -> GateStep:
    """The one place the chain is wired.

    Fixed order, not caller-configurable -- unlike serving's /score
    ValidationChain (design-patterns-guide.md Chain of Responsibility
    example), which IS meant to be freely recomposed. No code path here
    constructs a shorter or reordered chain: a caller-supplied `list[GateStep]`
    would reintroduce the "step silently skipped" risk Hard Warning #4 flags,
    so this stays a plain function returning a hard-coded chain, never a
    public constructor parameter.
    """
    return UnitTestStep(next_step=ContractStep(next_step=MarginStep()))
