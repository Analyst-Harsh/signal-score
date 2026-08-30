"""Tests for signalscore.evaluation.gate."""

from pathlib import Path

from signalscore.evaluation.gate import (
    ContractStep,
    GateContext,
    MarginStep,
    UnitTestStep,
    build_promotion_gate,
)
from tests.evaluation._helpers import build_artifact, make_rows

# --- UnitTestStep -----------------------------------------------------------


def test_unit_test_step_passes_when_pointed_at_a_passing_file(tmp_path: Path) -> None:
    (tmp_path / "test_ok.py").write_text("def test_trivially_true():\n    assert True\n")
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    step = UnitTestStep(test_path=str(tmp_path))

    result = step.handle(GateContext(candidate=artifact, production=None, eval_rows=rows))

    assert result.passed is True
    assert result.failed_stage is None


def test_unit_test_step_fails_when_pointed_at_a_failing_file(tmp_path: Path) -> None:
    (tmp_path / "test_broken.py").write_text("def test_always_fails():\n    assert False\n")
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    step = UnitTestStep(test_path=str(tmp_path))

    result = step.handle(GateContext(candidate=artifact, production=None, eval_rows=rows))

    assert result.passed is False
    assert result.failed_stage == "unit_tests"
    assert result.detail is not None
    assert "failed" in result.detail


# --- ContractStep -------------------------------------------------------


def test_contract_step_passes_for_a_healthy_candidate() -> None:
    """Exercises the real default eval-set path (ContractStep offers no
    override -- by design, it always checks the real frozen eval set). This
    repo's data/processed/kubernetes-kubernetes/eval_set_v1.jsonl is expected
    to match the eval-set-v1 tag it was frozen at, so this is a real, fast
    (sub-second) integration check, not a mock.
    """
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    step = ContractStep()

    result = step.handle(GateContext(candidate=artifact, production=None, eval_rows=rows))

    assert result.passed is True


# --- MarginStep --------------------------------------------------------


def test_margin_step_auto_passes_when_no_production() -> None:
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)
    step = MarginStep()

    result = step.handle(GateContext(candidate=artifact, production=None, eval_rows=rows))

    assert result.passed is True


# --- build_promotion_gate: fixed order, non-configurable -----------------


def test_build_promotion_gate_wires_the_fixed_order() -> None:
    gate = build_promotion_gate()

    assert isinstance(gate, UnitTestStep)
    next_step = gate._next  # pyright: ignore[reportPrivateUsage]
    assert isinstance(next_step, ContractStep)
    next_next_step = next_step._next  # pyright: ignore[reportPrivateUsage]
    assert isinstance(next_next_step, MarginStep)
    assert next_next_step._next is None  # pyright: ignore[reportPrivateUsage]


def test_failing_unit_test_step_short_circuits_before_contract_and_margin_run(
    tmp_path: Path,
) -> None:
    """Same fixed shape build_promotion_gate() wires (UnitTestStep ->
    ContractStep -> MarginStep), pointed at a real throwaway broken pytest
    file under tmp_path instead of the real tests/features/ dir -- fast and
    decoupled from the real suite's pass/fail state, per the plan's own
    testing guidance. Records whether the downstream steps ever ran via real
    subclassing, not unittest.mock.
    """
    (tmp_path / "test_broken.py").write_text("def test_always_fails():\n    assert False\n")
    calls: list[str] = []

    class _RecordingContractStep(ContractStep):
        def _check(self, ctx: GateContext) -> str | None:
            del ctx
            calls.append("contract")
            return None

    class _RecordingMarginStep(MarginStep):
        def _check(self, ctx: GateContext) -> str | None:
            del ctx
            calls.append("margin")
            return None

    chain = UnitTestStep(
        next_step=_RecordingContractStep(next_step=_RecordingMarginStep()),
        test_path=str(tmp_path),
    )
    rows = make_rows(n_per_class=6)
    artifact = build_artifact(rows)

    result = chain.handle(GateContext(candidate=artifact, production=None, eval_rows=rows))

    assert result.passed is False
    assert result.failed_stage == "unit_tests"
    assert calls == []
