import pytest

from minicoder.domain.errors import AgentStateError, DomainValidationError
from minicoder.domain.state import (
    AgentPhase,
    AgentRunResult,
    AgentStateMachine,
    AgentStopReason,
)


def test_state_machine_accepts_the_model_tool_model_completion_path() -> None:
    state = AgentStateMachine(max_steps=2)

    state.begin_model_call()
    assert state.phase is AgentPhase.CALL_MODEL
    assert state.model_steps == 1

    state.begin_tool_execution()
    assert state.phase is AgentPhase.EXECUTE_TOOLS

    state.begin_model_call()
    state.complete()

    assert state.phase is AgentPhase.COMPLETE
    assert state.model_steps == 2


def test_state_machine_accepts_plan_before_execution() -> None:
    state = AgentStateMachine(max_steps=2, planning_required=True)

    state.begin_planning_call()
    assert state.phase is AgentPhase.PLANNING
    assert state.plan_completed is False

    state.plan_ready()

    assert state.phase is AgentPhase.PLAN_READY
    assert state.plan_completed is True
    assert state.can_call_model is True

    state.begin_model_call()
    state.complete()

    assert state.phase is AgentPhase.COMPLETE
    assert state.model_steps == 2


def test_state_machine_blocks_execution_and_tools_until_plan_is_ready() -> None:
    state = AgentStateMachine(max_steps=3, planning_required=True)

    with pytest.raises(AgentStateError, match="before a plan is ready"):
        state.begin_model_call()

    state.begin_planning_call()

    with pytest.raises(AgentStateError, match="before a plan is ready"):
        state.begin_tool_execution()

    state.plan_ready()
    state.begin_model_call()
    state.begin_tool_execution()

    assert state.phase is AgentPhase.EXECUTE_TOOLS


def test_state_machine_allows_bounded_planning_format_retries() -> None:
    state = AgentStateMachine(max_steps=3, planning_required=True)

    state.begin_planning_call()
    state.begin_planning_call()
    state.plan_ready()

    assert state.phase is AgentPhase.PLAN_READY
    assert state.model_steps == 2


def test_state_machine_rejects_planning_when_it_is_disabled() -> None:
    state = AgentStateMachine(max_steps=1)

    with pytest.raises(AgentStateError, match="cannot call planning model"):
        state.begin_planning_call()


def test_state_machine_rejects_a_model_call_after_the_step_limit() -> None:
    state = AgentStateMachine(max_steps=1)
    state.begin_model_call()
    state.begin_tool_execution()

    assert state.can_call_model is False
    with pytest.raises(AgentStateError, match="cannot call model"):
        state.begin_model_call()


def test_state_machine_can_fail_a_host_preflight_before_any_model_request() -> None:
    state = AgentStateMachine(max_steps=1)

    state.fail()

    assert state.phase is AgentPhase.FAILED
    assert state.model_steps == 0


def test_state_machine_rejects_transition_out_of_a_terminal_phase() -> None:
    state = AgentStateMachine(max_steps=1)
    state.begin_model_call()
    state.complete()

    with pytest.raises(AgentStateError, match="complete -> failed"):
        state.fail()


def test_state_machine_allows_policy_feedback_before_another_model_call() -> None:
    state = AgentStateMachine(max_steps=2)

    state.begin_model_call()
    state.require_revision()

    assert state.phase is AgentPhase.REVIEW_REQUIRED
    assert state.can_call_model is True

    state.begin_model_call()
    state.complete()

    assert state.phase is AgentPhase.COMPLETE


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_state_machine_requires_a_positive_integer_limit(max_steps: object) -> None:
    with pytest.raises(DomainValidationError, match="positive integer"):
        AgentStateMachine(max_steps=max_steps)  # type: ignore[arg-type]


def test_state_machine_requires_a_boolean_planning_flag() -> None:
    with pytest.raises(DomainValidationError, match="must be a boolean"):
        AgentStateMachine(
            max_steps=1,
            planning_required="yes",  # type: ignore[arg-type]
        )


def test_run_result_rejects_a_non_terminal_phase() -> None:
    with pytest.raises(DomainValidationError, match="terminal"):
        AgentRunResult(
            phase=AgentPhase.CALL_MODEL,
            stop_reason=AgentStopReason.MODEL_ERROR,
            model_steps=1,
            messages=(),
            failure_message="request failed",
        )


def test_failed_run_result_cannot_claim_a_final_response() -> None:
    with pytest.raises(DomainValidationError, match="failure message"):
        AgentRunResult(
            phase=AgentPhase.FAILED,
            stop_reason=AgentStopReason.FINAL_RESPONSE,
            model_steps=1,
            messages=(),
            final_response="done",
        )
