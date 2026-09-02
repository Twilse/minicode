"""Explicit state and terminal result values for one agent task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from minicoder.domain.errors import AgentStateError, DomainValidationError
from minicoder.domain.models import Message


class AgentPhase(str, Enum):
    """Mutually exclusive phases in the synchronous agent loop."""

    READY = "ready"
    PLANNING = "planning"
    CALL_MODEL = "call_model"
    PLAN_READY = "plan_ready"
    EXECUTE_TOOLS = "execute_tools"
    REVIEW_REQUIRED = "review_required"
    COMPLETE = "complete"
    FAILED = "failed"


class AgentStopReason(str, Enum):
    """Stable reason why one agent task stopped."""

    FINAL_RESPONSE = "final_response"
    MAX_STEPS = "max_steps"
    MODEL_ERROR = "model_error"
    PLANNING_ERROR = "planning_error"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
    USER_INTERRUPTED = "user_interrupted"
    VERIFICATION_UNSUPPORTED = "verification_unsupported"


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Immutable terminal outcome and conversation snapshot for one task."""

    phase: AgentPhase  # COMPLETE or FAILED terminal phase.
    stop_reason: AgentStopReason  # Stable explanation for termination.
    model_steps: int  # Task-loop model requests attempted; housekeeping is excluded.
    messages: tuple[Message, ...]  # Complete User/Assistant/Tool history; no System.
    final_response: str | None = None  # Final assistant text on successful completion.
    failure_message: str | None = None  # Host-readable reason when the task failed.

    def __post_init__(self) -> None:
        if self.phase not in {AgentPhase.COMPLETE, AgentPhase.FAILED}:
            raise DomainValidationError("agent run result phase must be terminal")
        if self.model_steps < 0:
            raise DomainValidationError("agent model_steps must be non-negative")
        if self.phase is AgentPhase.COMPLETE:
            if (
                self.stop_reason is not AgentStopReason.FINAL_RESPONSE
                or self.final_response is None
                or not self.final_response.strip()
                or self.failure_message is not None
            ):
                raise DomainValidationError(
                    "completed agent runs require only a final response"
                )
        elif (
            self.stop_reason is AgentStopReason.FINAL_RESPONSE
            or self.final_response is not None
            or self.failure_message is None
            or not self.failure_message.strip()
        ):
            raise DomainValidationError(
                "failed agent runs require only a failure message"
            )


class AgentStateMachine:
    """Validate phase transitions and count model-request steps."""

    _ALLOWED_TRANSITIONS = {
        AgentPhase.READY: frozenset(
            {
                AgentPhase.PLANNING,
                AgentPhase.CALL_MODEL,
                AgentPhase.FAILED,
            }
        ),
        AgentPhase.PLANNING: frozenset(
            {AgentPhase.PLAN_READY, AgentPhase.FAILED}
        ),
        AgentPhase.CALL_MODEL: frozenset(
            {
                AgentPhase.EXECUTE_TOOLS,
                AgentPhase.REVIEW_REQUIRED,
                AgentPhase.COMPLETE,
                AgentPhase.FAILED,
            }
        ),
        AgentPhase.PLAN_READY: frozenset(
            {AgentPhase.CALL_MODEL, AgentPhase.FAILED}
        ),
        AgentPhase.EXECUTE_TOOLS: frozenset(
            {AgentPhase.CALL_MODEL, AgentPhase.FAILED}
        ),
        AgentPhase.REVIEW_REQUIRED: frozenset(
            {AgentPhase.CALL_MODEL, AgentPhase.FAILED}
        ),
        AgentPhase.COMPLETE: frozenset(),
        AgentPhase.FAILED: frozenset(),
    }

    def __init__(
        self,
        *,
        max_steps: int,
        planning_required: bool = False,
    ) -> None:
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps <= 0
        ):
            raise DomainValidationError("agent max_steps must be a positive integer")
        if not isinstance(planning_required, bool):
            raise DomainValidationError(
                "agent planning_required must be a boolean"
            )
        self._max_steps = max_steps
        self._model_steps = 0
        self._phase = AgentPhase.READY
        self._planning_required = planning_required
        self._plan_completed = not planning_required

    @property
    def phase(self) -> AgentPhase:
        return self._phase

    @property
    def model_steps(self) -> int:
        return self._model_steps

    @property
    def plan_completed(self) -> bool:
        return self._plan_completed

    @property
    def can_call_model(self) -> bool:
        return (
            self._phase
            in {
                AgentPhase.READY,
                AgentPhase.PLANNING,
                AgentPhase.PLAN_READY,
                AgentPhase.EXECUTE_TOOLS,
                AgentPhase.REVIEW_REQUIRED,
            }
            and self._model_steps < self._max_steps
        )

    def begin_planning_call(self) -> None:
        if (
            not self._planning_required
            or self._plan_completed
            or self._phase not in {AgentPhase.READY, AgentPhase.PLANNING}
            or self._model_steps >= self._max_steps
        ):
            raise AgentStateError(
                f"cannot call planning model from {self._phase.value} at step "
                f"{self._model_steps}/{self._max_steps}"
            )
        if self._phase is AgentPhase.READY:
            self._transition(AgentPhase.PLANNING)
        self._model_steps += 1

    def begin_model_call(self) -> None:
        if self._planning_required and not self._plan_completed:
            raise AgentStateError(
                "cannot call execution model before a plan is ready"
            )
        if not self.can_call_model:
            raise AgentStateError(
                f"cannot call model from {self._phase.value} at step "
                f"{self._model_steps}/{self._max_steps}"
            )
        self._transition(AgentPhase.CALL_MODEL)
        self._model_steps += 1

    def begin_tool_execution(self) -> None:
        if self._planning_required and not self._plan_completed:
            raise AgentStateError(
                "cannot execute tools before a plan is ready"
            )
        self._transition(AgentPhase.EXECUTE_TOOLS)

    def plan_ready(self) -> None:
        if not self._planning_required:
            raise AgentStateError("cannot accept a plan when planning is disabled")
        self._transition(AgentPhase.PLAN_READY)
        self._plan_completed = True

    def complete(self) -> None:
        self._transition(AgentPhase.COMPLETE)

    def require_revision(self) -> None:
        self._transition(AgentPhase.REVIEW_REQUIRED)

    def fail(self) -> None:
        self._transition(AgentPhase.FAILED)

    def _transition(self, target: AgentPhase) -> None:
        if target not in self._ALLOWED_TRANSITIONS[self._phase]:
            raise AgentStateError(
                f"invalid agent transition: {self._phase.value} -> {target.value}"
            )
        self._phase = target
