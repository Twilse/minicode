"""Immutable audit events emitted by the agent application core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from math import isfinite
from types import MappingProxyType

from minicoder.domain.errors import DomainValidationError

EventDetail = str | int | float | bool | None


class AgentEventKind(str, Enum):
    """Auditable event kinds produced by the currently implemented agent loop."""

    TASK_STARTED = "task_started"
    MODEL_REQUESTED = "model_requested"
    MODEL_RETRY_SCHEDULED = "model_retry_scheduled"
    CONTEXT_COMPACTED = "context_compacted"
    COMPLETION_REJECTED = "completion_rejected"
    TOOL_CALLED = "tool_called"
    TOOL_FINISHED = "tool_finished"
    VERIFICATION_PASSED = "verification_passed"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """One ordered, provider-neutral fact from an agent session."""

    run_id: str  # Opaque identifier shared by every event in one session.
    sequence: int  # One-based event order within the session.
    kind: AgentEventKind  # Stable event name interpreted by output adapters.
    occurred_at: datetime  # Timezone-aware creation time supplied by EventBus.
    model_step: int  # Model request number associated with the event; zero at start.
    details: Mapping[str, EventDetail] = field(default_factory=dict)  # Safe facts only.

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise DomainValidationError("agent event run_id must be non-blank text")
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence <= 0
        ):
            raise DomainValidationError(
                "agent event sequence must be a positive integer"
            )
        if not isinstance(self.kind, AgentEventKind):
            raise DomainValidationError("agent event kind must be an AgentEventKind")
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.tzinfo is None
            or self.occurred_at.utcoffset() is None
        ):
            raise DomainValidationError("agent event time must be timezone-aware")
        if (
            not isinstance(self.model_step, int)
            or isinstance(self.model_step, bool)
            or self.model_step < 0
        ):
            raise DomainValidationError(
                "agent event model_step must be a non-negative integer"
            )

        copied_details = dict(self.details)
        for name, value in copied_details.items():
            if not isinstance(name, str) or not name.strip():
                raise DomainValidationError(
                    "agent event detail names must be non-blank text"
                )
            if value is not None and not isinstance(
                value,
                (str, int, float, bool),
            ):
                raise DomainValidationError(
                    "agent event details must contain only scalar JSON values"
                )
            if isinstance(value, float) and not isfinite(value):
                raise DomainValidationError(
                    "agent event floating-point details must be finite"
                )
        object.__setattr__(
            self,
            "details",
            MappingProxyType(copied_details),
        )
