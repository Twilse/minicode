"""Immutable values used to restore the most recent workspace session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import Message


class ArchivedTurnStatus(str, Enum):
    """Terminal or interrupted state recorded for the latest archived turn."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecentSessionContext:
    """Bounded recovery data loaded from one previous process."""

    session_id: str  # Identifier of the previous MiniCoder process.
    recorded_at: datetime  # Time of the latest usable archive record.
    context_summary: str  # Model-maintained rolling summary or safe fallback.
    last_task: str  # Exact most recent external user request.
    status: ArchivedTurnStatus  # Whether that turn completed, failed, or was cut off.
    stop_reason: str | None  # Stable AgentStopReason value when one was recorded.
    recent_messages: tuple[Message, ...] = ()  # Tail of the exact archived history.

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise DomainValidationError("recent session id must be non-blank text")
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() is None
        ):
            raise DomainValidationError(
                "recent session recorded_at must be timezone-aware"
            )
        if (
            not isinstance(self.context_summary, str)
            or not self.context_summary.strip()
        ):
            raise DomainValidationError(
                "recent session context_summary must be non-blank text"
            )
        if not isinstance(self.last_task, str) or not self.last_task.strip():
            raise DomainValidationError("recent session last_task must be non-blank text")
        if not isinstance(self.status, ArchivedTurnStatus):
            raise DomainValidationError(
                "recent session status must be an ArchivedTurnStatus"
            )
        if self.stop_reason is not None and (
            not isinstance(self.stop_reason, str) or not self.stop_reason.strip()
        ):
            raise DomainValidationError(
                "recent session stop_reason must be non-blank text or None"
            )
        if any(not isinstance(message, Message) for message in self.recent_messages):
            raise DomainValidationError(
                "recent session messages must contain only Message values"
            )
