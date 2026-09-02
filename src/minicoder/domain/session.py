"""Immutable values used to restore the most recent workspace session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import Message, MessageRole


class ArchivedTurnStatus(str, Enum):
    """Terminal or interrupted state recorded for the latest archived turn."""

    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ArchivedDialogueTurn:
    """One external user turn reconstructed from an exact session archive."""

    session_id: str  # Process archive that owns this turn.
    turn_index: int  # One-based turn position inside that process.
    recorded_at: datetime  # Time when the external user request was recorded.
    task: str  # Exact user input before host-only instructions were added.
    status: ArchivedTurnStatus  # Completed, failed, or interrupted outcome.
    final_response: str | None = None  # Exact visible model answer, when available.
    failure_message: str | None = None  # Exact host failure, when available.

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise DomainValidationError("archived dialogue session_id must be non-blank")
        if (
            not isinstance(self.turn_index, int)
            or isinstance(self.turn_index, bool)
            or self.turn_index <= 0
        ):
            raise DomainValidationError(
                "archived dialogue turn_index must be positive"
            )
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() is None
        ):
            raise DomainValidationError(
                "archived dialogue recorded_at must be timezone-aware"
            )
        if not isinstance(self.task, str) or not self.task.strip():
            raise DomainValidationError("archived dialogue task must be non-blank")
        if not isinstance(self.status, ArchivedTurnStatus):
            raise DomainValidationError(
                "archived dialogue status must be an ArchivedTurnStatus"
            )
        for name, value in (
            ("final_response", self.final_response),
            ("failure_message", self.failure_message),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise DomainValidationError(
                    f"archived dialogue {name} must be non-blank text or None"
                )


@dataclass(frozen=True, slots=True)
class ContextCheckpoint:
    """One reusable summary covering an exact prefix of archived messages."""

    summary: str  # Bounded summary used instead of the covered raw prefix.
    covered_message_count: int  # Number of covered non-System history messages.
    source_hash: str  # SHA-256 digest of the exact covered message prefix.

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise DomainValidationError("context checkpoint summary must be non-blank")
        if (
            not isinstance(self.covered_message_count, int)
            or isinstance(self.covered_message_count, bool)
            or self.covered_message_count <= 0
        ):
            raise DomainValidationError(
                "context checkpoint covered_message_count must be positive"
            )
        if (
            not isinstance(self.source_hash, str)
            or len(self.source_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.source_hash)
        ):
            raise DomainValidationError(
                "context checkpoint source_hash must be a lowercase SHA-256 digest"
            )


@dataclass(frozen=True, slots=True)
class RecentSessionContext:
    """Complete recovery data loaded from one previous process."""

    session_id: str  # Identifier of the previous MiniCoder process.
    recorded_at: datetime  # Time of the latest usable archive record.
    last_task: str  # Exact most recent external user request.
    status: ArchivedTurnStatus  # Whether that turn completed, failed, or was cut off.
    stop_reason: str | None  # Stable AgentStopReason value when one was recorded.
    messages: tuple[Message, ...] = ()  # Exact User/Assistant/Tool protocol history.
    final_response: str | None = None  # Last visible answer when the turn completed.
    failure_message: str | None = None  # Last host failure when the turn failed.
    context_checkpoint: ContextCheckpoint | None = None  # Reusable working summary.

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
        if any(not isinstance(message, Message) for message in self.messages):
            raise DomainValidationError(
                "recent session messages must contain only Message values"
            )
        if any(message.role is MessageRole.SYSTEM for message in self.messages):
            raise DomainValidationError(
                "recent session messages must not contain system messages"
            )
        for name, value in (
            ("final_response", self.final_response),
            ("failure_message", self.failure_message),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise DomainValidationError(
                    f"recent session {name} must be non-blank text or None"
                )
        if self.context_checkpoint is not None and not isinstance(
            self.context_checkpoint,
            ContextCheckpoint,
        ):
            raise DomainValidationError(
                "recent session context_checkpoint must be ContextCheckpoint or None"
            )
