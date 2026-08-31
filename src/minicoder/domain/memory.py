"""Immutable values used by session maintenance and project memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from minicoder.domain.errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class ProjectMemoryRecord:
    """One bounded model-selected durable project fact."""

    recorded_at: datetime  # Timezone-aware time when this memory was persisted.
    summary: str  # Historical project facts presented as data on a future run.

    def __post_init__(self) -> None:
        if (
            not isinstance(self.recorded_at, datetime)
            or self.recorded_at.tzinfo is None
            or self.recorded_at.utcoffset() is None
        ):
            raise DomainValidationError(
                "project memory recorded_at must be timezone-aware"
            )
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise DomainValidationError(
                "project memory summary must be non-blank text"
            )


@dataclass(frozen=True, slots=True)
class TurnMaintenanceDecision:
    """One post-turn rolling-context update and optional durable memory."""

    context_summary: str  # Model-maintained summary used by following turns.
    memory_summary: str | None  # Durable fact selected by the model, or None.
    used_fallback: bool = False  # Whether host fallback replaced a failed model call.

    def __post_init__(self) -> None:
        if (
            not isinstance(self.context_summary, str)
            or not self.context_summary.strip()
        ):
            raise DomainValidationError(
                "maintenance context_summary must be non-blank text"
            )
        if self.memory_summary is not None and (
            not isinstance(self.memory_summary, str)
            or not self.memory_summary.strip()
        ):
            raise DomainValidationError(
                "maintenance memory_summary must be non-blank text or None"
            )
        if not isinstance(self.used_fallback, bool):
            raise DomainValidationError(
                "maintenance used_fallback must be a boolean"
            )
