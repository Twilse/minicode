"""Immutable values used by persistent project memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from minicoder.domain.errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class ProjectMemoryRecord:
    """One bounded model-generated summary from a completed project turn."""

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
