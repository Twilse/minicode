"""Build a minimal history-area boundary for an unfinished previous process."""

from __future__ import annotations

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.session import ArchivedTurnStatus, RecentSessionContext

DEFAULT_RECENT_SESSION_BOUNDARY_CHARS = 4_000


def format_recent_session_boundary(
    context: RecentSessionContext | None,
    *,
    max_chars: int = DEFAULT_RECENT_SESSION_BOUNDARY_CHARS,
) -> str:
    """Return only an unfinished-process boundary; exact messages restore details."""

    if context is None or context.status is ArchivedTurnStatus.COMPLETE:
        return ""
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
    ):
        raise DomainValidationError(
            "recent session boundary max_chars must be a positive integer"
        )

    text = (
        "The previous MiniCoder process ended before successful completion.\n"
        f"Status: {context.status.value}\n"
        f"Stop reason: {context.stop_reason or 'not recorded'}"
    )
    return text if len(text) <= max_chars else text[:max_chars]
