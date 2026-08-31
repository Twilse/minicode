"""Build bounded model-facing context from an exact previous-session archive."""

from __future__ import annotations

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import MessageRole
from minicoder.domain.session import RecentSessionContext

DEFAULT_RECENT_SESSION_CONTEXT_CHARS = 18_000


def format_recent_session_context(
    context: RecentSessionContext | None,
    *,
    max_chars: int = DEFAULT_RECENT_SESSION_CONTEXT_CHARS,
) -> str:
    """Return recent-session data without treating old text as instructions."""

    if context is None:
        return ""
    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars <= 0
    ):
        raise DomainValidationError(
            "recent session context max_chars must be a positive integer"
        )

    header = (
        f"Previous process session: {context.session_id}\n"
        f"Last task: {context.last_task}\n"
        f"Last status: {context.status.value}\n"
        f"Stop reason: {context.stop_reason or 'not recorded'}\n"
        "Rolling summary:\n"
        f"{context.context_summary.strip()}"
    )
    lines = [header, "Recent archived conversation tail:"]
    for message in context.recent_messages:
        if message.role is MessageRole.SYSTEM:
            continue
        if message.role is MessageRole.ASSISTANT and message.tool_calls:
            calls = ", ".join(
                f"{call.name}({call.arguments_json})" for call in message.tool_calls
            )
            lines.append(f"assistant tool calls: {calls}")
        if message.content:
            lines.append(f"{message.role.value}: {message.content}")
    return _fit_recent_lines(lines, max_chars)


def _fit_recent_lines(lines: list[str], max_chars: int) -> str:
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    marker = "\n...[older recovered context omitted]...\n"
    first = lines[0]
    available_tail = max_chars - len(first) - len(marker) - 1
    if available_tail <= 0:
        return first[:max_chars]
    tail = "\n".join(lines[1:])
    return f"{first}\n{marker}{tail[-available_tail:]}"
