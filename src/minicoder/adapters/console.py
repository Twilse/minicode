"""Concise terminal observer for sanitized agent events."""

from __future__ import annotations

import sys
from typing import TextIO

from minicoder.domain.events import AgentEvent, AgentEventKind


class ConsoleEventSink:
    """Render one human-readable line per agent event without hidden reasoning."""

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = sys.stdout if output is None else output

    def handle(self, event: AgentEvent) -> None:
        print(_format_event(event), file=self._output, flush=True)


def _format_event(event: AgentEvent) -> str:
    details = event.details
    if event.kind is AgentEventKind.TASK_STARTED:
        return (
            f"[TASK] started ({details['tool_count']} tools, "
            f"max {details['max_steps']} model steps)"
        )
    if event.kind is AgentEventKind.MODEL_REQUESTED:
        return (
            f"[MODEL] step {event.model_step} requested "
            f"({details['message_count']} messages)"
        )
    if event.kind is AgentEventKind.MODEL_RETRY_SCHEDULED:
        return (
            f"[MODEL] retry {details['retry_number']} after "
            f"{details['error_type']} ({details['delay_seconds']}s)"
        )
    if event.kind is AgentEventKind.CONTEXT_COMPACTED:
        return (
            f"[CONTEXT] compacted {details['original_chars']} -> "
            f"{details['prepared_chars']} chars "
            f"({details['omitted_message_count']} messages omitted)"
        )
    if event.kind is AgentEventKind.COMPLETION_REJECTED:
        return (
            f"[REVIEW] final response rejected ({details['reason']}; "
            f"{details['modified_file_count']} modified files)"
        )
    if event.kind is AgentEventKind.TOOL_CALLED:
        return (
            f"[TOOL] {details['tool_name']} called "
            f"(call_id={details['call_id']})"
        )
    if event.kind is AgentEventKind.TOOL_FINISHED:
        status = "succeeded"
        if not details["ok"]:
            status = f"failed ({details['error_code']})"
        return (
            f"[TOOL] {details['tool_name']} {status} "
            f"(call_id={details['call_id']})"
        )
    if event.kind is AgentEventKind.VERIFICATION_PASSED:
        return f"[VERIFY] {details['verification_kind']} passed"
    if event.kind is AgentEventKind.TASK_COMPLETED:
        return f"[DONE] completed after {event.model_step} model steps"
    if event.kind is AgentEventKind.TASK_FAILED:
        return (
            f"[FAILED] {details['reason']} after {event.model_step} model steps: "
            f"{details['message']}"
        )
    return f"[EVENT] {event.kind.value}"
