"""Deterministic conversation budgeting without provider tokenizers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import Message, MessageRole

_SUMMARY_HEADING = "\n\n[Earlier conversation summary]\n"
_SHORTENING_MARKER = "\n...[shortened for context budget]...\n"
_MODIFYING_TOOLS = frozenset({"create_file", "replace_text"})


@dataclass(frozen=True, slots=True)
class ContextWindow:
    """One model-ready history snapshot and its compaction measurements."""

    messages: tuple[Message, ...]  # Messages selected for the next model request.
    budget_chars: int  # Configured approximate character budget.
    original_chars: int  # Estimated size before context management.
    prepared_chars: int  # Estimated size of the selected model-facing history.
    omitted_message_count: int = 0  # Older messages represented only by a summary.
    shortened_message_count: int = 0  # Retained messages whose content was shortened.

    @property
    def compacted(self) -> bool:
        return self.omitted_message_count > 0 or self.shortened_message_count > 0

    @property
    def budget_exceeded(self) -> bool:
        """Report an unavoidable overflow caused by permanent/protocol fields."""

        return self.prepared_chars > self.budget_chars


class ContextSummaryStrategy(Protocol):
    """Produce a bounded replacement for older conversation messages."""

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        """Return deterministic text without exposing private reasoning state."""

        ...


class DeterministicContextSummary:
    """Summarize visible actions and outcomes without another model request."""

    def summarize(self, messages: Sequence[Message], *, max_chars: int) -> str:
        if max_chars <= 0 or not messages:
            return ""

        calls: dict[str, tuple[str, str | None]] = {}
        actions: list[str] = []
        modified_files: list[str] = []
        failures: list[str] = []
        findings: list[str] = []
        user_requests: list[str] = []

        for message in messages:
            if message.role is MessageRole.USER:
                if message.content and message.content.strip():
                    user_requests.append(_one_line(message.content, 120))
                continue

            if message.role is MessageRole.ASSISTANT:
                if message.content and message.content.strip():
                    findings.append(f"assistant: {_one_line(message.content, 120)}")
                for call in message.tool_calls:
                    path = _argument_path(call.arguments_json)
                    calls[call.id] = (call.name, path)
                    action = call.name if path is None else f"{call.name}({path})"
                    actions.append(action)
                    if call.name in _MODIFYING_TOOLS and path is not None:
                        modified_files.append(path)
                continue

            if message.role is not MessageRole.TOOL:
                continue
            name, path = calls.get(
                message.tool_call_id or "",
                ("unknown_tool", None),
            )
            payload = _tool_payload(message.content)
            if payload is None:
                if message.content and message.content.strip():
                    findings.append(
                        f"{name}: {_one_line(message.content, 120)}"
                    )
                continue

            content = payload.get("content")
            if payload.get("ok") is False:
                error_code = payload.get("error_code")
                label = name if path is None else f"{name}({path})"
                detail = error_code if isinstance(error_code, str) else "failed"
                if isinstance(content, str) and content.strip():
                    detail = f"{detail}: {_one_line(content, 100)}"
                failures.append(f"{label} -> {detail}")
            elif isinstance(content, str) and content.strip():
                findings.append(f"{name}: {_one_line(content, 120)}")

        lines = [
            _summary_line("Earlier user requests", user_requests[-4:], limit=4),
            f"Removed messages: {len(messages)}.",
            _summary_line("Tool activity", actions, limit=8),
            _summary_line("Modified files", _unique(modified_files), limit=8),
            _summary_line("Recorded failures", failures, limit=5),
            _summary_line("Key visible results", findings[-4:], limit=4),
        ]
        return _fit_summary_lines(lines, max_chars)


class ContextManager:
    """Keep permanent instructions and recent protocol-complete exchanges."""

    def __init__(
        self,
        *,
        budget_chars: int,
        summary_strategy: ContextSummaryStrategy | None = None,
    ) -> None:
        if (
            not isinstance(budget_chars, int)
            or isinstance(budget_chars, bool)
            or budget_chars <= 0
        ):
            raise DomainValidationError(
                "context budget_chars must be a positive integer"
            )
        self._budget_chars = budget_chars
        self._summary_strategy = (
            DeterministicContextSummary()
            if summary_strategy is None
            else summary_strategy
        )

    @property
    def budget_chars(self) -> int:
        return self._budget_chars

    def prepare(
        self,
        messages: Sequence[Message],
        *,
        current_user_index: int | None = None,
    ) -> ContextWindow:
        """Return one bounded snapshot while retaining the full input elsewhere."""

        original = tuple(messages)
        if len(original) < 2:
            raise DomainValidationError(
                "context requires an initial system message and user task"
            )
        if original[0].role is not MessageRole.SYSTEM:
            raise DomainValidationError("context first message must be system")
        if any(message.role is MessageRole.SYSTEM for message in original[1:]):
            raise DomainValidationError(
                "context may contain only one system message"
            )

        if current_user_index is None:
            user_indexes = tuple(
                index
                for index, message in enumerate(original)
                if message.role is MessageRole.USER
            )
            if not user_indexes:
                raise DomainValidationError("context requires a user message")
            current_user_index = user_indexes[-1]
        if (
            not isinstance(current_user_index, int)
            or isinstance(current_user_index, bool)
            or current_user_index <= 0
            or current_user_index >= len(original)
            or original[current_user_index].role is not MessageRole.USER
        ):
            raise DomainValidationError(
                "context current_user_index must identify a user message"
            )

        original_chars = conversation_char_count(original)
        if original_chars <= self._budget_chars:
            return ContextWindow(
                messages=original,
                budget_chars=self._budget_chars,
                original_chars=original_chars,
                prepared_chars=original_chars,
            )

        system = original[0]
        current_user = original[current_user_index]
        before_groups = _conversation_groups(original[1:current_user_index])
        after_groups = _conversation_groups(original[current_user_index + 1 :])
        summary_reserve = min(1_200, max(160, self._budget_chars // 5))
        recent_budget = max(
            0,
            self._budget_chars
            - conversation_char_count((system, current_user))
            - len(_SUMMARY_HEADING)
            - summary_reserve,
        )
        recent_after = _select_recent_groups(
            after_groups,
            recent_budget,
            keep_latest=True,
        )
        remaining_budget = max(
            0,
            recent_budget
            - conversation_char_count(
                tuple(message for group in recent_after for message in group)
            ),
        )
        recent_before = _select_recent_groups(
            before_groups,
            remaining_budget,
            keep_latest=False,
        )
        omitted_before = before_groups[: len(before_groups) - len(recent_before)]
        omitted_after = after_groups[: len(after_groups) - len(recent_after)]
        omitted = tuple(
            message
            for group in (*omitted_before, *omitted_after)
            for message in group
        )
        before = tuple(message for group in recent_before for message in group)
        after = tuple(message for group in recent_after for message in group)

        summary = self._summary_strategy.summarize(
            omitted,
            max_chars=summary_reserve,
        )
        prepared = (
            _with_summary(system, summary)
            + before
            + (current_user,)
            + after
        )
        shortened = 0
        if conversation_char_count(prepared) > self._budget_chars:
            prepared, shortened = _shorten_recent_content(
                prepared,
                recent_start=1,
                budget_chars=self._budget_chars,
            )
        if conversation_char_count(prepared) > self._budget_chars and summary:
            prepared = (system,) + prepared[1:]

        prepared_chars = conversation_char_count(prepared)
        return ContextWindow(
            messages=prepared,
            budget_chars=self._budget_chars,
            original_chars=original_chars,
            prepared_chars=prepared_chars,
            omitted_message_count=len(omitted),
            shortened_message_count=shortened,
        )


def conversation_char_count(messages: Sequence[Message]) -> int:
    """Estimate provider payload size from normalized text-bearing fields."""

    return sum(_message_char_count(message) for message in messages)


def _message_char_count(message: Message) -> int:
    total = len(message.role.value)
    total += len(message.content or "")
    total += len(message.tool_call_id or "")
    total += len(message.reasoning_content or "")
    for call in message.tool_calls:
        total += len(call.id) + len(call.name) + len(call.arguments_json)
    return total


def _conversation_groups(
    messages: Sequence[Message],
) -> tuple[tuple[Message, ...], ...]:
    groups: list[tuple[Message, ...]] = []
    index = 0
    while index < len(messages):
        first = messages[index]
        group = [first]
        index += 1
        if first.role is MessageRole.ASSISTANT and first.tool_calls:
            while index < len(messages) and messages[index].role is MessageRole.TOOL:
                group.append(messages[index])
                index += 1
        groups.append(tuple(group))
    return tuple(groups)


def _select_recent_groups(
    groups: Sequence[tuple[Message, ...]],
    budget_chars: int,
    *,
    keep_latest: bool = True,
) -> tuple[tuple[Message, ...], ...]:
    if not groups:
        return ()
    selected: list[tuple[Message, ...]] = []
    used = 0
    for group in reversed(groups):
        group_chars = conversation_char_count(group)
        if used + group_chars > budget_chars and (
            selected or not keep_latest
        ):
            break
        selected.append(group)
        used += group_chars
        if used >= budget_chars:
            break
    selected.reverse()
    return tuple(selected)


def _with_summary(system: Message, summary: str) -> tuple[Message, ...]:
    if not summary:
        return (system,)
    summarized_system = Message(
        role=MessageRole.SYSTEM,
        content=f"{system.content or ''}{_SUMMARY_HEADING}{summary}",
    )
    return (summarized_system,)


def _shorten_recent_content(
    messages: tuple[Message, ...],
    *,
    recent_start: int,
    budget_chars: int,
) -> tuple[tuple[Message, ...], int]:
    prepared = list(messages)
    shortened = 0
    for index in range(recent_start, len(prepared)):
        excess = conversation_char_count(prepared) - budget_chars
        if excess <= 0:
            break
        message = prepared[index]
        if message.role is not MessageRole.TOOL or not message.content:
            continue
        target_chars = max(0, len(message.content) - excess)
        replacement = _message_with_shorter_content(message, target_chars)
        if replacement.content != message.content:
            prepared[index] = replacement
            shortened += 1
    return tuple(prepared), shortened


def _message_with_shorter_content(message: Message, max_chars: int) -> Message:
    content = message.content or ""
    if message.role is MessageRole.TOOL:
        payload = _tool_payload(content)
        if payload is not None and isinstance(payload.get("content"), str):
            original_body = payload["content"]
            low = 0
            high = len(original_body)
            minimal = dict(payload)
            minimal["content"] = ""
            best = json.dumps(
                minimal,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            while low <= high:
                middle = (low + high) // 2
                candidate = dict(payload)
                candidate["content"] = _shorten_text(original_body, middle)
                serialized = json.dumps(
                    candidate,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if len(serialized) <= max_chars:
                    best = serialized
                    low = middle + 1
                else:
                    high = middle - 1
            if len(best) < len(content):
                return Message(
                    role=MessageRole.TOOL,
                    content=best,
                    tool_call_id=message.tool_call_id,
                )
            return message

    shortened = _shorten_text(content, max_chars)
    if shortened == content:
        return message
    return Message(
        role=message.role,
        content=shortened,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
    )


def _tool_payload(content: str | None) -> dict[str, Any] | None:
    if content is None:
        return None
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _argument_path(arguments_json: str) -> str | None:
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) and path else None


def _summary_line(label: str, values: Sequence[str], *, limit: int) -> str:
    selected = list(values[:limit])
    if not selected:
        return ""
    suffix = f"; +{len(values) - limit} more" if len(values) > limit else ""
    return f"{label}: {'; '.join(selected)}{suffix}."


def _fit_summary_lines(lines: Sequence[str], max_chars: int) -> str:
    selected: list[str] = []
    used = 0
    for line in lines:
        if not line:
            continue
        separator_chars = 1 if selected else 0
        remaining = max_chars - used - separator_chars
        if remaining <= 0:
            break
        selected.append(_shorten_text(line, remaining))
        used += separator_chars + len(selected[-1])
        if len(line) > remaining:
            break
    return "\n".join(selected)


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _one_line(text: str, max_chars: int) -> str:
    return _shorten_text(" ".join(text.split()), max_chars)


def _shorten_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_SHORTENING_MARKER):
        return _SHORTENING_MARKER.strip()[:max_chars]
    remaining = max_chars - len(_SHORTENING_MARKER)
    head_chars = remaining * 3 // 10
    tail_chars = remaining - head_chars
    return f"{text[:head_chars]}{_SHORTENING_MARKER}{text[-tail_chars:]}"
