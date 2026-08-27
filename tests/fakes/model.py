"""Deterministic model adapter used by application tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from minicoder.domain.models import AssistantTurn, Message, ToolDefinition


@dataclass(frozen=True, slots=True)
class RecordedModelRequest:
    """One immutable request observed by the fake adapter."""

    messages: tuple[Message, ...]  # Conversation snapshot supplied by the core.
    tools: tuple[ToolDefinition, ...]  # Tool definitions supplied with the request.


class FakeModelAdapter:
    """Return scripted turns and retain requests without any network access."""

    def __init__(self, turns: Iterable[AssistantTurn]) -> None:
        self._turns = deque(turns)
        self.requests: list[RecordedModelRequest] = []

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        self.requests.append(
            RecordedModelRequest(
                messages=tuple(messages),
                tools=tuple(tools),
            )
        )
        if not self._turns:
            raise AssertionError("fake model has no scripted turn remaining")
        return self._turns.popleft()
