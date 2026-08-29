"""Deterministic tool port used by AgentEngine tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from minicoder.domain.models import ToolCall, ToolDefinition, ToolResult


class FakeToolAdapter:
    """Return scripted correlated results and record calls in execution order."""

    def __init__(
        self,
        results: Iterable[ToolResult] = (),
        *,
        definitions: Sequence[ToolDefinition] = (),
    ) -> None:
        self._results = deque(results)
        self._definitions = tuple(definitions)
        self.calls: list[ToolCall] = []

    def definitions(self) -> Sequence[ToolDefinition]:
        return self._definitions

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if not self._results:
            raise AssertionError("fake tool adapter has no scripted result remaining")
        result = self._results.popleft()
        if result.call_id != call.id or result.tool_name != call.name:
            raise AssertionError("fake tool result does not match the requested call")
        return result
