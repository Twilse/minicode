"""Ports owned by the application core and implemented by outer adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from minicoder.domain.models import (
    AssistantTurn,
    Message,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class ModelPort(Protocol):
    """Obtain one normalized assistant turn without exposing an SDK type."""

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        """Send one conversation snapshot and return one assistant turn."""

        ...


class ToolPort(Protocol):
    """Expose registered local tools without coupling the core to dispatch details."""

    def definitions(self) -> Sequence[ToolDefinition]:
        """Return the tool schemas that may be advertised to a model."""

        ...

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate and execute one model-issued tool call."""

        ...
