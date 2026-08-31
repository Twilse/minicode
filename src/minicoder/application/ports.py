"""Ports owned by the application core and implemented by outer adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from minicoder.domain.events import AgentEvent
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    ProcessResult,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from minicoder.domain.session import RecentSessionContext
from minicoder.domain.state import AgentRunResult


class EventSinkPort(Protocol):
    """Consume one audit event without influencing agent decisions."""

    def handle(self, event: AgentEvent) -> None:
        """Render or persist one already-sanitized event."""

        ...


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


class ProjectMemoryPort(Protocol):
    """Load and append workspace-scoped project memories."""

    def load_recent(self) -> Sequence[ProjectMemoryRecord]:
        """Return recent valid records in chronological order."""

        ...

    def append(self, record: ProjectMemoryRecord) -> None:
        """Persist one model-selected durable memory without changing older records."""

        ...


class SessionArchivePort(Protocol):
    """Persist exact model exchanges and restore the previous process context."""

    @property
    def session_id(self) -> str:
        """Return the identifier of the process-owned archive."""

        ...

    def load_latest_context(self) -> RecentSessionContext | None:
        """Load the latest usable session for the same workspace."""

        ...

    def record_turn_started(self, *, task: str, turn_index: int) -> None:
        """Persist the exact external request before the agent starts work."""

        ...

    def record_model_request(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        request_kind: str,
        turn_index: int,
        model_step: int,
    ) -> None:
        """Persist one exact normalized request, including current tool schemas."""

        ...

    def record_model_response(
        self,
        *,
        turn: AssistantTurn,
        request_kind: str,
        turn_index: int,
        model_step: int,
    ) -> None:
        """Persist one exact normalized model response."""

        ...

    def record_tool_result(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
        turn_index: int,
        model_step: int,
    ) -> None:
        """Persist the complete local result and host metadata."""

        ...

    def record_turn_result(
        self,
        *,
        task: str,
        result: AgentRunResult,
        turn_index: int,
    ) -> None:
        """Persist one terminal turn snapshot, including failures."""

        ...

    def record_maintenance(
        self,
        *,
        context_summary: str,
        memory_summary: str | None,
        used_fallback: bool,
        turn_index: int,
        model_step: int,
    ) -> None:
        """Persist the post-turn model maintenance decision."""

        ...

    def close(self, *, context_summary: str | None) -> None:
        """Mark a normal process close while retaining every prior record."""

        ...


class ToolPort(Protocol):
    """Expose registered local tools without coupling the core to dispatch details."""

    def definitions(self) -> Sequence[ToolDefinition]:
        """Return the tool schemas that may be advertised to a model."""

        ...

    def execute(self, call: ToolCall) -> ToolResult:
        """Validate and execute one model-issued tool call."""

        ...


class ProcessPort(Protocol):
    """Execute one non-interactive child process without exposing subprocess types."""

    def run(
        self,
        *,
        argv: Sequence[str],
        cwd: Path,
        timeout_seconds: float,
    ) -> ProcessResult:
        """Run an argument vector in one directory and capture its bounded lifetime."""

        ...
