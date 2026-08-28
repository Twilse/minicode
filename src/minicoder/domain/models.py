"""Immutable values exchanged between the agent, model, and tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

from minicoder.domain.errors import DomainValidationError


class MessageRole(str, Enum):
    """Roles supported by the model conversation protocol."""

    SYSTEM = "system"  # High-priority instructions that govern the conversation.
    USER = "user"  # A task or follow-up supplied by the user.
    ASSISTANT = "assistant"  # A response produced by the language model.
    TOOL = "tool"  # The result of a tool call returned to the language model.


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model request to invoke one named local tool."""

    id: str  # Model-issued identifier used to correlate the eventual result.
    name: str  # Registered name of the local tool to invoke.
    arguments_json: str  # Original JSON argument payload returned by the model.

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DomainValidationError("tool call id must not be empty")
        if not self.name.strip():
            raise DomainValidationError("tool call name must not be empty")
        if not self.arguments_json.strip():
            raise DomainValidationError("tool call arguments must not be empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A provider-neutral function tool advertised to a language model."""

    name: str  # Stable tool name used in a later ToolCall.
    description: str  # Plain-language guidance that helps the model choose the tool.
    parameters_schema: Mapping[str, Any]  # JSON Schema for the tool arguments object.

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise DomainValidationError("tool definition name must not be empty")
        if not self.description.strip():
            raise DomainValidationError("tool definition description must not be empty")
        if self.parameters_schema.get("type") != "object":
            raise DomainValidationError(
                "tool definition parameters_schema must describe an object"
            )
        object.__setattr__(
            self,
            "parameters_schema",
            MappingProxyType(dict(self.parameters_schema)),
        )


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized conversation message, independent of an SDK type."""

    role: MessageRole  # Protocol role that determines how the message is interpreted.
    content: str | None = None  # Optional visible text carried by the message.
    tool_calls: tuple[ToolCall, ...] = ()  # Tool requests made by an assistant message.
    tool_call_id: str | None = None  # Correlation ID carried only by a tool message.
    reasoning_content: str | None = None  # Optional reasoning exposed by some models.

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise DomainValidationError("message role must be a MessageRole")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise DomainValidationError("only assistant messages may contain tool calls")
        if self.role is MessageRole.TOOL and (
            not self.tool_call_id or not self.tool_call_id.strip()
        ):
            raise DomainValidationError("tool messages require a tool_call_id")
        if self.role is not MessageRole.TOOL and self.tool_call_id is not None:
            raise DomainValidationError("only tool messages may contain a tool_call_id")
        if (
            self.reasoning_content is not None
            and self.role is not MessageRole.ASSISTANT
        ):
            raise DomainValidationError(
                "only assistant messages may contain reasoning_content"
            )


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """The normalized result of one model request."""

    content: str | None  # Optional visible text returned by the language model.
    tool_calls: tuple[ToolCall, ...] = ()  # Local tool invocations requested this turn.
    reasoning_content: str | None = None  # Optional model reasoning kept with the turn.

    def as_message(self) -> Message:
        """Convert the turn into the assistant message stored in history."""

        return Message(
            role=MessageRole.ASSISTANT,
            content=self.content,
            tool_calls=self.tool_calls,
            reasoning_content=self.reasoning_content,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A uniform result returned by every local tool."""

    call_id: str  # Identifier of the ToolCall that requested this result.
    tool_name: str  # Registered name of the tool that produced the result.
    ok: bool  # Whether the tool operation completed successfully.
    content: str  # Human- and model-readable result or error explanation.
    error_code: str | None = None  # Stable machine-readable code for a failed result.
    metadata: Mapping[str, Any] = field(default_factory=dict)  # Host-side result details.

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise DomainValidationError("tool result call_id must not be empty")
        if not self.tool_name.strip():
            raise DomainValidationError("tool result tool_name must not be empty")
        if self.ok and self.error_code is not None:
            raise DomainValidationError("successful tool results cannot have an error_code")
        if not self.ok and (
            not self.error_code or not self.error_code.strip()
        ):
            raise DomainValidationError("failed tool results require an error_code")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def model_content(self) -> str:
        """Return a deterministic provider-neutral envelope for a tool message."""

        return json.dumps(
            {
                "ok": self.ok,
                "content": self.content,
                "error_code": self.error_code,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def as_message(self) -> Message:
        """Convert the result into the tool message returned to the model."""

        return Message(
            role=MessageRole.TOOL,
            content=self.model_content(),
            tool_call_id=self.call_id,
        )


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Provider-neutral facts captured from one local child process."""

    stdout: str  # Standard output decoded and normalized to LF line endings.
    stderr: str  # Standard error decoded and normalized to LF line endings.
    exit_code: int | None  # Child exit status, or None when execution timed out.
    timed_out: bool  # Whether the host terminated the child after its deadline.
    duration_seconds: float  # Monotonic elapsed execution time in seconds.

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise DomainValidationError("process output must be text")
        if not isinstance(self.timed_out, bool):
            raise DomainValidationError("process timed_out must be a boolean")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise DomainValidationError("process exit_code must be an integer or None")
        if self.timed_out and self.exit_code is not None:
            raise DomainValidationError("timed-out processes cannot have an exit_code")
        if not self.timed_out and self.exit_code is None:
            raise DomainValidationError("completed processes require an exit_code")
        if (
            not isinstance(self.duration_seconds, (int, float))
            or isinstance(self.duration_seconds, bool)
            or not isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise DomainValidationError(
                "process duration_seconds must be finite and non-negative"
            )

    def combined_output(self) -> str:
        """Return all captured output while labeling distinct streams when needed."""

        if not self.stdout:
            return self.stderr
        if not self.stderr:
            return self.stdout
        separator = "" if self.stdout.endswith("\n") else "\n"
        return f"[stdout]\n{self.stdout}{separator}[stderr]\n{self.stderr}"
