"""Immutable values exchanged between the agent, model, and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from minicoder.domain.errors import DomainValidationError


class MessageRole(str, Enum):
    """Roles supported by the model conversation protocol."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model request to invoke one named local tool."""

    id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise DomainValidationError("tool call id must not be empty")
        if not self.name.strip():
            raise DomainValidationError("tool call name must not be empty")
        if not self.arguments_json.strip():
            raise DomainValidationError("tool call arguments must not be empty")


@dataclass(frozen=True, slots=True)
class Message:
    """One normalized conversation message, independent of an SDK type."""

    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise DomainValidationError("only assistant messages may contain tool calls")
        if self.role is MessageRole.TOOL and not self.tool_call_id:
            raise DomainValidationError("tool messages require a tool_call_id")
        if self.role is not MessageRole.TOOL and self.tool_call_id is not None:
            raise DomainValidationError("only tool messages may contain a tool_call_id")


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    """The normalized result of one model request."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning_content: str | None = None

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

    call_id: str
    tool_name: str
    ok: bool
    content: str
    error_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise DomainValidationError("tool result call_id must not be empty")
        if not self.tool_name.strip():
            raise DomainValidationError("tool result tool_name must not be empty")
        if self.ok and self.error_code is not None:
            raise DomainValidationError("successful tool results cannot have an error_code")
        if not self.ok and not self.error_code:
            raise DomainValidationError("failed tool results require an error_code")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def as_message(self) -> Message:
        """Convert the result into the tool message returned to the model."""

        return Message(
            role=MessageRole.TOOL,
            content=self.content,
            tool_call_id=self.call_id,
        )
