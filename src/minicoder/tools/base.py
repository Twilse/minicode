"""Provider-neutral commands and contracts for locally executed tools."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import ToolDefinition, ToolResult


@dataclass(frozen=True, slots=True)
class ToolCommand:
    """A model tool call whose JSON arguments have passed schema validation."""

    call_id: str  # Correlation identifier copied from the model's ToolCall.
    tool_name: str  # Registered tool selected by the language model.
    arguments: Mapping[str, Any]  # Parsed and validated JSON object for the handler.

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise DomainValidationError("tool command call_id must not be empty")
        if not self.tool_name.strip():
            raise DomainValidationError("tool command name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise DomainValidationError("tool command arguments must be a mapping")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )


class Tool(Protocol):
    """One registered local capability and its provider-neutral definition."""

    @property
    def definition(self) -> ToolDefinition:
        """Describe this tool to the model and the argument validator."""

        ...

    def execute(self, command: ToolCommand) -> ToolResult:
        """Execute one validated command and return a correlated result."""

        ...
