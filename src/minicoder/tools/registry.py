"""Registry and Command dispatcher for locally implemented tools."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from minicoder.domain.errors import ToolRegistrationError
from minicoder.domain.models import ToolCall, ToolDefinition, ToolResult
from minicoder.tools.base import Tool, ToolCommand
from minicoder.tools.validation import (
    ToolArgumentsError,
    compile_arguments_validator,
    parse_and_validate_arguments,
)

UNKNOWN_TOOL = "UNKNOWN_TOOL"
INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
TOOL_CONTRACT_ERROR = "TOOL_CONTRACT_ERROR"


@dataclass(frozen=True, slots=True)
class _RegisteredTool:
    tool: Tool
    definition: ToolDefinition
    validator: Draft202012Validator


class ToolRegistry:
    """Register tool definitions and dispatch validated model commands by name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, _RegisteredTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Add one tool and reject ambiguous names or invalid schemas early."""

        definition = tool.definition
        if definition.name in self._tools:
            raise ToolRegistrationError(
                f"tool {definition.name!r} is already registered"
            )
        validator = compile_arguments_validator(definition)
        self._tools[definition.name] = _RegisteredTool(
            tool=tool,
            definition=definition,
            validator=validator,
        )

    def definitions(self) -> Sequence[ToolDefinition]:
        """Return definitions in stable registration order for the model request."""

        return tuple(entry.definition for entry in self._tools.values())

    def execute(self, call: ToolCall) -> ToolResult:
        """Dispatch one raw model call and turn recoverable failures into results."""

        registered = self._tools.get(call.name)
        if registered is None:
            available = ", ".join(sorted(self._tools)) or "none"
            return _failure_result(
                call,
                error_code=UNKNOWN_TOOL,
                content=(
                    f"Unknown tool {call.name!r}. Available tools: {available}."
                ),
            )

        try:
            arguments = parse_and_validate_arguments(
                call.arguments_json,
                registered.validator,
            )
        except ToolArgumentsError as exc:
            return _failure_result(
                call,
                error_code=INVALID_ARGUMENTS,
                content=str(exc),
            )

        command = ToolCommand(
            call_id=call.id,
            tool_name=call.name,
            arguments=arguments,
        )
        try:
            result = registered.tool.execute(command)
        except Exception as exc:
            return _failure_result(
                call,
                error_code=TOOL_EXECUTION_ERROR,
                content=f"Tool {call.name!r} failed unexpectedly.",
                metadata={"exception_type": type(exc).__name__},
            )

        if not isinstance(result, ToolResult):
            return _failure_result(
                call,
                error_code=TOOL_CONTRACT_ERROR,
                content=f"Tool {call.name!r} returned an invalid result.",
            )
        if result.call_id != call.id or result.tool_name != call.name:
            return _failure_result(
                call,
                error_code=TOOL_CONTRACT_ERROR,
                content=f"Tool {call.name!r} returned an uncorrelated result.",
            )
        return result


def _failure_result(
    call: ToolCall,
    *,
    error_code: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=content,
        error_code=error_code,
        metadata={} if metadata is None else metadata,
    )
