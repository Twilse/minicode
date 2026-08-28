from __future__ import annotations

from typing import Any

import pytest

from minicoder.application.ports import ToolPort
from minicoder.domain.errors import ToolRegistrationError
from minicoder.domain.models import ToolCall, ToolDefinition, ToolResult
from minicoder.tools.base import ToolCommand
from minicoder.tools.registry import (
    INVALID_ARGUMENTS,
    TOOL_CONTRACT_ERROR,
    TOOL_EXECUTION_ERROR,
    UNKNOWN_TOOL,
    ToolRegistry,
)


def _definition(
    name: str = "read_file",
    *,
    schema: dict[str, Any] | None = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Execute {name} for a test.",
        parameters_schema=(
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            }
            if schema is None
            else schema
        ),
    )


class RecordingTool:
    def __init__(
        self,
        definition: ToolDefinition | None = None,
        *,
        result: ToolResult | None = None,
    ) -> None:
        self._definition = _definition() if definition is None else definition
        self._result = result
        self.commands: list[ToolCommand] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        self.commands.append(command)
        if self._result is not None:
            return self._result
        return ToolResult(
            call_id=command.call_id,
            tool_name=command.tool_name,
            ok=True,
            content=f"read {command.arguments['path']}",
        )


def test_registry_exposes_definitions_and_dispatches_a_valid_command() -> None:
    read_tool = RecordingTool()
    search_tool = RecordingTool(
        _definition(
            "search_text",
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    )
    registry: ToolPort = ToolRegistry((read_tool, search_tool))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
    )

    assert [definition.name for definition in registry.definitions()] == [
        "read_file",
        "search_text",
    ]
    assert result.ok is True
    assert result.content == "read main.py"
    assert read_tool.commands[0].arguments == {"path": "main.py"}
    with pytest.raises(TypeError):
        read_tool.commands[0].arguments["path"] = "other.py"  # type: ignore[index]


def test_registry_returns_a_correlated_result_for_an_unknown_tool() -> None:
    registry = ToolRegistry((RecordingTool(),))

    result = registry.execute(
        ToolCall(id="call-9", name="delete_everything", arguments_json="{}")
    )

    assert result == ToolResult(
        call_id="call-9",
        tool_name="delete_everything",
        ok=False,
        content=(
            "Unknown tool 'delete_everything'. Available tools: read_file."
        ),
        error_code=UNKNOWN_TOOL,
    )


@pytest.mark.parametrize(
    ("arguments_json", "message_part"),
    [
        ("{", "not valid JSON"),
        ("[]", "must be an object"),
        ('{"path":NaN}', "non-JSON value"),
        ('{"path":"a","path":"b"}', "duplicate field"),
    ],
)
def test_registry_rejects_malformed_json_arguments(
    arguments_json: str,
    message_part: str,
) -> None:
    tool = RecordingTool()
    registry = ToolRegistry((tool,))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json=arguments_json,
        )
    )

    assert result.ok is False
    assert result.error_code == INVALID_ARGUMENTS
    assert message_part in result.content
    assert tool.commands == []


@pytest.mark.parametrize(
    ("arguments_json", "path_part"),
    [
        ("{}", "$"),
        ('{"path":3}', "$['path']"),
        ('{"path":"main.py","extra":true}', "$"),
    ],
)
def test_registry_rejects_arguments_that_violate_the_tool_schema(
    arguments_json: str,
    path_part: str,
) -> None:
    tool = RecordingTool()
    registry = ToolRegistry((tool,))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json=arguments_json,
        )
    )

    assert result.error_code == INVALID_ARGUMENTS
    assert "schema validation" in result.content
    assert path_part in result.content
    assert tool.commands == []


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry((RecordingTool(),))

    with pytest.raises(ToolRegistrationError, match="already registered"):
        registry.register(RecordingTool())


def test_registry_rejects_an_invalid_json_schema_during_registration() -> None:
    invalid_definition = _definition(
        schema={
            "type": "object",
            "properties": {"path": {"type": "not-a-json-schema-type"}},
        }
    )

    with pytest.raises(ToolRegistrationError, match="invalid parameters schema"):
        ToolRegistry((RecordingTool(invalid_definition),))


def test_compiled_schema_is_not_changed_by_later_nested_mutation() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    registry = ToolRegistry((RecordingTool(_definition(schema=schema)),))
    schema["properties"]["path"]["type"] = "integer"

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
    )

    assert result.ok is True


def test_registry_converts_an_unexpected_handler_exception_to_a_result() -> None:
    tool = RecordingTool()

    def explode(command: ToolCommand) -> ToolResult:
        raise RuntimeError(f"unexpected failure for {command.tool_name}")

    tool.execute = explode  # type: ignore[method-assign]
    registry = ToolRegistry((tool,))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
    )

    assert result.error_code == TOOL_EXECUTION_ERROR
    assert result.metadata == {"exception_type": "RuntimeError"}
    assert "unexpected failure for" not in result.content


@pytest.mark.parametrize(
    "returned_result",
    [
        "not a ToolResult",
        ToolResult(
            call_id="wrong-call",
            tool_name="read_file",
            ok=True,
            content="wrong correlation",
        ),
        ToolResult(
            call_id="call-1",
            tool_name="wrong_tool",
            ok=True,
            content="wrong correlation",
        ),
    ],
)
def test_registry_rejects_results_that_break_the_tool_contract(
    returned_result: object,
) -> None:
    tool = RecordingTool(result=returned_result)  # type: ignore[arg-type]
    registry = ToolRegistry((tool,))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json='{"path":"main.py"}',
        )
    )

    assert result.error_code == TOOL_CONTRACT_ERROR
    assert result.call_id == "call-1"
    assert result.tool_name == "read_file"


def test_registry_preserves_an_expected_failure_returned_by_a_tool() -> None:
    expected = ToolResult(
        call_id="call-1",
        tool_name="read_file",
        ok=False,
        content="File does not exist.",
        error_code="FILE_NOT_FOUND",
    )
    registry = ToolRegistry((RecordingTool(result=expected),))

    result = registry.execute(
        ToolCall(
            id="call-1",
            name="read_file",
            arguments_json='{"path":"missing.py"}',
        )
    )

    assert result is expected
