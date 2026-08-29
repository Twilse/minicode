import json

import pytest

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ProcessResult,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def test_assistant_turn_converts_to_history_message() -> None:
    call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"path":"main.py"}',
    )

    message = AssistantTurn(content=None, tool_calls=(call,)).as_message()

    assert message.role is MessageRole.ASSISTANT
    assert message.tool_calls == (call,)


@pytest.mark.parametrize("content", [None, "", "   ", "\t\n"])
def test_assistant_turn_without_tools_requires_final_text(
    content: str | None,
) -> None:
    with pytest.raises(DomainValidationError, match="non-blank content"):
        AssistantTurn(content=content)


def test_assistant_turn_only_keeps_reasoning_for_tool_continuation() -> None:
    with pytest.raises(DomainValidationError, match="tool-calling turn"):
        AssistantTurn(
            content="final answer",
            reasoning_content="provider continuation state",
        )


def test_tool_result_converts_to_correlated_tool_message() -> None:
    result = ToolResult(
        call_id="call-1",
        tool_name="read_file",
        ok=True,
        content="print('hello')",
        metadata={"chars": 14},
    )

    message = result.as_message()

    assert message.role is MessageRole.TOOL
    assert message.tool_call_id == "call-1"
    assert json.loads(message.content or "") == {
        "ok": True,
        "content": "print('hello')",
        "error_code": None,
        "metadata": {},
    }
    assert "chars" not in (message.content or "")
    assert result.metadata["chars"] == 14


def test_tool_result_only_exposes_explicit_model_metadata() -> None:
    result = ToolResult(
        call_id="call-1",
        tool_name="run_command",
        ok=True,
        content="tests passed",
        metadata={"argv": ["pytest"], "exit_code": 0},
        model_metadata={"exit_code": 0},
    )

    payload = json.loads(result.model_content())

    assert payload["metadata"] == {"exit_code": 0}
    assert "argv" not in result.model_content()


def test_tool_result_rejects_non_json_model_metadata() -> None:
    with pytest.raises(DomainValidationError, match="model_metadata"):
        ToolResult(
            call_id="call-1",
            tool_name="read_file",
            ok=True,
            content="ok",
            model_metadata={"unsafe": object()},
        )


def test_tool_result_copies_metadata_to_preserve_immutability() -> None:
    source = {"exit_code": 0}
    result = ToolResult(
        call_id="call-1",
        tool_name="run_command",
        ok=True,
        content="ok",
        metadata=source,
    )

    source["exit_code"] = 1

    assert result.metadata["exit_code"] == 0
    with pytest.raises(TypeError):
        result.metadata["exit_code"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "call",
    [
        {"id": "", "name": "read_file", "arguments_json": "{}"},
        {"id": "1", "name": "", "arguments_json": "{}"},
        {"id": "1", "name": "read_file", "arguments_json": ""},
    ],
)
def test_tool_call_rejects_empty_protocol_fields(call: dict[str, str]) -> None:
    with pytest.raises(DomainValidationError):
        ToolCall(**call)


def test_message_rejects_tool_call_on_user_role() -> None:
    call = ToolCall(id="1", name="read_file", arguments_json="{}")

    with pytest.raises(DomainValidationError, match="only assistant"):
        Message(role=MessageRole.USER, tool_calls=(call,))


@pytest.mark.parametrize("tool_call_id", [None, "", "   ", "\t\n"])
def test_tool_message_requires_non_blank_correlation_id(
    tool_call_id: str | None,
) -> None:
    with pytest.raises(DomainValidationError, match="tool_call_id"):
        Message(
            role=MessageRole.TOOL,
            content="result",
            tool_call_id=tool_call_id,
        )


@pytest.mark.parametrize("error_code", [None, "", "   ", "\t\n"])
def test_failed_tool_result_requires_non_blank_error_code(
    error_code: str | None,
) -> None:
    with pytest.raises(DomainValidationError, match="require an error_code"):
        ToolResult(
            call_id="call-1",
            tool_name="read_file",
            ok=False,
            content="missing",
            error_code=error_code,
        )


def test_message_rejects_plain_string_role_at_runtime() -> None:
    with pytest.raises(DomainValidationError, match="MessageRole"):
        Message(role="user", content="hello")  # type: ignore[arg-type]


def test_tool_definition_copies_its_top_level_schema() -> None:
    schema = {"type": "object", "properties": {}}
    definition = ToolDefinition(
        name="read_file",
        description="Read one text file.",
        parameters_schema=schema,
    )

    schema["type"] = "string"

    assert definition.parameters_schema["type"] == "object"
    with pytest.raises(TypeError):
        definition.parameters_schema["type"] = "string"  # type: ignore[index]


@pytest.mark.parametrize(
    "definition",
    [
        {
            "name": "",
            "description": "Read one text file.",
            "parameters_schema": {"type": "object"},
        },
        {
            "name": "read_file",
            "description": "   ",
            "parameters_schema": {"type": "object"},
        },
        {
            "name": "read_file",
            "description": "Read one text file.",
            "parameters_schema": {"type": "string"},
        },
    ],
)
def test_tool_definition_rejects_invalid_protocol_fields(
    definition: dict[str, object],
) -> None:
    with pytest.raises(DomainValidationError):
        ToolDefinition(**definition)  # type: ignore[arg-type]


def test_message_rejects_reasoning_on_non_assistant_role() -> None:
    with pytest.raises(DomainValidationError, match="only assistant"):
        Message(
            role=MessageRole.USER,
            content="hello",
            reasoning_content="hidden reasoning",
        )


def test_process_result_combines_distinct_output_streams() -> None:
    result = ProcessResult(
        stdout="tests started\n",
        stderr="one warning\n",
        exit_code=0,
        timed_out=False,
        duration_seconds=0.25,
    )

    assert result.combined_output() == (
        "[stdout]\ntests started\n[stderr]\none warning\n"
    )


@pytest.mark.parametrize(
    "values",
    [
        {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "timed_out": True,
            "duration_seconds": 1.0,
        },
        {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": 1.0,
        },
        {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": -1.0,
        },
        {
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "duration_seconds": "slow",
        },
    ],
)
def test_process_result_rejects_inconsistent_facts(values: dict[str, object]) -> None:
    with pytest.raises(DomainValidationError):
        ProcessResult(**values)  # type: ignore[arg-type]
