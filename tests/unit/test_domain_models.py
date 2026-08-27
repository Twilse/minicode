import pytest

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
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
    assert result.metadata["chars"] == 14


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
