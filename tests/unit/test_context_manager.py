import json
from collections.abc import Sequence

from minicoder.application.context import (
    ContextManager,
    conversation_char_count,
)
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)


def _exchange(
    index: int,
    *,
    name: str,
    path: str,
    content: str,
    reasoning: str | None = None,
    ok: bool = True,
) -> tuple[Message, Message]:
    call = ToolCall(
        id=f"call-{index}",
        name=name,
        arguments_json=json.dumps({"path": path}, separators=(",", ":")),
    )
    turn = AssistantTurn(
        content=None,
        tool_calls=(call,),
        reasoning_content=reasoning,
    )
    result = ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=ok,
        content=content,
        error_code=None if ok else "TOOL_FAILED",
    )
    return turn.as_message(), result.as_message()


def test_context_returns_the_original_messages_when_they_fit() -> None:
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect the project"),
    )

    window = ContextManager(budget_chars=1_000).prepare(messages)

    assert window.messages == messages
    assert window.compacted is False
    assert window.budget_exceeded is False
    assert window.original_chars == conversation_char_count(messages)


def test_context_summarizes_old_groups_and_keeps_the_latest_group_complete() -> None:
    old_reasoning = "private old reasoning must not enter the summary"
    latest_reasoning = "latest protocol continuation"
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="repair the project"),
        *_exchange(
            1,
            name="create_file",
            path="src/new.py",
            content="created " + "a" * 240,
            reasoning=old_reasoning,
        ),
        *_exchange(
            2,
            name="read_file",
            path="src/old.py",
            content="observed " + "b" * 240,
        ),
        *_exchange(
            3,
            name="run_command",
            path="tests",
            content="latest result " + "c" * 240,
            reasoning=latest_reasoning,
        ),
    )

    window = ContextManager(budget_chars=700).prepare(messages)

    assert window.compacted is True
    assert window.omitted_message_count == 4
    assert window.prepared_chars <= 700
    assert window.messages[0].role is MessageRole.SYSTEM
    assert window.messages[1] == messages[1]
    assert "Earlier conversation summary" in (window.messages[0].content or "")
    assert "create_file(src/new.py)" in (window.messages[0].content or "")
    assert old_reasoning not in "".join(
        message.content or "" for message in window.messages
    )
    assert window.messages[-2].tool_calls[0].id == "call-3"
    assert window.messages[-2].reasoning_content == latest_reasoning
    assert window.messages[-1].tool_call_id == "call-3"


def test_context_accepts_a_replaceable_summary_strategy() -> None:
    class RecordingSummary:
        def __init__(self) -> None:
            self.received: tuple[Message, ...] = ()

        def summarize(
            self,
            messages: Sequence[Message],
            *,
            max_chars: int,
        ) -> str:
            self.received = tuple(messages)
            return "custom bounded summary"[:max_chars]

    strategy = RecordingSummary()
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect the project"),
        *_exchange(
            1,
            name="read_file",
            path="old.py",
            content="old " + "a" * 250,
        ),
        *_exchange(
            2,
            name="read_file",
            path="new.py",
            content="new " + "b" * 250,
        ),
    )

    window = ContextManager(
        budget_chars=550,
        summary_strategy=strategy,
    ).prepare(messages)

    assert len(strategy.received) == 2
    assert "custom bounded summary" in (window.messages[0].content or "")


def test_context_shortens_a_huge_latest_tool_result_as_valid_json() -> None:
    exchange = _exchange(
        1,
        name="read_file",
        path="large.log",
        content="begin\n" + "x" * 5_000 + "\nend",
        reasoning="required continuation state",
    )
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="read the log"),
        *exchange,
    )

    window = ContextManager(budget_chars=450).prepare(messages)

    assert window.shortened_message_count == 1
    assert window.prepared_chars <= 450
    assert window.messages[-2] == exchange[0]
    assert window.messages[-1].tool_call_id == "call-1"
    payload = json.loads(window.messages[-1].content or "")
    assert payload["ok"] is True
    assert "shortened for context budget" in payload["content"]
    assert payload["content"].startswith("begin")
    assert payload["content"].endswith("end")


def test_context_keeps_every_result_from_the_latest_multi_tool_turn() -> None:
    first_call = ToolCall(id="call-a", name="read_file", arguments_json="{}")
    second_call = ToolCall(id="call-b", name="search_text", arguments_json="{}")
    assistant = AssistantTurn(
        content=None,
        tool_calls=(first_call, second_call),
        reasoning_content="shared continuation state",
    ).as_message()
    first_result = ToolResult(
        call_id=first_call.id,
        tool_name=first_call.name,
        ok=True,
        content="a" * 2_000,
    ).as_message()
    second_result = ToolResult(
        call_id=second_call.id,
        tool_name=second_call.name,
        ok=True,
        content="b" * 2_000,
    ).as_message()
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect two things"),
        assistant,
        first_result,
        second_result,
    )

    window = ContextManager(budget_chars=550).prepare(messages)

    assert window.messages[-3].tool_calls == (first_call, second_call)
    assert window.messages[-3].reasoning_content == "shared continuation state"
    assert [message.tool_call_id for message in window.messages[-2:]] == [
        "call-a",
        "call-b",
    ]
    assert all(
        isinstance(json.loads(message.content or ""), dict)
        for message in window.messages[-2:]
    )
    assert window.prepared_chars <= 550


def test_context_reports_when_permanent_information_alone_exceeds_budget() -> None:
    messages = (
        Message(role=MessageRole.SYSTEM, content="s" * 200),
        Message(role=MessageRole.USER, content="u" * 200),
    )

    window = ContextManager(budget_chars=100).prepare(messages)

    assert window.messages == messages
    assert window.budget_exceeded is True
    assert window.compacted is False
