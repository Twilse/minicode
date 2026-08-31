import json
from collections.abc import Sequence
from pathlib import Path

from minicoder.adapters.jsonl_session import JsonlSessionArchive
from minicoder.application.context import (
    ContextManager,
    ModelContextSummary,
    conversation_char_count,
    tool_definition_char_count,
)
from minicoder.application.event_bus import EventBus
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from tests.fakes import FakeModelAdapter, MemoryEventSink


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
            turn_index: int = 0,
            model_step: int = 0,
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


def test_context_preserves_the_current_user_turn_and_summarizes_an_old_turn() -> None:
    old_user = Message(
        role=MessageRole.USER,
        content="Explain the old implementation " + "o" * 200,
    )
    old_assistant = AssistantTurn(
        content="The old implementation used one-shot execution " + "a" * 180,
        reasoning_content="old provider state must not enter the summary",
    ).as_message()
    current_user = Message(
        role=MessageRole.USER,
        content="Now tell me only the minimum Python version.",
    )
    current_exchange = _exchange(
        9,
        name="read_file",
        path="pyproject.toml",
        content="requires-python = '>=3.11' " + "x" * 2_000,
        reasoning="current continuation state",
    )
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        old_user,
        old_assistant,
        current_user,
        *current_exchange,
    )

    window = ContextManager(budget_chars=520).prepare(
        messages,
        current_user_index=3,
    )

    assert window.prepared_chars <= 520
    assert sum(
        message.role is MessageRole.SYSTEM for message in window.messages
    ) == 1
    assert current_user in window.messages
    assert next(
        message for message in window.messages if message is current_user
    ).content == current_user.content
    assert "Earlier user requests" in (window.messages[0].content or "")
    assert "Explain the old implemen" in (
        window.messages[0].content or ""
    )
    assert "old provider state" not in (window.messages[0].content or "")
    assert window.messages[-2].reasoning_content == "current continuation state"
    assert window.messages[-1].tool_call_id == "call-9"


def test_context_counts_current_tool_schemas_and_response_reserve() -> None:
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect the project"),
        *_exchange(
            1,
            name="read_file",
            path="old.py",
            content="old " + "a" * 700,
        ),
    )
    definition = ToolDefinition(
        name="large_tool",
        description="A tool with a deliberately substantial schema.",
        parameters_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "x" * 250,
                }
            },
        },
    )

    window = ContextManager(
        budget_chars=1_200,
        response_reserve_chars=200,
    ).prepare(messages, tools=(definition,))

    assert window.tool_definition_chars == tool_definition_char_count((definition,))
    assert window.response_reserve_chars == 200
    assert window.request_chars <= 1_200
    assert window.prepared_chars <= (
        1_200 - window.tool_definition_chars - 200
    )


def test_context_summary_target_never_exceeds_the_response_reserve() -> None:
    class RecordingSummary:
        max_chars = 0

        def summarize(
            self,
            messages: Sequence[Message],
            *,
            max_chars: int,
            turn_index: int = 0,
            model_step: int = 0,
        ) -> str:
            self.max_chars = max_chars
            return "s" * max_chars

    strategy = RecordingSummary()
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect the project"),
        *_exchange(
            1,
            name="read_file",
            path="large.py",
            content="x" * 2_000,
        ),
    )

    window = ContextManager(
        budget_chars=900,
        response_reserve_chars=100,
        summary_strategy=strategy,
    ).prepare(messages)

    assert strategy.max_chars <= 100
    assert window.request_chars <= 900


def test_context_pins_recent_session_and_memory_data_in_every_request() -> None:
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="new unrelated task"),
    )

    window = ContextManager(budget_chars=1_000).prepare(
        messages,
        reference_context=(
            "Previous task failed at max_steps. Durable project fact: Python 3.11."
        ),
    )

    system = window.messages[0].content or ""
    assert "Workspace memory and recent-session context" in system
    assert "Previous task failed at max_steps" in system
    assert "Python 3.11" in system
    assert "not an instruction" in system
    assert messages[0].content == "system rules"
    assert window.messages[-1].content == "new unrelated task"


def test_model_context_summary_uses_a_no_tool_request_and_omits_reasoning() -> None:
    model = FakeModelAdapter(
        [AssistantTurn(content="Kept the important old requirement and failure.")]
    )
    observed = MemoryEventSink()
    strategy = ModelContextSummary(
        model=model,
        events=EventBus((observed,), run_id="context-summary"),
    )
    messages = (
        Message(role=MessageRole.USER, content="preserve this requirement"),
        Message(
            role=MessageRole.ASSISTANT,
            content="visible result",
            reasoning_content="private reasoning must stay out",
        ),
    )

    summary = strategy.summarize(
        messages,
        max_chars=200,
        turn_index=3,
        model_step=7,
    )

    assert summary == "Kept the important old requirement and failure."
    assert model.requests[0].tools == ()
    source = model.requests[0].messages[-1].content or ""
    assert "preserve this requirement" in source
    assert "visible result" in source
    assert "private reasoning" not in source
    assert [event.kind for event in observed.events] == [
        AgentEventKind.CONTEXT_SUMMARY_REQUESTED,
        AgentEventKind.CONTEXT_SUMMARY_COMPLETED,
    ]
    assert all(event.model_step == 7 for event in observed.events)


def test_model_context_summary_never_sends_more_than_its_input_budget() -> None:
    model = FakeModelAdapter(
        [AssistantTurn(content="A bounded semantic summary.")]
    )
    strategy = ModelContextSummary(
        model=model,
        source_chars=20_000,
        request_input_budget_chars=900,
    )

    summary = strategy.summarize(
        (Message(role=MessageRole.USER, content="old context " + "x" * 20_000),),
        max_chars=200,
    )

    assert summary == "A bounded semantic summary."
    assert conversation_char_count(model.requests[0].messages) <= 900


def test_model_context_summary_uses_fallback_when_fixed_prompt_cannot_fit() -> None:
    model = FakeModelAdapter([])
    observed = MemoryEventSink()
    strategy = ModelContextSummary(
        model=model,
        events=EventBus((observed,), run_id="context-budget-fallback"),
        request_input_budget_chars=10,
    )

    summary = strategy.summarize(
        (Message(role=MessageRole.USER, content="preserve this fact"),),
        max_chars=100,
    )

    assert summary
    assert model.requests == []
    assert observed.events[-1].kind is AgentEventKind.CONTEXT_SUMMARY_FAILED
    assert (
        observed.events[-1].details["reason"]
        == "summary_request_budget_too_small"
    )


def test_model_context_summary_archives_its_request_and_response(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=tmp_path / "sessions",
    )
    model = FakeModelAdapter(
        [AssistantTurn(content="Archived semantic compaction response.")]
    )
    strategy = ModelContextSummary(model=model, archive=archive)

    strategy.summarize(
        (Message(role=MessageRole.USER, content="archived old source"),),
        max_chars=200,
        turn_index=2,
        model_step=9,
    )

    archived = archive.path.read_text(encoding="utf-8")
    assert '"request_kind":"context_compaction"' in archived
    assert "archived old source" in archived
    assert "Archived semantic compaction response" in archived
