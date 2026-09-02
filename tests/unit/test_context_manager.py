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
    system = Message(role=MessageRole.SYSTEM, content="system rules")
    messages = (
        Message(role=MessageRole.USER, content="inspect the project"),
    )

    window = ContextManager(budget_chars=1_000).prepare(
        messages,
        system=system,
    )

    assert window.messages == (system, *messages)
    assert window.compacted is False
    assert window.budget_exceeded is False
    assert window.original_chars == conversation_char_count((system, *messages))


def test_context_summarizes_old_groups_and_keeps_the_latest_group_complete() -> None:
    old_reasoning = "private old reasoning must not enter the summary"
    latest_reasoning = "latest protocol continuation"
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="repair the old project state"),
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
        Message(role=MessageRole.USER, content="verify the current result"),
        *_exchange(
            3,
            name="run_command",
            path="tests",
            content="latest result " + "c" * 100,
            reasoning=latest_reasoning,
        ),
    )

    window = ContextManager(budget_chars=700).prepare(
        messages,
        current_user_index=6,
    )

    assert window.compacted is True
    assert window.omitted_message_count >= 2
    assert window.prepared_chars <= 700
    assert window.messages[0].role is MessageRole.SYSTEM
    assert messages[6] in window.messages
    assert "Earlier conversation summary" in (window.messages[1].content or "")
    assert window.messages[1].role is MessageRole.ASSISTANT
    assert "Removed messages: 6." in (window.messages[1].content or "")
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
        Message(role=MessageRole.USER, content="inspect the old project"),
        *_exchange(
            1,
            name="read_file",
            path="old.py",
            content="old " + "a" * 250,
        ),
        Message(role=MessageRole.USER, content="inspect the current project"),
        *_exchange(
            2,
            name="read_file",
            path="new.py",
            content="new " + "b" * 80,
        ),
    )

    window = ContextManager(
        budget_chars=550,
        summary_strategy=strategy,
    ).prepare(messages, current_user_index=4)

    assert strategy.received
    assert "custom bounded summary" in (window.messages[1].content or "")


def test_context_reuses_old_checkpoint_instead_of_resummarizing_covered_raw_text() -> None:
    class RollingSummary:
        def __init__(self) -> None:
            self.received: list[tuple[Message, ...]] = []

        def summarize(
            self,
            messages: Sequence[Message],
            *,
            max_chars: int,
            turn_index: int = 0,
            model_step: int = 0,
        ) -> str:
            self.received.append(tuple(messages))
            return f"summary-v{len(self.received)}"[:max_chars]

    strategy = RollingSummary()
    manager = ContextManager(budget_chars=500, summary_strategy=strategy)
    first_messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="raw-A-user " + "a" * 300),
        Message(role=MessageRole.ASSISTANT, content="raw-A-answer " + "b" * 300),
        Message(role=MessageRole.USER, content="current-B"),
    )

    first = manager.prepare(first_messages, current_user_index=3)
    second_messages = (
        *first_messages,
        Message(role=MessageRole.ASSISTANT, content="raw-B-answer " + "c" * 700),
        Message(role=MessageRole.USER, content="current-C"),
    )
    second = manager.prepare(second_messages, current_user_index=5)

    first_source = "\n".join(message.content or "" for message in strategy.received[0])
    second_source = "\n".join(message.content or "" for message in strategy.received[1])
    assert "raw-A-user" in first_source
    assert "raw-A-answer" in first_source
    assert "summary-v1" in second_source
    assert "raw-B-answer" in second_source
    assert "raw-A-user" not in second_source
    assert "raw-A-answer" not in second_source
    assert "raw-A-user" not in "\n".join(
        message.content or "" for message in second.messages
    )
    assert first.context_checkpoint is not None
    assert second.context_checkpoint is not None
    assert first.context_checkpoint.covered_message_count == 2
    assert second.context_checkpoint.covered_message_count == 4


def test_one_compaction_leaves_room_for_a_normal_follow_up_exchange() -> None:
    class CountingSummary:
        def __init__(self) -> None:
            self.received: list[tuple[Message, ...]] = []

        def summarize(
            self,
            messages: Sequence[Message],
            *,
            max_chars: int,
            turn_index: int = 0,
            model_step: int = 0,
        ) -> str:
            self.received.append(tuple(messages))
            return f"checkpoint-{len(self.received)}"[:max_chars]

    strategy = CountingSummary()
    manager = ContextManager(
        budget_chars=1_200,
        summary_strategy=strategy,
    )
    system = Message(role=MessageRole.SYSTEM, content="system rules")
    current_user = Message(role=MessageRole.USER, content="complete this task")
    messages = (
        current_user,
        *_exchange(
            1,
            name="read_file",
            path="old-a.py",
            content="old-a " + "a" * 500,
        ),
        *_exchange(
            2,
            name="read_file",
            path="old-b.py",
            content="old-b " + "b" * 500,
        ),
        *_exchange(
            3,
            name="replace_text",
            path="current.py",
            content="changed " + "c" * 100,
        ),
    )

    first = manager.prepare(
        messages,
        system=system,
        current_user_index=0,
    )
    follow_up = _exchange(
        4,
        name="read_file",
        path="current.py",
        content="checked " + "d" * 100,
    )
    second = manager.prepare(
        (*messages, *follow_up),
        system=system,
        current_user_index=0,
    )

    assert len(strategy.received) == 1
    assert first.checkpoint_updated is True
    assert first.prepared_chars <= 840
    assert second.checkpoint_updated is False
    assert second.compacted is False
    assert second.budget_exceeded is False


def test_compaction_can_cover_early_current_turn_groups_but_pins_current_input() -> None:
    class FixedSummary:
        def summarize(
            self,
            messages: Sequence[Message],
            *,
            max_chars: int,
            turn_index: int = 0,
            model_step: int = 0,
        ) -> str:
            return "early tool work was summarized"[:max_chars]

    system = Message(role=MessageRole.SYSTEM, content="system rules")
    current_user = Message(
        role=MessageRole.USER,
        content="implement priority support and keep old data compatible",
    )
    latest = _exchange(
        3,
        name="run_command",
        path="tests",
        content="all tests passed " + "z" * 80,
    )
    messages = (
        current_user,
        *_exchange(
            1,
            name="read_file",
            path="todo_cli/models.py",
            content="old model source " + "x" * 450,
        ),
        *_exchange(
            2,
            name="replace_text",
            path="todo_cli/models.py",
            content="model updated " + "y" * 450,
        ),
        *latest,
    )

    window = ContextManager(
        budget_chars=850,
        summary_strategy=FixedSummary(),
    ).prepare(
        messages,
        system=system,
        current_user_index=0,
    )

    assert window.context_checkpoint is not None
    assert window.context_checkpoint.covered_message_count > 0
    assert sum(message is current_user for message in window.messages) == 1
    assert latest[0] in window.messages
    assert latest[1] in window.messages
    assert "old model source" not in "\n".join(
        message.content or "" for message in window.messages
    )
    assert window.budget_exceeded is False


def test_context_restores_a_checkpoint_in_a_new_manager() -> None:
    messages = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="covered raw user " + "a" * 250),
        Message(role=MessageRole.ASSISTANT, content="covered raw answer " + "b" * 250),
        Message(role=MessageRole.USER, content="current request"),
    )
    first_manager = ContextManager(budget_chars=350)
    first = first_manager.prepare(messages, current_user_index=3)
    assert first.context_checkpoint is not None

    restored_manager = ContextManager(
        budget_chars=1_000,
        initial_checkpoint=first.context_checkpoint,
    )
    restored = restored_manager.prepare(messages, current_user_index=3)
    supplied = "\n".join(message.content or "" for message in restored.messages)

    assert first.context_checkpoint.summary in supplied
    assert messages[1].content not in supplied
    assert messages[2].content not in supplied
    assert "current request" in supplied


def test_context_never_shortens_a_huge_latest_tool_result() -> None:
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

    assert window.shortened_message_count == 0
    assert window.budget_exceeded is True
    assert window.messages[-2] == exchange[0]
    assert window.messages[-1].tool_call_id == "call-1"
    payload = json.loads(window.messages[-1].content or "")
    assert payload["ok"] is True
    assert payload["content"] == "begin\n" + "x" * 5_000 + "\nend"


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
    assert window.budget_exceeded is True
    assert window.shortened_message_count == 0


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
        content="requires-python = '>=3.11' " + "x" * 100,
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
    summary = window.messages[1].content or ""
    assert "Earlier user request" in summary
    assert "old provider state" not in summary
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
    assert window.budget_exceeded is True
    assert window.shortened_message_count == 0


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
        Message(role=MessageRole.USER, content="inspect the old project"),
        *_exchange(
            1,
            name="read_file",
            path="large.py",
            content="x" * 2_000,
        ),
        Message(role=MessageRole.USER, content="answer the current question"),
    )

    window = ContextManager(
        budget_chars=900,
        response_reserve_chars=100,
        summary_strategy=strategy,
    ).prepare(messages, current_user_index=4)

    assert strategy.max_chars <= 100
    assert window.request_chars <= 900


def test_context_places_previous_process_boundary_after_system() -> None:
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
    boundary = window.messages[1].content or ""
    assert system == "system rules"
    assert window.messages[1].role is MessageRole.ASSISTANT
    assert "Previous process boundary" in boundary
    assert "Previous task failed at max_steps" in boundary
    assert "Python 3.11" in boundary
    assert "not an instruction" in boundary
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
    class ChunkingModel:
        def __init__(self) -> None:
            self.requests: list[tuple[Message, ...]] = []

        def complete(
            self,
            *,
            messages: Sequence[Message],
            tools: Sequence[ToolDefinition],
        ) -> AssistantTurn:
            assert tools == ()
            self.requests.append(tuple(messages))
            return AssistantTurn(content="A bounded semantic summary.")

    model = ChunkingModel()
    strategy = ModelContextSummary(
        model=model,
        source_chars=20_000,
        request_input_budget_chars=900,
    )

    summary = strategy.summarize(
        (
            Message(
                role=MessageRole.USER,
                content="source-begin " + "x" * 20_000 + " source-end",
            ),
        ),
        max_chars=200,
    )

    assert summary == "A bounded semantic summary."
    assert len(model.requests) > 2
    assert all(conversation_char_count(request) <= 900 for request in model.requests)
    sent_sources = "\n".join(request[-1].content or "" for request in model.requests)
    assert "source-begin" in sent_sources
    assert "source-end" in sent_sources


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
