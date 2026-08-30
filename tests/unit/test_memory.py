from minicoder.application.event_bus import EventBus
from minicoder.application.memory import ModelMemorySummarizer
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import AssistantTurn, MessageRole, ToolCall
from tests.fakes import FakeModelAdapter, MemoryEventSink


def test_model_memory_summarizer_uses_one_bounded_no_tool_request() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=(
                    "Updated app.py, preserved compatibility, and passed tests. "
                    "secret-key"
                )
            )
        ]
    )
    observed = MemoryEventSink()
    summarizer = ModelMemorySummarizer(
        model=model,
        events=EventBus((observed,), run_id="memory-summary"),
        sensitive_values=("secret-key",),
        task_input_chars=60,
        outcome_input_chars=70,
        summary_chars=90,
    )

    summary = summarizer.summarize(
        task="task-start " + "t" * 100 + " task-end secret-key",
        outcome="outcome-start " + "o" * 100 + " outcome-end",
        model_step=4,
    )

    assert summary == (
        "Updated app.py, preserved compatibility, and passed tests. <redacted>"
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.tools == ()
    assert [message.role for message in request.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    source = request.messages[1].content or ""
    assert "secret-key" not in source
    assert "task-start" in source and "redacted" in source
    assert "outcome-start" in source and "outcome-end" in source
    assert [event.kind for event in observed.events] == [
        AgentEventKind.MEMORY_SUMMARY_REQUESTED,
        AgentEventKind.MEMORY_SUMMARY_COMPLETED,
    ]
    assert all(event.model_step == 4 for event in observed.events)


def test_model_memory_summarizer_falls_back_after_model_error() -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("offline")

    observed = MemoryEventSink()
    summarizer = ModelMemorySummarizer(
        model=FailingModel(),
        events=EventBus((observed,), run_id="memory-fallback"),
    )

    summary = summarizer.summarize(
        task="Add a parser",
        outcome="Parser added and tests passed",
        model_step=2,
    )

    assert summary == (
        "User goal: Add a parser\n"
        "Completed outcome: Parser added and tests passed"
    )
    assert [event.kind for event in observed.events] == [
        AgentEventKind.MEMORY_SUMMARY_REQUESTED,
        AgentEventKind.MEMORY_SUMMARY_FAILED,
    ]
    assert observed.events[-1].details["reason"] == "model_error"
    assert observed.events[-1].details["error_type"] == "ModelConnectionError"


def test_model_memory_summarizer_does_not_execute_unexpected_tool_call() -> None:
    call = ToolCall(id="memory-call", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [AssistantTurn(content=None, tool_calls=(call,))]
    )
    observed = MemoryEventSink()
    summarizer = ModelMemorySummarizer(
        model=model,
        events=EventBus((observed,), run_id="memory-invalid"),
    )

    summary = summarizer.summarize(
        task="Inspect app.py",
        outcome="No changes were needed",
        model_step=1,
    )

    assert summary.startswith("User goal: Inspect app.py")
    assert model.requests[0].tools == ()
    assert observed.events[-1].kind is AgentEventKind.MEMORY_SUMMARY_FAILED
    assert observed.events[-1].details["reason"] == "invalid_summary_response"
