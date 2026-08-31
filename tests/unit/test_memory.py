import json
from datetime import datetime, timezone

from minicoder.application.context import conversation_char_count
from minicoder.application.event_bus import EventBus
from minicoder.application.memory import ModelTurnMemoryMaintainer
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from minicoder.domain.state import (
    AgentPhase,
    AgentRunResult,
    AgentStopReason,
)
from tests.fakes import FakeModelAdapter, MemoryEventSink


def _result(
    *,
    phase: AgentPhase = AgentPhase.COMPLETE,
    messages: tuple[Message, ...] = (),
) -> AgentRunResult:
    if phase is AgentPhase.COMPLETE:
        return AgentRunResult(
            phase=phase,
            stop_reason=AgentStopReason.FINAL_RESPONSE,
            model_steps=4,
            messages=messages,
            final_response="Implemented the parser and passed four tests.",
        )
    return AgentRunResult(
        phase=phase,
        stop_reason=AgentStopReason.MAX_STEPS,
        model_steps=40,
        messages=messages,
        failure_message=(
            "Reached the model limit; parser.py changed but tests were not run."
        ),
    )


def test_maintainer_updates_context_and_selects_durable_memory() -> None:
    decision_json = json.dumps(
        {
            "context_summary": (
                "Goal: add parser. Completed: parser.py. Verification: 4 tests passed."
            ),
            "memory_action": "append",
            "memory_summary": (
                "parser.py now validates compatibility; four tests passed. secret-key"
            ),
        }
    )
    model = FakeModelAdapter([AssistantTurn(content=decision_json)])
    observed = MemoryEventSink()
    call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"path":"parser.py"}',
    )
    messages = (
        Message(role=MessageRole.USER, content="Add parser secret-key"),
        AssistantTurn(
            content=None,
            tool_calls=(call,),
            reasoning_content="private provider reasoning",
        ).as_message(),
        ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="parser source",
        ).as_message(),
        Message(role=MessageRole.ASSISTANT, content="done"),
    )
    maintainer = ModelTurnMemoryMaintainer(
        model=model,
        events=EventBus((observed,), run_id="maintenance"),
        sensitive_values=("secret-key",),
    )
    existing_memory = ProjectMemoryRecord(
        recorded_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        summary="The parser already uses Python 3.11.",
    )

    decision = maintainer.maintain(
        task="Add parser secret-key",
        result=_result(messages=messages),
        turn_messages=messages,
        previous_context="Earlier project context.",
        project_memory=(existing_memory,),
        model_step=4,
    )

    assert "4 tests passed" in decision.context_summary
    assert decision.memory_summary == (
        "parser.py now validates compatibility; four tests passed. <redacted>"
    )
    assert decision.used_fallback is False
    assert len(model.requests) == 1
    assert model.requests[0].tools == ()
    source = model.requests[0].messages[-1].content or ""
    assert "secret-key" not in source
    assert "read_file" in source
    assert "parser source" in source
    assert "The parser already uses Python 3.11" in source
    assert "private provider reasoning" not in source
    assert [event.kind for event in observed.events] == [
        AgentEventKind.MEMORY_SUMMARY_REQUESTED,
        AgentEventKind.MEMORY_SUMMARY_COMPLETED,
    ]
    assert observed.events[-1].details["memory_selected"] is True


def test_maintainer_allows_the_model_to_skip_long_term_memory() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": "The user asked a transient question.",
                        "memory_action": "none",
                        "memory_summary": None,
                    }
                )
            )
        ]
    )
    maintainer = ModelTurnMemoryMaintainer(model=model)

    decision = maintainer.maintain(
        task="What time is it?",
        result=_result(),
        turn_messages=(),
        previous_context=None,
        model_step=1,
    )

    assert decision.context_summary == "The user asked a transient question."
    assert decision.memory_summary is None


def test_maintainer_forces_none_when_long_term_memory_is_disabled() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": "Keep this only in rolling context.",
                        "memory_action": "append",
                        "memory_summary": "Model tried to persist this.",
                    }
                )
            )
        ]
    )
    maintainer = ModelTurnMemoryMaintainer(
        model=model,
        allow_long_term_memory=False,
    )

    decision = maintainer.maintain(
        task="Temporary work",
        result=_result(),
        turn_messages=(),
        previous_context=None,
        model_step=1,
    )

    assert decision.context_summary == "Keep this only in rolling context."
    assert decision.memory_summary is None


def test_maintainer_fallback_preserves_failed_turn_status() -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("offline")

    observed = MemoryEventSink()
    maintainer = ModelTurnMemoryMaintainer(
        model=FailingModel(),
        events=EventBus((observed,), run_id="maintenance-fallback"),
    )

    decision = maintainer.maintain(
        task="Finish the parser",
        result=_result(phase=AgentPhase.FAILED),
        turn_messages=(),
        previous_context="Parser implementation is partial.",
        model_step=40,
    )

    assert "Parser implementation is partial" in decision.context_summary
    assert "stop reason: max_steps" in decision.context_summary
    assert "tests were not run" in decision.context_summary
    assert decision.memory_summary is None
    assert decision.used_fallback is True
    assert observed.events[-1].kind is AgentEventKind.MEMORY_SUMMARY_FAILED
    assert observed.events[-1].details["reason"] == "model_error"


def test_maintainer_rejects_an_unexpected_tool_call() -> None:
    call = ToolCall(id="memory-call", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [AssistantTurn(content=None, tool_calls=(call,))]
    )
    maintainer = ModelTurnMemoryMaintainer(model=model)

    decision = maintainer.maintain(
        task="Inspect app.py",
        result=_result(),
        turn_messages=(),
        previous_context=None,
        model_step=1,
    )

    assert decision.used_fallback is True
    assert decision.memory_summary is None
    assert model.requests[0].tools == ()


def test_maintainer_never_sends_more_than_its_input_budget() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": "Bounded rolling context.",
                        "memory_action": "none",
                        "memory_summary": None,
                    }
                )
            )
        ]
    )
    maintainer = ModelTurnMemoryMaintainer(
        model=model,
        transcript_input_chars=20_000,
        request_input_budget_chars=1_800,
    )
    messages = (
        Message(role=MessageRole.USER, content="large transcript " + "x" * 20_000),
    )

    decision = maintainer.maintain(
        task="Keep the request bounded",
        result=_result(messages=messages),
        turn_messages=messages,
        previous_context="previous " + "y" * 10_000,
        model_step=4,
    )

    assert decision.context_summary == "Bounded rolling context."
    assert conversation_char_count(model.requests[0].messages) <= 1_800


def test_maintainer_uses_fallback_when_fixed_prompt_cannot_fit() -> None:
    model = FakeModelAdapter([])
    observed = MemoryEventSink()
    maintainer = ModelTurnMemoryMaintainer(
        model=model,
        events=EventBus((observed,), run_id="maintenance-budget-fallback"),
        request_input_budget_chars=10,
    )

    decision = maintainer.maintain(
        task="Remember unfinished work",
        result=_result(phase=AgentPhase.FAILED),
        turn_messages=(),
        previous_context="Work is still unfinished.",
        model_step=40,
    )

    assert decision.used_fallback is True
    assert "unfinished" in decision.context_summary
    assert model.requests == []
    assert observed.events[-1].kind is AgentEventKind.MEMORY_SUMMARY_FAILED
    assert (
        observed.events[-1].details["reason"]
        == "maintenance_request_budget_too_small"
    )
