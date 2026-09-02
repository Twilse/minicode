import json
from datetime import datetime, timezone

from minicoder.application.event_bus import EventBus
from minicoder.application.memory import ModelLongTermMemoryMaintainer
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import AssistantTurn, Message, MessageRole, ToolCall
from minicoder.domain.state import AgentPhase, AgentRunResult, AgentStopReason
from tests.fakes import FakeModelAdapter, MemoryEventSink


def _result() -> AgentRunResult:
    return AgentRunResult(
        phase=AgentPhase.COMPLETE,
        stop_reason=AgentStopReason.FINAL_RESPONSE,
        model_steps=4,
        messages=(),
        final_response="Implemented the parser and passed four tests.",
    )


def _record(index: int, summary: str) -> ProjectMemoryRecord:
    return ProjectMemoryRecord(
        recorded_at=datetime(2026, 8, index, tzinfo=timezone.utc),
        summary=summary,
    )


def test_maintainer_selects_one_valuable_redacted_memory_and_sends_all_existing() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {
                        "memory_action": "append",
                        "memory_summary": (
                            "parser.py uses a stable compatibility check. secret-key"
                        ),
                    }
                )
            )
        ]
    )
    observed = MemoryEventSink()
    existing = tuple(_record(index, f"durable-memory-{index}") for index in range(1, 10))
    maintainer = ModelLongTermMemoryMaintainer(
        model=model,
        events=EventBus((observed,), run_id="memory"),
        sensitive_values=("secret-key",),
    )

    decision = maintainer.maintain(
        task="Add parser secret-key",
        result=_result(),
        turn_messages=(Message(role=MessageRole.USER, content="Add parser"),),
        project_memory=existing,
        model_step=4,
    )

    assert decision.memory_summary == (
        "parser.py uses a stable compatibility check. <redacted>"
    )
    source = model.requests[0].messages[-1].content or ""
    assert "secret-key" not in source
    assert all(record.summary in source for record in existing)
    assert "most calls should produce 'none'" in (
        model.requests[0].messages[0].content or ""
    )
    assert [event.kind for event in observed.events] == [
        AgentEventKind.MEMORY_SUMMARY_REQUESTED,
        AgentEventKind.MEMORY_SUMMARY_COMPLETED,
    ]


def test_maintainer_accepts_none_as_the_normal_decision() -> None:
    model = FakeModelAdapter(
        [AssistantTurn(content='{"memory_action":"none","memory_summary":null}')]
    )

    decision = ModelLongTermMemoryMaintainer(model=model).maintain(
        task="What time is it?",
        result=_result(),
        turn_messages=(),
        model_step=1,
    )

    assert decision.memory_summary is None
    assert decision.used_fallback is False


def test_maintainer_suppresses_a_duplicate_even_when_model_requests_append() -> None:
    summary = "The project requires Python 3.11 for all supported environments."
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {"memory_action": "append", "memory_summary": summary}
                )
            )
        ]
    )

    decision = ModelLongTermMemoryMaintainer(model=model).maintain(
        task="Repeat the Python requirement",
        result=_result(),
        turn_messages=(),
        project_memory=(_record(1, summary),),
        model_step=1,
    )

    assert decision.memory_summary is None


def test_maintainer_rejects_old_rolling_context_response_shape() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": "obsolete rolling context",
                        "memory_action": "none",
                        "memory_summary": None,
                    }
                )
            )
        ]
    )

    decision = ModelLongTermMemoryMaintainer(model=model).maintain(
        task="Temporary work",
        result=_result(),
        turn_messages=(),
        model_step=1,
    )

    assert decision.memory_summary is None
    assert decision.used_fallback is True


def test_maintainer_defaults_to_none_on_model_or_tool_protocol_failure() -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("offline")

    failed = ModelLongTermMemoryMaintainer(model=FailingModel()).maintain(
        task="Finish parser",
        result=_result(),
        turn_messages=(),
        model_step=4,
    )
    call = ToolCall(id="memory-call", name="read_file", arguments_json="{}")
    tool_model = FakeModelAdapter([AssistantTurn(content=None, tool_calls=(call,))])
    tool_failed = ModelLongTermMemoryMaintainer(model=tool_model).maintain(
        task="Inspect app.py",
        result=_result(),
        turn_messages=(),
        model_step=1,
    )

    assert failed.memory_summary is None and failed.used_fallback is True
    assert tool_failed.memory_summary is None and tool_failed.used_fallback is True
    assert tool_model.requests[0].tools == ()


def test_maintainer_skips_call_instead_of_truncating_existing_memory() -> None:
    model = FakeModelAdapter([])
    observed = MemoryEventSink()
    existing = (_record(1, "x" * 2_000),)
    maintainer = ModelLongTermMemoryMaintainer(
        model=model,
        events=EventBus((observed,), run_id="memory-budget"),
        request_input_budget_chars=500,
    )

    decision = maintainer.maintain(
        task="Do not lose old memory",
        result=_result(),
        turn_messages=(),
        project_memory=existing,
        model_step=1,
    )

    assert decision.memory_summary is None
    assert decision.used_fallback is True
    assert model.requests == []
    assert observed.events[-1].details["reason"] == (
        "memory_request_budget_too_small_for_all_records"
    )
