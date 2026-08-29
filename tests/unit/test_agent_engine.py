import json

import pytest

from minicoder.application.agent_engine import AgentEngine
from minicoder.application.event_bus import EventBus
from minicoder.domain.errors import ModelConnectionError
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import (
    AssistantTurn,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from minicoder.domain.state import AgentPhase, AgentStopReason
from tests.fakes import FakeModelAdapter, FakeToolAdapter, MemoryEventSink


def _definition(name: str = "read_file") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Execute {name} for a test.",
        parameters_schema={"type": "object", "properties": {}},
    )


def test_engine_completes_after_one_final_model_turn() -> None:
    model = FakeModelAdapter([AssistantTurn(content="Task completed.")])
    tools = FakeToolAdapter(definitions=(_definition(),))
    engine = AgentEngine(model=model, tools=tools, max_steps=3)

    result = engine.run("Inspect the project")

    assert result.phase is AgentPhase.COMPLETE
    assert result.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert result.model_steps == 1
    assert result.final_response == "Task completed."
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert model.requests[0].tools == (_definition(),)


def test_engine_returns_tool_results_and_reasoning_to_the_next_model_turn() -> None:
    call = ToolCall(
        id="call-read",
        name="read_file",
        arguments_json='{"path":"app.py"}',
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=None,
                tool_calls=(call,),
                reasoning_content="private continuation state",
            ),
            AssistantTurn(content="The file was inspected."),
        ]
    )
    tool_result = ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        content="print('hello')",
    )
    tools = FakeToolAdapter([tool_result], definitions=(_definition(),))
    engine = AgentEngine(model=model, tools=tools, max_steps=3)

    result = engine.run("Read app.py")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 2
    assert tools.calls == [call]
    second_request = model.requests[1]
    assert [message.role for message in second_request.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assistant_message = second_request.messages[2]
    assert assistant_message.tool_calls == (call,)
    assert assistant_message.reasoning_content == "private continuation state"
    tool_message = second_request.messages[3]
    assert tool_message.tool_call_id == call.id
    assert json.loads(tool_message.content or "")["ok"] is True


def test_engine_executes_every_tool_call_in_model_order() -> None:
    first_call = ToolCall(id="call-1", name="first", arguments_json="{}")
    second_call = ToolCall(id="call-2", name="second", arguments_json="{}")
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(first_call, second_call)),
            AssistantTurn(content="Both calls completed."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=first_call.id,
                tool_name=first_call.name,
                ok=True,
                content="first result",
            ),
            ToolResult(
                call_id=second_call.id,
                tool_name=second_call.name,
                ok=False,
                content="second failed safely",
                error_code="SAFE_FAILURE",
            ),
        ],
        definitions=(_definition("first"), _definition("second")),
    )
    engine = AgentEngine(model=model, tools=tools, max_steps=3)

    result = engine.run("Call both tools")

    assert result.phase is AgentPhase.COMPLETE
    assert tools.calls == [first_call, second_call]
    second_request = model.requests[1]
    assert [message.tool_call_id for message in second_request.messages[-2:]] == [
        first_call.id,
        second_call.id,
    ]
    assert json.loads(second_request.messages[-1].content or "")["ok"] is False


def test_engine_stops_after_max_model_steps_but_keeps_last_tool_results() -> None:
    first_call = ToolCall(id="call-1", name="read_file", arguments_json="{}")
    second_call = ToolCall(id="call-2", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(first_call,)),
            AssistantTurn(content=None, tool_calls=(second_call,)),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=first_call.id,
                tool_name=first_call.name,
                ok=True,
                content="first",
            ),
            ToolResult(
                call_id=second_call.id,
                tool_name=second_call.name,
                ok=True,
                content="second",
            ),
        ],
        definitions=(_definition(),),
    )
    engine = AgentEngine(model=model, tools=tools, max_steps=2)

    result = engine.run("Keep using tools")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert result.model_steps == 2
    assert result.final_response is None
    assert "maximum of 2" in (result.failure_message or "")
    assert result.messages[-1].role is MessageRole.TOOL
    assert result.messages[-1].tool_call_id == second_call.id


def test_engine_turns_expected_model_errors_into_a_failed_result() -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("network unavailable")

    engine = AgentEngine(
        model=FailingModel(),
        tools=FakeToolAdapter(),
        max_steps=3,
    )

    result = engine.run("Inspect the project")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert result.model_steps == 1
    assert "network unavailable" in (result.failure_message or "")
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


def test_engine_publishes_a_deterministic_model_tool_model_event_sequence() -> None:
    call = ToolCall(id="call-read", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(call,)),
            AssistantTurn(content="done"),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=call.id,
                tool_name=call.name,
                ok=True,
                content="file contents that are not copied into events",
            )
        ],
        definitions=(_definition(),),
    )
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=3,
        events=EventBus((events,), run_id="run-engine"),
    )

    result = engine.run("Read one file")

    assert result.phase is AgentPhase.COMPLETE
    assert [event.kind for event in events.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TASK_COMPLETED,
    ]
    assert [event.sequence for event in events.events] == [1, 2, 3, 4, 5, 6]
    assert events.events[3].details == {
        "call_id": "call-read",
        "tool_name": "read_file",
        "ok": True,
        "error_code": None,
        "content_chars": 45,
    }
    assert events.events[-1].details == {"response_chars": 4}


def test_engine_records_user_interruption_and_preserves_keyboard_interrupt() -> None:
    class InterruptingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise KeyboardInterrupt

    events = MemoryEventSink()
    engine = AgentEngine(
        model=InterruptingModel(),
        tools=FakeToolAdapter(),
        max_steps=3,
        events=EventBus((events,), run_id="run-interrupt"),
    )

    with pytest.raises(KeyboardInterrupt):
        engine.run("Wait for the model")

    assert [event.kind for event in events.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TASK_FAILED,
    ]
    assert events.events[-1].details["reason"] == "user_interrupted"
