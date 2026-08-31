import json
from datetime import datetime, timezone

import pytest

from minicoder.application.agent_engine import AgentEngine
from minicoder.application.context import ContextManager, conversation_char_count
from minicoder.application.event_bus import EventBus
from minicoder.application.retry import ExponentialBackoffRetryStrategy
from minicoder.domain.errors import (
    DomainValidationError,
    ModelConnectionError,
    ModelServiceError,
)
from minicoder.domain.events import AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.domain.models import (
    AssistantTurn,
    Message,
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
    system_prompt = model.requests[0].messages[0].content or ""
    assert "recognized test or compiler" in system_prompt
    assert "direct application runs are general" in system_prompt


def test_engine_plans_without_tools_before_following_the_plan() -> None:
    call = ToolCall(id="call-read", name="read_file", arguments_json="{}")
    plan = (
        "1. Inspect the relevant file.\n"
        "2. Make the requested change.\n"
        "3. Run verification."
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=plan),
            AssistantTurn(content=None, tool_calls=(call,)),
            AssistantTurn(content="Completed according to the adjusted plan."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                content="file was not found",
                error_code="FILE_NOT_FOUND",
            )
        ],
        definitions=(_definition(),),
    )
    observed = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=3,
        planning_enabled=True,
        events=EventBus((observed,), run_id="run-planning"),
    )

    result = engine.run("Update the parser")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 3
    assert model.requests[0].tools == ()
    assert model.requests[1].tools == (_definition(),)
    assert [message.role for message in model.requests[1].messages[-2:]] == [
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert model.requests[1].messages[-2].content == plan
    execution_requirement = model.requests[1].messages[-1].content or ""
    assert "default execution contract" in execution_requirement
    assert "adapt the remaining steps" in execution_requirement
    assert "[plan_step=N]" in execution_requirement
    assert "never in the final answer" in execution_requirement
    planning_requirement = model.requests[0].messages[-1].content or ""
    assert "exactly 1 step for a direct answer" in planning_requirement
    assert "not an outline of the final answer" in planning_requirement
    third_request = model.requests[2].messages
    assert third_request[-1].role is MessageRole.TOOL
    assert "FILE_NOT_FOUND" in (third_request[-1].content or "")
    assert tools.calls == [call]
    assert [event.kind for event in observed.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.PLANNING_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.PLANNING_COMPLETED,
        AgentEventKind.PLAN_STEP_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.PLAN_STEP_COMPLETED,
        AgentEventKind.PLAN_STEPS_UNTRACKED,
        AgentEventKind.PLAN_COMPLETED,
        AgentEventKind.TASK_COMPLETED,
    ]
    assert observed.events[2].details["request_kind"] == "planning"
    assert observed.events[3].details["plan_item_count"] == 3
    assert observed.events[3].details["display_plan"] == plan
    assert observed.events[4].details["plan_step"] == 1
    assert observed.events[5].details["request_kind"] == "execution"
    assert [
        (event.details["plan_step"], event.kind)
        for event in observed.events[9:10]
    ] == [
        (1, AgentEventKind.PLAN_STEP_COMPLETED),
    ]
    assert observed.events[10].details == {
        "first_plan_step": 2,
        "last_plan_step": 3,
        "untracked_plan_item_count": 2,
        "plan_item_count": 3,
    }
    assert observed.events[11].details["untracked_plan_item_count"] == 2


def test_engine_removes_reserved_annotations_from_any_final_response() -> None:
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=(
                    "[plan_step=1] I remember the todo project.\n\n"
                    "[plan_step=2] No files were changed."
                )
            )
        ]
    )
    engine = AgentEngine(model=model, tools=FakeToolAdapter(), max_steps=1)

    result = engine.run("Summarize the remembered project")

    assert result.final_response == (
        "I remember the todo project.\n\nNo files were changed."
    )
    assert "[plan_step=" not in (result.messages[-1].content or "")


def test_engine_rejects_a_final_response_containing_only_host_annotations() -> None:
    model = FakeModelAdapter([AssistantTurn(content="[plan_step=1]")])
    engine = AgentEngine(model=model, tools=FakeToolAdapter(), max_steps=1)

    result = engine.run("Answer the question")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.MODEL_ERROR
    assert result.final_response is None
    assert "reserved host annotations" in (result.failure_message or "")


def test_engine_returns_out_of_order_tool_calls_to_the_model_without_execution() -> None:
    call = ToolCall(
        id="call-too-early",
        name="run_command",
        arguments_json='{"argv":["python","-m","pytest"]}',
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=(
                    "1. Inspect files.\n"
                    "2. Implement the change.\n"
                    "3. Verify with pytest."
                )
            ),
            AssistantTurn(
                content="[plan_step=3]",
                tool_calls=(call,),
            ),
            AssistantTurn(content="I must inspect the files first."),
        ]
    )
    tools = FakeToolAdapter(definitions=(_definition("run_command"),))
    observed = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=3,
        planning_enabled=True,
        events=EventBus((observed,), run_id="run-strict-plan"),
    )

    result = engine.run("Update and verify the project")

    assert result.phase is AgentPhase.COMPLETE
    assert tools.calls == []
    feedback = model.requests[2].messages[-1]
    assert feedback.role is MessageRole.TOOL
    assert "PLAN_STEP_OUT_OF_ORDER" in (feedback.content or "")
    rejection = next(
        event
        for event in observed.events
        if event.kind is AgentEventKind.PLAN_TOOL_REJECTED
    )
    assert rejection.details["expected_plan_step"] == 1
    assert rejection.details["attempted_plan_step"] == 3
    assert AgentEventKind.TOOL_CALLED not in {
        event.kind for event in observed.events
    }


def test_engine_counts_planning_against_the_model_step_limit() -> None:
    model = FakeModelAdapter([AssistantTurn(content="1. Inspect the project.")])
    tools = FakeToolAdapter(definitions=(_definition(),))
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=1,
        planning_enabled=True,
    )

    result = engine.run("Inspect the project")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert result.model_steps == 1
    assert tools.calls == []
    assert len(model.requests) == 1
    assert model.requests[0].tools == ()


def test_engine_rejects_a_tool_call_from_the_planning_phase() -> None:
    call = ToolCall(id="call-plan", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [AssistantTurn(content=None, tool_calls=(call,))]
    )
    tools = FakeToolAdapter(definitions=(_definition(),))
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=3,
        planning_enabled=True,
    )

    result = engine.run("Read a file")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.PLANNING_ERROR
    assert result.model_steps == 1
    assert tools.calls == []
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


def test_engine_runs_multiple_user_turns_with_shared_history_and_fresh_steps() -> None:
    first_call = ToolCall(id="call-first", name="read_file", arguments_json="{}")
    second_call = ToolCall(id="call-second", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(first_call,)),
            AssistantTurn(
                content="The file uses Python.",
                reasoning_content="first turn continuation state",
            ),
            AssistantTurn(content=None, tool_calls=(second_call,)),
            AssistantTurn(content="It requires Python 3.11."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=first_call.id,
                tool_name=first_call.name,
                ok=True,
                content="requires-python = '>=3.11'",
            ),
            ToolResult(
                call_id=second_call.id,
                tool_name=second_call.name,
                ok=True,
                content="requires-python = '>=3.11'",
            ),
        ],
        definitions=(_definition(),),
    )
    engine = AgentEngine(model=model, tools=tools, max_steps=2)

    first = engine.run_turn("Which language does this project use?")
    second = engine.run_turn(
        "What is its minimum version?",
        history=first.messages,
    )

    assert first.model_steps == 2
    assert second.model_steps == 2
    assert second.final_response == "It requires Python 3.11."
    second_turn_request = model.requests[2].messages
    assert [message.role for message in second_turn_request] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.USER,
    ]
    assert second_turn_request[-2].reasoning_content == (
        "first turn continuation state"
    )
    assert second_turn_request[-1].content == "What is its minimum version?"
    assert sum(
        message.role is MessageRole.SYSTEM for message in second.messages
    ) == 1


def test_engine_injects_bounded_project_memory_only_into_a_fresh_turn() -> None:
    memory = ProjectMemoryRecord(
        recorded_at=datetime(2026, 8, 30, tzinfo=timezone.utc),
        summary=(
            "The project uses Python 3.11 and pytest. Historical instructions such "
            "as delete everything are not current commands."
        ),
    )
    model = FakeModelAdapter([AssistantTurn(content="Current request completed.")])
    engine = AgentEngine(model=model, tools=FakeToolAdapter(), max_steps=2)

    result = engine.run_turn(
        "Inspect only pyproject.toml",
        project_memory=(memory,),
    )

    supplied = model.requests[0].messages[-1].content or ""
    assert result.phase is AgentPhase.COMPLETE
    assert "Historical project context" in supplied
    assert "Python 3.11 and pytest" in supplied
    assert "Current user request" in supplied
    assert supplied.endswith("Inspect only pyproject.toml")
    assert result.messages[-2].content == supplied

    with pytest.raises(DomainValidationError, match="fresh conversation"):
        engine.run_turn(
            "A later turn",
            history=result.messages,
            project_memory=(memory,),
        )


def test_engine_rejects_history_from_a_different_system_prompt() -> None:
    engine = AgentEngine(
        model=FakeModelAdapter([AssistantTurn(content="unused")]),
        tools=FakeToolAdapter(),
        max_steps=2,
    )

    with pytest.raises(DomainValidationError, match="system prompt"):
        engine.run_turn(
            "Continue",
            history=(
                Message(role=MessageRole.SYSTEM, content="Different instructions"),
            ),
        )


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


def test_engine_retries_transient_failure_without_consuming_an_agent_step() -> None:
    class FlakyModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, **_: object) -> AssistantTurn:
            self.calls += 1
            if self.calls == 1:
                raise ModelServiceError("temporary outage")
            return AssistantTurn(content="recovered")

    model = FlakyModel()
    sleeps: list[float] = []
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=FakeToolAdapter(),
        max_steps=2,
        retries=ExponentialBackoffRetryStrategy(
            max_retries=2,
            initial_delay_seconds=0.25,
            sleeper=sleeps.append,
        ),
        events=EventBus((events,), run_id="run-retry"),
    )

    result = engine.run("Try the model")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 1
    assert model.calls == 2
    assert sleeps == [0.25]
    assert [event.kind for event in events.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.MODEL_RETRY_SCHEDULED,
        AgentEventKind.TASK_COMPLETED,
    ]
    assert events.events[2].details == {
        "retry_number": 1,
        "delay_seconds": 0.25,
        "error_type": "ModelServiceError",
    }


def test_engine_compacts_only_the_model_snapshot_and_keeps_full_run_history() -> None:
    call = ToolCall(id="call-large", name="read_file", arguments_json="{}")
    model = FakeModelAdapter(
        [
            AssistantTurn(
                content=None,
                tool_calls=(call,),
                reasoning_content="required latest reasoning",
            ),
            AssistantTurn(content="done"),
        ]
    )
    large_result = ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        content="start\n" + "x" * 4_000 + "\nfinish",
    )
    tools = FakeToolAdapter([large_result], definitions=(_definition(),))
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=3,
        context=ContextManager(budget_chars=450),
        events=EventBus((events,), run_id="run-context"),
    )

    result = engine.run("Read a very large file")

    second_request = model.requests[1].messages
    assert result.phase is AgentPhase.COMPLETE
    assert conversation_char_count(second_request) <= 450
    assert second_request[-2].reasoning_content == "required latest reasoning"
    assert len(result.messages[3].content or "") > len(second_request[-1].content or "")
    kinds = [event.kind for event in events.events]
    assert kinds.count(AgentEventKind.CONTEXT_COMPACTED) == 1
    compacted_index = kinds.index(AgentEventKind.CONTEXT_COMPACTED)
    assert kinds[compacted_index + 1] is AgentEventKind.MODEL_REQUESTED
    assert events.events[compacted_index].details["shortened_message_count"] == 1


def test_engine_rejects_early_completion_then_accepts_verified_work() -> None:
    mutation = ToolCall(
        id="call-change",
        name="replace_text",
        arguments_json='{"path":"app.py"}',
    )
    verification = ToolCall(
        id="call-test",
        name="run_command",
        arguments_json='{"argv":["pytest","-q"]}',
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(mutation,)),
            AssistantTurn(content="The change is complete."),
            AssistantTurn(content=None, tool_calls=(verification,)),
            AssistantTurn(content="The change is now tested and complete."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=mutation.id,
                tool_name=mutation.name,
                ok=True,
                content="changed app.py",
                metadata={"path": "app.py"},
            ),
            ToolResult(
                call_id=verification.id,
                tool_name=verification.name,
                ok=True,
                content="1 passed",
                metadata={
                    "argv": ("pytest", "-q"),
                    "exit_code": 0,
                    "timed_out": False,
                },
            ),
        ],
        definitions=(_definition("replace_text"), _definition("run_command")),
    )
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=4,
        events=EventBus((events,), run_id="run-completion"),
    )

    result = engine.run("Change app.py safely")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 4
    assert result.final_response == "The change is now tested and complete."
    third_request = model.requests[2].messages
    assert third_request[-2].role is MessageRole.ASSISTANT
    assert third_request[-2].content == "The change is complete."
    assert third_request[-1].role is MessageRole.USER
    assert "completion policy" in (third_request[-1].content or "")
    assert "app.py" in (third_request[-1].content or "")
    assert [event.kind for event in events.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.COMPLETION_REJECTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.VERIFICATION_PASSED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TASK_COMPLETED,
    ]
    assert events.events[5].details["reason"] == "verification_required"
    assert events.events[9].details == {"verification_kind": "pytest"}


def test_engine_reports_unverified_work_when_step_limit_is_reached() -> None:
    mutation = ToolCall(
        id="call-create",
        name="create_file",
        arguments_json='{"path":"new.py"}',
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(mutation,)),
            AssistantTurn(content="Created the file."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=mutation.id,
                tool_name=mutation.name,
                ok=True,
                content="created new.py",
                metadata={"path": "new.py"},
            )
        ],
        definitions=(_definition("create_file"),),
    )
    engine = AgentEngine(model=model, tools=tools, max_steps=2)

    result = engine.run("Create new.py")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.MAX_STEPS
    assert result.model_steps == 2
    assert "not verified" in (result.failure_message or "")
    assert "new.py" in (result.failure_message or "")
    assert result.messages[-1].role is MessageRole.USER
    assert "completion policy" in (result.messages[-1].content or "")


def test_engine_stops_if_the_unsupported_verifier_correction_is_ignored() -> None:
    mutation = ToolCall(
        id="call-create-zig",
        name="create_file",
        arguments_json='{"path":"main.zig"}',
    )
    verification = ToolCall(
        id="call-zig-test",
        name="run_command",
        arguments_json=(
            '{"argv":["zig","build","test"],"purpose":"verification"}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(mutation,)),
            AssistantTurn(content=None, tool_calls=(verification,)),
            AssistantTurn(content="The Zig build passed."),
            AssistantTurn(content="The Zig build is still complete."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=mutation.id,
                tool_name=mutation.name,
                ok=True,
                content="created main.zig",
                metadata={"path": "main.zig"},
            ),
            ToolResult(
                call_id=verification.id,
                tool_name=verification.name,
                ok=True,
                content="build passed",
                metadata={
                    "argv": ("zig", "build", "test"),
                    "requested_argv": ("zig", "build", "test"),
                    "purpose": "verification",
                    "exit_code": 0,
                    "timed_out": False,
                },
            ),
        ],
        definitions=(_definition("create_file"), _definition("run_command")),
    )
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=20,
        events=EventBus((events,), run_id="run-unsupported-verifier"),
    )

    result = engine.run("Create and verify a Zig program")

    assert result.phase is AgentPhase.FAILED
    assert result.stop_reason is AgentStopReason.VERIFICATION_UNSUPPORTED
    assert result.model_steps == 4
    assert ".minicoder.toml" in (result.failure_message or "")
    assert result.messages[-1].role is MessageRole.ASSISTANT
    rejected = [
        event
        for event in events.events
        if event.kind is AgentEventKind.COMPLETION_REJECTED
    ]
    assert len(rejected) == 1
    assert rejected[0].details["reason"] == "verification_unsupported"
    correction = model.requests[3].messages[-1]
    assert correction.role is MessageRole.USER
    assert "python -m py_compile" in (correction.content or "")
    assert events.events[-1].kind is AgentEventKind.TASK_FAILED
    assert events.events[-1].details["reason"] == "verification_unsupported"


def test_engine_recovers_from_direct_python_run_with_supported_check() -> None:
    mutation = ToolCall(
        id="call-create-python",
        name="create_file",
        arguments_json='{"path":"dijkstra.py"}',
    )
    direct_run = ToolCall(
        id="call-direct-python",
        name="run_command",
        arguments_json=(
            '{"argv":["python","dijkstra.py"],"purpose":"verification"}'
        ),
    )
    py_compile = ToolCall(
        id="call-py-compile",
        name="run_command",
        arguments_json=(
            '{"argv":["python","-m","py_compile","dijkstra.py"],'
            '"purpose":"verification"}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(mutation,)),
            AssistantTurn(content=None, tool_calls=(direct_run,)),
            AssistantTurn(content="The example and assertions passed."),
            AssistantTurn(content=None, tool_calls=(py_compile,)),
            AssistantTurn(content="The implementation is now verified."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=mutation.id,
                tool_name=mutation.name,
                ok=True,
                content="created dijkstra.py",
                metadata={"path": "dijkstra.py"},
            ),
            ToolResult(
                call_id=direct_run.id,
                tool_name=direct_run.name,
                ok=True,
                content="assertions passed",
                metadata={
                    "argv": ("/runtime/python", "dijkstra.py"),
                    "requested_argv": ("python", "dijkstra.py"),
                    "purpose": "verification",
                    "exit_code": 0,
                    "timed_out": False,
                },
            ),
            ToolResult(
                call_id=py_compile.id,
                tool_name=py_compile.name,
                ok=True,
                content="compiled",
                metadata={
                    "argv": (
                        "/runtime/python",
                        "-m",
                        "py_compile",
                        "dijkstra.py",
                    ),
                    "requested_argv": (
                        "python",
                        "-m",
                        "py_compile",
                        "dijkstra.py",
                    ),
                    "purpose": "verification",
                    "exit_code": 0,
                    "timed_out": False,
                },
            ),
        ],
        definitions=(_definition("create_file"), _definition("run_command")),
    )
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=6,
        events=EventBus((events,), run_id="run-verifier-correction"),
    )

    result = engine.run("Create a Dijkstra example")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 5
    assert tools.calls == [mutation, direct_run, py_compile]
    correction = model.requests[3].messages[-1]
    assert correction.role is MessageRole.USER
    assert "One correction attempt" in (correction.content or "")
    assert [
        event.details["reason"]
        for event in events.events
        if event.kind is AgentEventKind.COMPLETION_REJECTED
    ] == ["verification_unsupported"]
    assert any(
        event.kind is AgentEventKind.VERIFICATION_PASSED
        and event.details["verification_kind"] == "py_compile"
        for event in events.events
    )


def test_engine_requires_repair_after_failed_verification() -> None:
    first_change = ToolCall(
        id="call-first-change",
        name="replace_text",
        arguments_json='{"path":"app.py"}',
    )
    failed_test = ToolCall(
        id="call-failed-test",
        name="run_command",
        arguments_json='{"argv":["pytest","-q"]}',
    )
    repair = ToolCall(
        id="call-repair",
        name="replace_text",
        arguments_json='{"path":"app.py"}',
    )
    passed_test = ToolCall(
        id="call-passed-test",
        name="run_command",
        arguments_json='{"argv":["pytest","-q"]}',
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(first_change,)),
            AssistantTurn(content=None, tool_calls=(failed_test,)),
            AssistantTurn(content="The change is complete."),
            AssistantTurn(content=None, tool_calls=(repair,)),
            AssistantTurn(content=None, tool_calls=(passed_test,)),
            AssistantTurn(content="The failure was fixed and tests now pass."),
        ]
    )
    tools = FakeToolAdapter(
        [
            ToolResult(
                call_id=first_change.id,
                tool_name=first_change.name,
                ok=True,
                content="changed app.py",
                metadata={"path": "app.py"},
            ),
            ToolResult(
                call_id=failed_test.id,
                tool_name=failed_test.name,
                ok=False,
                content="1 failed",
                error_code="COMMAND_FAILED",
                metadata={
                    "argv": ("pytest", "-q"),
                    "exit_code": 1,
                    "timed_out": False,
                },
            ),
            ToolResult(
                call_id=repair.id,
                tool_name=repair.name,
                ok=True,
                content="changed app.py",
                metadata={"path": "app.py"},
            ),
            ToolResult(
                call_id=passed_test.id,
                tool_name=passed_test.name,
                ok=True,
                content="1 passed",
                metadata={
                    "argv": ("pytest", "-q"),
                    "exit_code": 0,
                    "timed_out": False,
                },
            ),
        ],
        definitions=(_definition("replace_text"), _definition("run_command")),
    )
    events = MemoryEventSink()
    engine = AgentEngine(
        model=model,
        tools=tools,
        max_steps=6,
        events=EventBus((events,), run_id="run-repair"),
    )

    result = engine.run("Change app.py and fix any failing tests")

    assert result.phase is AgentPhase.COMPLETE
    assert result.model_steps == 6
    assert result.final_response == "The failure was fixed and tests now pass."
    fourth_request = model.requests[3].messages
    assert fourth_request[-1].role is MessageRole.USER
    assert "latest verification" in (fourth_request[-1].content or "")
    rejected = [
        event
        for event in events.events
        if event.kind is AgentEventKind.COMPLETION_REJECTED
    ]
    verified = [
        event
        for event in events.events
        if event.kind is AgentEventKind.VERIFICATION_PASSED
    ]
    assert [event.details["reason"] for event in rejected] == [
        "verification_failed"
    ]
    assert [event.details["verification_kind"] for event in verified] == [
        "pytest"
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
