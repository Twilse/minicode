from pathlib import Path

from minicoder.adapters.subprocess_runner import PosixSubprocessAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.events import AgentEventKind
from minicoder.domain.models import AssistantTurn, MessageRole, ToolCall
from minicoder.domain.state import AgentPhase, AgentStopReason
from tests.fakes import FakeModelAdapter, MemoryEventSink


def test_factory_session_runs_model_tool_model_loop(tmp_path: Path) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
            "MINICODER_MAX_STEPS": "3",
        },
        workspace=tmp_path,
        platform_name="darwin",
    )
    create_call = ToolCall(
        id="call-create",
        name="create_file",
        arguments_json=(
            '{"path":"created_by_agent.py","content":"value = 1\\n"}'
        ),
    )
    verify_call = ToolCall(
        id="call-verify",
        name="run_command",
        arguments_json=(
            '{"argv":["python","-m","py_compile","created_by_agent.py"]}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(create_call,)),
            AssistantTurn(content=None, tool_calls=(verify_call,)),
            AssistantTurn(content="Created and verified the requested file."),
        ]
    )
    events = MemoryEventSink()
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
        event_sinks=(events,),
    )

    result = session.run("Create created_by_agent.py")

    assert result.phase is AgentPhase.COMPLETE
    assert result.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert result.model_steps == 3
    assert (tmp_path / "created_by_agent.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert [message.role for message in model.requests[1].messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert [message.role for message in model.requests[2].messages[-2:]] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert session.closed is True
    assert [event.kind for event in events.events] == [
        AgentEventKind.TASK_STARTED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TOOL_CALLED,
        AgentEventKind.TOOL_FINISHED,
        AgentEventKind.VERIFICATION_PASSED,
        AgentEventKind.MODEL_REQUESTED,
        AgentEventKind.TASK_COMPLETED,
    ]


def test_factory_session_continues_unverified_work_in_a_second_user_turn(
    tmp_path: Path,
) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "not-used",
            "MINICODER_MAX_STEPS": "2",
        },
        workspace=tmp_path,
        platform_name="darwin",
    )
    create_call = ToolCall(
        id="call-create-multi",
        name="create_file",
        arguments_json=(
            '{"path":"continued.py","content":"value = 2\\n"}'
        ),
    )
    verify_call = ToolCall(
        id="call-verify-multi",
        name="run_command",
        arguments_json=(
            '{"argv":["python","-m","py_compile","continued.py"]}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(create_call,)),
            AssistantTurn(content="The file is created."),
            AssistantTurn(content=None, tool_calls=(verify_call,)),
            AssistantTurn(content="The file is now verified."),
        ]
    )
    events = MemoryEventSink()
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
        event_sinks=(events,),
    )

    with session:
        first = session.submit("Create continued.py")
        assert first.phase is AgentPhase.FAILED
        assert first.stop_reason is AgentStopReason.MAX_STEPS
        assert session.closed is False

        second = session.submit("Continue by verifying the file")

    assert second.phase is AgentPhase.COMPLETE
    assert second.model_steps == 2
    assert second.final_response == "The file is now verified."
    assert (tmp_path / "continued.py").read_text(encoding="utf-8") == (
        "value = 2\n"
    )
    second_turn_request = model.requests[2].messages
    assert second_turn_request[-1].content == "Continue by verifying the file"
    assert sum(
        message.role is MessageRole.SYSTEM for message in second_turn_request
    ) == 1
    assert session.closed is True
    assert [event.kind for event in events.events].count(
        AgentEventKind.TASK_STARTED
    ) == 2
    assert [event.sequence for event in events.events] == list(
        range(1, len(events.events) + 1)
    )
