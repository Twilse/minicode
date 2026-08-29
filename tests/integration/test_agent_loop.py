from pathlib import Path

from minicoder.adapters.subprocess_runner import PosixSubprocessAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.models import AssistantTurn, MessageRole, ToolCall
from minicoder.domain.state import AgentPhase, AgentStopReason
from tests.fakes import FakeModelAdapter


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
            '{"path":"created_by_agent.txt","content":"agent result\\n"}'
        ),
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(create_call,)),
            AssistantTurn(content="Created and verified the requested file."),
        ]
    )
    session = ApplicationFactory.create_agent_session(
        context,
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
    )

    result = session.run("Create created_by_agent.txt")

    assert result.phase is AgentPhase.COMPLETE
    assert result.stop_reason is AgentStopReason.FINAL_RESPONSE
    assert result.model_steps == 2
    assert (tmp_path / "created_by_agent.txt").read_text(encoding="utf-8") == (
        "agent result\n"
    )
    assert [message.role for message in model.requests[1].messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert session.closed is True
