import json
from pathlib import Path

from minicoder.adapters.jsonl_session import JsonlSessionArchive
from minicoder.adapters.subprocess_runner import PosixSubprocessAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.models import AssistantTurn, ToolCall
from minicoder.domain.state import AgentPhase, AgentStopReason
from tests.fakes import FakeModelAdapter


def _context(workspace: Path):
    return ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "not-used",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
            "MINICODER_MAX_STEPS": "1",
            "MINICODER_PLANNING_ENABLED": "false",
            "MINICODER_MEMORY_ENABLED": "false",
            "MINICODER_SESSION_ARCHIVE_ENABLED": "true",
        },
        workspace=workspace,
        platform_name="darwin",
    )


def test_new_process_always_receives_the_latest_failed_session_context(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    session_root = tmp_path / "sessions"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('hello')\n", encoding="utf-8")
    call = ToolCall(
        id="call-list",
        name="list_files",
        arguments_json='{"path":"."}',
    )
    first_model = FakeModelAdapter(
        [
            AssistantTurn(content=None, tool_calls=(call,)),
        ]
    )
    first_archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=session_root,
    )
    first_session = ApplicationFactory.create_agent_session(
        _context(workspace),
        model_adapter=first_model,
        process_adapter=PosixSubprocessAdapter(),
        session_archive=first_archive,
    )

    failed = first_session.run("Finish the old feature")

    assert failed.phase is AgentPhase.FAILED
    assert failed.stop_reason is AgentStopReason.MAX_STEPS
    archived_text = first_archive.path.read_text(encoding="utf-8")
    assert "Finish the old feature" in archived_text
    assert '"name":"list_files"' in archived_text
    assert '"type":"tool_result"' in archived_text
    assert '"request_kind":"maintenance"' not in archived_text

    second_model = FakeModelAdapter(
        [
            AssistantTurn(content="I inspected the unrelated current request."),
        ]
    )
    second_archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=session_root,
    )
    second_session = ApplicationFactory.create_agent_session(
        _context(workspace),
        model_adapter=second_model,
        process_adapter=PosixSubprocessAdapter(),
        session_archive=second_archive,
    )

    completed = second_session.run("Inspect a new unrelated concern")

    assert completed.phase is AgentPhase.COMPLETE
    first_request = second_model.requests[0]
    system = first_request.messages[0].content or ""
    boundary = next(
        message.content or ""
        for message in first_request.messages
        if "Previous process boundary" in (message.content or "")
    )
    assert "Previous process boundary" not in system
    assert "Finish the old feature" not in system
    assert "Previous process boundary" in boundary
    assert "Status: failed" in boundary
    assert "Stop reason: max_steps" in boundary
    assert "maximum of 1 model steps" not in boundary
    assert any(
        message.tool_calls and message.tool_calls[0].name == "list_files"
        for message in first_request.messages
    )
    assert any(
        message.content == "Finish the old feature"
        for message in first_request.messages
    )
    assert first_request.messages[-1].content == "Inspect a new unrelated concern"
    assert {definition.name for definition in first_request.tools} >= {
        "list_files",
        "read_file",
        "replace_text",
        "run_command",
    }
