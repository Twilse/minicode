from pathlib import Path
import json

from minicoder.adapters.jsonl_memory import JsonlProjectMemoryStore
from minicoder.adapters.jsonl_session import JsonlSessionArchive
from minicoder.adapters.jsonl_trace import JsonlTraceSink
from minicoder.adapters.subprocess_runner import PosixSubprocessAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.domain.models import AssistantTurn, MessageRole
from minicoder.domain.state import AgentPhase
from tests.fakes import FakeModelAdapter


def _context(workspace: Path, api_key: str = "private-api-key"):
    return ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": api_key,
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "test-model",
            "MINICODER_MEMORY_ENABLED": "true",
        },
        workspace=workspace,
        platform_name="darwin",
    )


def test_successful_turn_is_summarized_and_loaded_by_a_new_session(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    memory_root = tmp_path / "private-memory"
    workspace.mkdir()
    first_store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=memory_root,
        sensitive_values=("private-api-key",),
    )
    first_model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Add the parser.\n2. Run tests."),
            AssistantTurn(content="Added parser.py and passed 4 tests."),
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": (
                            "Added parser.py with compatibility checks; "
                            "4 tests passed."
                        ),
                        "memory_action": "append",
                        "memory_summary": (
                            "Added parser.py with compatibility checks; "
                            "4 tests passed. private-api-key"
                        ),
                    }
                )
            ),
        ]
    )
    first_archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=tmp_path / "private-sessions",
    )
    first_session = ApplicationFactory.create_agent_session(
        _context(workspace),
        model_adapter=first_model,
        process_adapter=PosixSubprocessAdapter(),
        memory_store=first_store,
        session_archive=first_archive,
    )

    first_result = first_session.run("Add a parser without exposing this task body")

    assert first_result.phase is AgentPhase.COMPLETE
    assert len(first_model.requests) == 3
    assert first_model.requests[2].tools == ()
    assert [message.role for message in first_model.requests[2].messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    persisted = first_store.path.read_text(encoding="utf-8")
    assert "Added parser.py with compatibility checks" in persisted
    assert "private-api-key" not in persisted
    assert "Add a parser without exposing this task body" not in persisted
    assert all(
        "compatibility checks" not in (message.content or "")
        for message in first_result.messages
    )

    second_store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=memory_root,
        sensitive_values=("private-api-key",),
    )
    second_model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Review the project memory.\n2. Answer."),
            AssistantTurn(content="Used the earlier parser information."),
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": (
                            "Confirmed the previous parser context was reused."
                        ),
                        "memory_action": "none",
                        "memory_summary": None,
                    }
                )
            ),
        ]
    )
    second_archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=tmp_path / "private-sessions",
    )
    second_session = ApplicationFactory.create_agent_session(
        _context(workspace),
        model_adapter=second_model,
        process_adapter=PosixSubprocessAdapter(),
        memory_store=second_store,
        session_archive=second_archive,
    )

    second_result = second_session.run("What parser work was done before?")

    first_request = second_model.requests[0].messages
    assert second_result.phase is AgentPhase.COMPLETE
    assert "Workspace memory and recent-session context" in (
        first_request[0].content or ""
    )
    assert "Recent cross-process and current-session context" in (
        first_request[0].content or ""
    )
    assert "Durable project memory" in (first_request[0].content or "")
    assert "Added parser.py with compatibility checks" in (
        first_request[0].content or ""
    )
    assert "What parser work was done before?" in (
        first_request[-1].content or ""
    )
    assert "Host planning requirement" in (first_request[-1].content or "")


def test_project_memory_isolated_by_workspace_and_trace_stays_sanitized(
    tmp_path: Path,
) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    memory_root = tmp_path / "memory"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first_store = JsonlProjectMemoryStore(
        workspace=first_workspace,
        storage_root=memory_root,
    )
    model = FakeModelAdapter(
        [
            AssistantTurn(content="1. Inspect the project.\n2. Answer."),
            AssistantTurn(content="private final response"),
            AssistantTurn(
                content=json.dumps(
                    {
                        "context_summary": "private current session context",
                        "memory_action": "append",
                        "memory_summary": "private semantic memory",
                    }
                )
            ),
        ]
    )
    trace_path = tmp_path / "trace.jsonl"
    session = ApplicationFactory.create_agent_session(
        _context(first_workspace),
        model_adapter=model,
        process_adapter=PosixSubprocessAdapter(),
        event_sinks=(JsonlTraceSink(trace_path),),
        memory_store=first_store,
        session_archive=JsonlSessionArchive(
            workspace=first_workspace,
            storage_root=tmp_path / "private-sessions",
        ),
    )

    session.run("private user task")

    second_store = JsonlProjectMemoryStore(
        workspace=second_workspace,
        storage_root=memory_root,
    )
    assert second_store.load_recent() == ()
    trace = trace_path.read_text(encoding="utf-8")
    assert "private user task" not in trace
    assert "private final response" not in trace
    assert "private semantic memory" not in trace
    assert "memory_summary_requested" in trace
    assert "memory_saved" in trace
