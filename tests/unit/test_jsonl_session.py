import json
from pathlib import Path

from minicoder.adapters.jsonl_session import JsonlSessionArchive
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from minicoder.domain.session import ArchivedTurnStatus
from minicoder.domain.state import AgentPhase, AgentRunResult, AgentStopReason


def _completed_result(messages: tuple[Message, ...]) -> AgentRunResult:
    return AgentRunResult(
        phase=AgentPhase.COMPLETE,
        stop_reason=AgentStopReason.FINAL_RESPONSE,
        model_steps=2,
        messages=messages,
        final_response="finished",
    )


def test_archive_records_complete_requests_responses_tools_and_turns(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    archive = JsonlSessionArchive(
        workspace=workspace,
        storage_root=tmp_path / "sessions",
    )
    call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"path":"private.py"}',
    )
    definition = ToolDefinition(
        name="read_file",
        description="Read one exact file.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    request = (
        Message(role=MessageRole.SYSTEM, content="system rules"),
        Message(role=MessageRole.USER, content="inspect private.py"),
    )
    response = AssistantTurn(
        content="calling the reader",
        tool_calls=(call,),
        reasoning_content="provider-visible reasoning",
    )
    result = ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        content="exact private tool output",
        metadata={"content_chars": 25},
    )
    history = request + (response.as_message(), result.as_message())

    archive.record_turn_started(task="inspect private.py", turn_index=1)
    archive.record_model_request(
        messages=request,
        tools=(definition,),
        request_kind="execution",
        turn_index=1,
        model_step=1,
    )
    archive.record_model_response(
        turn=response,
        request_kind="execution",
        turn_index=1,
        model_step=1,
    )
    archive.record_tool_result(
        call=call,
        result=result,
        turn_index=1,
        model_step=1,
    )
    archive.record_turn_result(
        task="inspect private.py",
        result=_completed_result(history),
        turn_index=1,
    )
    archive.record_maintenance(
        context_summary="Inspected private.py and finished.",
        memory_summary=None,
        used_fallback=False,
        turn_index=1,
        model_step=2,
    )
    archive.close(context_summary="Inspected private.py and finished.")

    records = [
        json.loads(line)
        for line in archive.path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["type"] for record in records] == [
        "session_started",
        "turn_started",
        "model_request",
        "model_response",
        "tool_result",
        "turn_result",
        "maintenance",
        "session_closed",
    ]
    serialized = archive.path.read_text(encoding="utf-8")
    assert "provider-visible reasoning" in serialized
    assert "exact private tool output" in serialized
    assert '"parameters_schema"' in serialized
    assert archive.path.stat().st_mode & 0o777 == 0o600


def test_new_archive_loads_latest_context_even_after_a_completed_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = tmp_path / "sessions"
    workspace.mkdir()
    first = JsonlSessionArchive(workspace=workspace, storage_root=root)
    messages = (
        Message(role=MessageRole.SYSTEM, content="system"),
        Message(role=MessageRole.USER, content="old request"),
        Message(role=MessageRole.ASSISTANT, content="old final"),
    )
    first.record_turn_started(task="old request", turn_index=1)
    first.record_turn_result(
        task="old request",
        result=_completed_result(messages),
        turn_index=1,
    )
    first.record_maintenance(
        context_summary="The previous process completed an important inspection.",
        memory_summary="Stable project fact.",
        used_fallback=False,
        turn_index=1,
        model_step=2,
    )
    first.close(context_summary="The previous process completed an inspection.")

    second = JsonlSessionArchive(workspace=workspace, storage_root=root)
    restored = second.load_latest_context()

    assert restored is not None
    assert restored.session_id == first.session_id
    assert restored.last_task == "old request"
    assert restored.status is ArchivedTurnStatus.COMPLETE
    assert restored.stop_reason == AgentStopReason.FINAL_RESPONSE.value
    assert restored.context_summary.startswith("The previous process completed")
    assert restored.recent_messages[-1].content == "old final"


def test_archive_recovers_an_interrupted_turn_without_maintenance(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = tmp_path / "sessions"
    workspace.mkdir()
    interrupted = JsonlSessionArchive(workspace=workspace, storage_root=root)
    completed_messages = (
        Message(role=MessageRole.SYSTEM, content="system"),
        Message(role=MessageRole.USER, content="first task"),
        Message(role=MessageRole.ASSISTANT, content="first task complete"),
    )
    interrupted.record_turn_started(task="first task", turn_index=1)
    interrupted.record_turn_result(
        task="first task",
        result=_completed_result(completed_messages),
        turn_index=1,
    )
    interrupted.record_maintenance(
        context_summary="The first task completed and its decisions still matter.",
        memory_summary=None,
        used_fallback=False,
        turn_index=1,
        model_step=2,
    )

    call = ToolCall(
        id="call-read",
        name="read_file",
        arguments_json='{"path":"unfinished.py"}',
    )
    interrupted.record_turn_started(task="unfinished task", turn_index=2)
    interrupted.record_model_request(
        messages=(
            Message(role=MessageRole.SYSTEM, content="system"),
            Message(role=MessageRole.USER, content="unfinished task"),
        ),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                parameters_schema={"type": "object"},
            ),
        ),
        request_kind="execution",
        turn_index=2,
        model_step=1,
    )
    interrupted.record_model_response(
        turn=AssistantTurn(
            content=None,
            tool_calls=(call,),
            reasoning_content="private interrupted reasoning",
        ),
        request_kind="execution",
        turn_index=2,
        model_step=1,
    )
    interrupted.record_tool_result(
        call=call,
        result=ToolResult(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="partial file contents",
        ),
        turn_index=2,
        model_step=1,
    )

    restarted = JsonlSessionArchive(workspace=workspace, storage_root=root)
    restored = restarted.load_latest_context()

    assert restored is not None
    assert restored.status is ArchivedTurnStatus.IN_PROGRESS
    assert restored.stop_reason is None
    assert "first task completed" in restored.context_summary
    assert "interrupted before a terminal result" in restored.context_summary
    assert restored.recent_messages[-2].tool_calls == (call,)
    assert "partial file contents" in (restored.recent_messages[-1].content or "")


def test_archive_ignores_other_workspaces_and_corrupt_lines(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = JsonlSessionArchive(workspace=first_workspace, storage_root=root)
    first.record_turn_started(task="first project task", turn_index=1)
    with first.path.open("a", encoding="utf-8") as archive_file:
        archive_file.write("not-json\n")

    same_workspace = JsonlSessionArchive(
        workspace=first_workspace,
        storage_root=root,
    )
    other_workspace = JsonlSessionArchive(
        workspace=second_workspace,
        storage_root=root,
    )

    assert same_workspace.load_latest_context() is not None
    assert other_workspace.load_latest_context() is None
