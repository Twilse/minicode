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
from minicoder.domain.session import ArchivedTurnStatus, ContextCheckpoint
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

    archive.record_turn_started(task="inspect private.py", history=(), turn_index=1)
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
        memory_summary=None,
        used_fallback=False,
        turn_index=1,
        model_step=2,
    )
    archive.close()

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
    first.record_turn_started(task="old request", history=(), turn_index=1)
    first.record_turn_result(
        task="old request",
        result=_completed_result(messages),
        turn_index=1,
    )
    first.record_maintenance(
        memory_summary="Stable project fact.",
        used_fallback=False,
        turn_index=1,
        model_step=2,
    )
    first.close()

    second = JsonlSessionArchive(workspace=workspace, storage_root=root)
    restored = second.load_latest_context()

    assert restored is not None
    assert restored.session_id == first.session_id
    assert restored.last_task == "old request"
    assert restored.status is ArchivedTurnStatus.COMPLETE
    assert restored.stop_reason == AgentStopReason.FINAL_RESPONSE.value
    assert restored.final_response == "finished"
    assert restored.messages == messages[1:]


def test_archive_loads_exact_external_dialogue_across_all_previous_processes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = tmp_path / "sessions"
    workspace.mkdir()

    first = JsonlSessionArchive(workspace=workspace, storage_root=root)
    first_messages = (
        Message(role=MessageRole.SYSTEM, content="internal system prompt"),
        Message(role=MessageRole.USER, content="internal augmented request"),
        Message(role=MessageRole.ASSISTANT, content="internal plan"),
    )
    first.record_turn_started(
        task="用户原始问题一",
        history=(),
        turn_index=1,
    )
    first.record_turn_result(
        task="用户原始问题一",
        result=AgentRunResult(
            phase=AgentPhase.COMPLETE,
            stop_reason=AgentStopReason.FINAL_RESPONSE,
            model_steps=2,
            messages=first_messages,
            final_response="最终回复一，包含 **Markdown**。",
        ),
        turn_index=1,
    )
    first.close()

    second = JsonlSessionArchive(workspace=workspace, storage_root=root)
    second.record_turn_started(
        task="用户原始问题二",
        history=first_messages,
        turn_index=1,
    )
    second.record_turn_result(
        task="用户原始问题二",
        result=AgentRunResult(
            phase=AgentPhase.FAILED,
            stop_reason=AgentStopReason.MODEL_ERROR,
            model_steps=1,
            messages=first_messages,
            failure_message="model unavailable",
        ),
        turn_index=1,
    )
    second.record_turn_started(
        task="用户原始问题三",
        history=first_messages,
        turn_index=2,
    )
    second.close()

    current = JsonlSessionArchive(workspace=workspace, storage_root=root)
    turns = current.load_dialogue_history()

    assert [turn.task for turn in turns] == [
        "用户原始问题一",
        "用户原始问题二",
        "用户原始问题三",
    ]
    assert [turn.status for turn in turns] == [
        ArchivedTurnStatus.COMPLETE,
        ArchivedTurnStatus.FAILED,
        ArchivedTurnStatus.IN_PROGRESS,
    ]
    assert turns[0].final_response == "最终回复一，包含 **Markdown**。"
    assert turns[1].failure_message == "model unavailable"
    assert turns[2].final_response is None
    assert all("internal" not in turn.task for turn in turns)


def test_archive_restores_more_than_the_old_twenty_four_message_limit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    root = tmp_path / "sessions"
    workspace.mkdir()
    first = JsonlSessionArchive(workspace=workspace, storage_root=root)
    messages = (
        Message(role=MessageRole.SYSTEM, content="system"),
        *tuple(
            Message(
                role=MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT,
                content=f"message-{index}",
            )
            for index in range(30)
        ),
    )
    first.record_turn_started(task="message-28", history=messages[:-2], turn_index=1)
    first.record_turn_result(
        task="message-28",
        result=_completed_result(messages),
        turn_index=1,
    )
    first.close()

    restored = JsonlSessionArchive(
        workspace=workspace,
        storage_root=root,
    ).load_latest_context()

    assert restored is not None
    assert len(restored.messages) == 30
    assert restored.messages == messages[1:]


def test_archive_restores_the_latest_context_checkpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = tmp_path / "sessions"
    workspace.mkdir()
    first = JsonlSessionArchive(workspace=workspace, storage_root=root)
    messages = (
        Message(role=MessageRole.SYSTEM, content="system"),
        Message(role=MessageRole.USER, content="old request"),
        Message(role=MessageRole.ASSISTANT, content="old result"),
    )
    checkpoint = ContextCheckpoint(
        summary="The old request completed successfully.",
        covered_message_count=2,
        source_hash="a" * 64,
    )
    first.record_turn_started(task="old request", history=(), turn_index=1)
    first.record_turn_result(
        task="old request",
        result=_completed_result(messages),
        turn_index=1,
    )
    first.record_context_checkpoint(
        checkpoint=checkpoint,
        turn_index=1,
        model_step=2,
    )
    first.close()

    restored = JsonlSessionArchive(
        workspace=workspace,
        storage_root=root,
    ).load_latest_context()

    assert restored is not None
    assert restored.context_checkpoint == checkpoint
    assert restored.messages == messages[1:]


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
    interrupted.record_turn_started(task="first task", history=(), turn_index=1)
    interrupted.record_turn_result(
        task="first task",
        result=_completed_result(completed_messages),
        turn_index=1,
    )
    interrupted.record_maintenance(
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
    interrupted.record_turn_started(
        task="unfinished task",
        history=completed_messages,
        turn_index=2,
    )
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
    assert completed_messages[1:] == restored.messages[:2]
    assert restored.messages[-2].tool_calls == (call,)
    assert "partial file contents" in (restored.messages[-1].content or "")


def test_archive_ignores_other_workspaces_and_corrupt_lines(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = JsonlSessionArchive(workspace=first_workspace, storage_root=root)
    first.record_turn_started(task="first project task", history=(), turn_index=1)
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
