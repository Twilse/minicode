"""Private append-only JSONL archive for complete MiniCoder sessions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from minicoder.application.ports import SessionArchivePort
from minicoder.domain.errors import SessionPersistenceError
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from minicoder.domain.session import (
    ArchivedDialogueTurn,
    ArchivedTurnStatus,
    ContextCheckpoint,
    RecentSessionContext,
)
from minicoder.domain.state import AgentPhase, AgentRunResult

SESSION_SCHEMA_VERSION = 1


class JsonlSessionArchive(SessionArchivePort):
    """Record exact local exchanges and recover the latest workspace session."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        storage_root: str | Path | None = None,
    ) -> None:
        resolved_workspace = Path(workspace).expanduser().resolve()
        root = (
            Path.home() / ".minicoder" / "sessions"
            if storage_root is None
            else Path(storage_root).expanduser()
        )
        digest = hashlib.sha256(
            str(resolved_workspace).encode("utf-8")
        ).hexdigest()[:24]
        self._directory = root / digest
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            if not self._directory.is_dir():
                raise OSError("session archive path is not a directory")
            _best_effort_chmod(root, 0o700)
            _best_effort_chmod(self._directory, 0o700)
        except OSError as exc:
            raise SessionPersistenceError(
                "could not prepare the private session archive"
            ) from exc

        now = datetime.now(timezone.utc)
        self._session_id = uuid4().hex
        timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
        self._path = self._directory / f"{timestamp}-{self._session_id}.jsonl"
        self._workspace = resolved_workspace
        self._sequence = 0
        self._closed = False
        self._append(
            "session_started",
            {"workspace": str(self._workspace)},
            recorded_at=now,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def path(self) -> Path:
        return self._path

    def load_latest_context(self) -> RecentSessionContext | None:
        """Read the newest earlier archive containing at least one user turn."""

        try:
            candidates = sorted(
                (
                    path
                    for path in self._directory.glob("*.jsonl")
                    if path != self._path and path.is_file()
                ),
                reverse=True,
            )
        except OSError as exc:
            raise SessionPersistenceError(
                "could not enumerate previous session archives"
            ) from exc

        for candidate in candidates:
            context = self._context_from_file(candidate)
            if context is not None:
                return context
        return None

    def load_dialogue_history(self) -> tuple[ArchivedDialogueTurn, ...]:
        """Reconstruct all external turns without exposing internal protocol logs."""

        try:
            candidates = sorted(
                path
                for path in self._directory.glob("*.jsonl")
                if path != self._path and path.is_file()
            )
        except OSError as exc:
            raise SessionPersistenceError(
                "could not enumerate previous session archives"
            ) from exc

        return tuple(
            turn
            for candidate in candidates
            for turn in self._dialogue_turns_from_file(candidate)
        )

    def record_turn_started(
        self,
        *,
        task: str,
        history: Sequence[Message],
        turn_index: int,
    ) -> None:
        self._append(
            "turn_started",
            {
                "task": task,
                "turn_index": turn_index,
                "history": [_message_payload(message) for message in history],
            },
        )

    def record_model_request(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        request_kind: str,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._append(
            "model_request",
            {
                "turn_index": turn_index,
                "model_step": model_step,
                "request_kind": request_kind,
                "messages": [_message_payload(message) for message in messages],
                "tools": [_tool_definition_payload(tool) for tool in tools],
            },
        )

    def record_model_response(
        self,
        *,
        turn: AssistantTurn,
        request_kind: str,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._append(
            "model_response",
            {
                "turn_index": turn_index,
                "model_step": model_step,
                "request_kind": request_kind,
                "response": _assistant_turn_payload(turn),
            },
        )

    def record_tool_result(
        self,
        *,
        call: ToolCall,
        result: ToolResult,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._append(
            "tool_result",
            {
                "turn_index": turn_index,
                "model_step": model_step,
                "call": _tool_call_payload(call),
                "result": {
                    "call_id": result.call_id,
                    "tool_name": result.tool_name,
                    "ok": result.ok,
                    "content": result.content,
                    "error_code": result.error_code,
                    "metadata": _json_safe(result.metadata),
                },
            },
        )

    def record_turn_result(
        self,
        *,
        task: str,
        result: AgentRunResult,
        turn_index: int,
    ) -> None:
        self._append(
            "turn_result",
            {
                "turn_index": turn_index,
                "task": task,
                "phase": result.phase.value,
                "stop_reason": result.stop_reason.value,
                "model_steps": result.model_steps,
                "final_response": result.final_response,
                "failure_message": result.failure_message,
                "messages": [_message_payload(message) for message in result.messages],
            },
        )

    def record_maintenance(
        self,
        *,
        memory_summary: str | None,
        used_fallback: bool,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._append(
            "maintenance",
            {
                "turn_index": turn_index,
                "model_step": model_step,
                "memory_summary": memory_summary,
                "used_fallback": used_fallback,
            },
        )

    def record_context_checkpoint(
        self,
        *,
        checkpoint: ContextCheckpoint,
        turn_index: int,
        model_step: int,
    ) -> None:
        self._append(
            "context_checkpoint",
            {
                "turn_index": turn_index,
                "model_step": model_step,
                "summary": checkpoint.summary,
                "covered_message_count": checkpoint.covered_message_count,
                "source_hash": checkpoint.source_hash,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._append(
            "session_closed",
            {},
        )
        self._closed = True

    def _append(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        recorded_at: datetime | None = None,
    ) -> None:
        if self._closed:
            raise SessionPersistenceError("session archive is already closed")
        self._sequence += 1
        timestamp = datetime.now(timezone.utc) if recorded_at is None else recorded_at
        record = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self._session_id,
            "sequence": self._sequence,
            "recorded_at": _timestamp(timestamp),
            "type": kind,
            "payload": _json_safe(payload),
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        existed = self._path.exists()
        try:
            with self._path.open("a", encoding="utf-8", newline="") as archive:
                archive.write(f"{line}\n")
                archive.flush()
            if not existed:
                _best_effort_chmod(self._path, 0o600)
        except OSError as exc:
            raise SessionPersistenceError(
                "could not append to the private session archive"
            ) from exc

    def _context_from_file(self, path: Path) -> RecentSessionContext | None:
        records = _read_records(path)
        if not records:
            return None
        session_id = ""
        last_recorded_at: datetime | None = None
        last_task = ""
        last_turn_index: int | None = None
        last_result: Mapping[str, Any] | None = None
        partial_messages: tuple[Message, ...] = ()
        context_checkpoint: ContextCheckpoint | None = None
        for record in records:
            raw_session_id = record.get("session_id")
            if isinstance(raw_session_id, str) and raw_session_id:
                session_id = raw_session_id
            parsed_time = _parse_timestamp(record.get("recorded_at"))
            if parsed_time is not None:
                last_recorded_at = parsed_time
            kind = record.get("type")
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if kind == "context_checkpoint":
                parsed_checkpoint = _parse_context_checkpoint(payload)
                if parsed_checkpoint is not None:
                    context_checkpoint = parsed_checkpoint
                continue
            if kind == "turn_started" and isinstance(payload.get("task"), str):
                last_task = payload["task"]
                raw_turn_index = payload.get("turn_index")
                last_turn_index = (
                    raw_turn_index
                    if isinstance(raw_turn_index, int)
                    and not isinstance(raw_turn_index, bool)
                    else None
                )
                last_result = None
                raw_history = payload.get("history")
                if isinstance(raw_history, list):
                    parsed_history = _conversation_messages(
                        _parse_messages(raw_history)
                    )
                    partial_messages = (
                        *parsed_history,
                        Message(role=MessageRole.USER, content=last_task),
                    )
                else:
                    # Schema-v1 archives created before full-history checkpoints
                    # can still recover from their first exact model request.
                    partial_messages = ()
            elif not _belongs_to_turn(payload, last_turn_index):
                continue
            elif kind == "model_request" and payload.get(
                "request_kind"
            ) in {"planning", "execution"}:
                parsed_request = _parse_messages(payload.get("messages"))
                if parsed_request and not partial_messages:
                    partial_messages = _conversation_messages(parsed_request)
            elif kind == "model_response" and payload.get(
                "request_kind"
            ) in {"planning", "execution"}:
                response_message = _parse_assistant_response(
                    payload.get("response")
                )
                if response_message is not None:
                    partial_messages = (*partial_messages, response_message)
            elif kind == "tool_result":
                result_message = _parse_tool_result_message(
                    payload.get("result")
                )
                if result_message is not None:
                    partial_messages = (*partial_messages, result_message)
            elif kind == "turn_result" and payload.get("task") == last_task:
                last_result = payload

        if not session_id or last_recorded_at is None or not last_task.strip():
            return None

        status = ArchivedTurnStatus.IN_PROGRESS
        stop_reason: str | None = None
        messages: tuple[Message, ...] = ()
        final_response: str | None = None
        failure_message: str | None = None
        if last_result is not None:
            raw_phase = last_result.get("phase")
            if raw_phase == AgentPhase.COMPLETE.value:
                status = ArchivedTurnStatus.COMPLETE
            elif raw_phase == AgentPhase.FAILED.value:
                status = ArchivedTurnStatus.FAILED
            raw_reason = last_result.get("stop_reason")
            if isinstance(raw_reason, str) and raw_reason.strip():
                stop_reason = raw_reason
            raw_final = last_result.get("final_response")
            if isinstance(raw_final, str) and raw_final.strip():
                final_response = raw_final
            raw_failure = last_result.get("failure_message")
            if isinstance(raw_failure, str) and raw_failure.strip():
                failure_message = raw_failure
            raw_messages = last_result.get("messages")
            if isinstance(raw_messages, list):
                messages = _conversation_messages(_parse_messages(raw_messages))
        else:
            messages = partial_messages

        return RecentSessionContext(
            session_id=session_id,
            recorded_at=last_recorded_at,
            last_task=last_task,
            status=status,
            stop_reason=stop_reason,
            messages=messages,
            final_response=final_response,
            failure_message=failure_message,
            context_checkpoint=context_checkpoint,
        )

    def _dialogue_turns_from_file(
        self,
        path: Path,
    ) -> tuple[ArchivedDialogueTurn, ...]:
        records = _read_records(path)
        if not records:
            return ()

        session_id = next(
            (
                value
                for record in records
                if isinstance((value := record.get("session_id")), str) and value
            ),
            "",
        )
        if not session_id:
            return ()

        started: dict[int, tuple[datetime, str]] = {}
        results: dict[int, Mapping[str, Any]] = {}
        order: list[int] = []
        for record in records:
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            raw_turn_index = payload.get("turn_index")
            if (
                not isinstance(raw_turn_index, int)
                or isinstance(raw_turn_index, bool)
                or raw_turn_index <= 0
            ):
                continue
            if record.get("type") == "turn_started":
                task = payload.get("task")
                recorded_at = _parse_timestamp(record.get("recorded_at"))
                if (
                    isinstance(task, str)
                    and task.strip()
                    and recorded_at is not None
                    and raw_turn_index not in started
                ):
                    started[raw_turn_index] = (recorded_at, task)
                    order.append(raw_turn_index)
            elif record.get("type") == "turn_result":
                results[raw_turn_index] = payload

        turns: list[ArchivedDialogueTurn] = []
        for turn_index in order:
            recorded_at, task = started[turn_index]
            result = results.get(turn_index)
            status = ArchivedTurnStatus.IN_PROGRESS
            final_response: str | None = None
            failure_message: str | None = None
            if result is not None and result.get("task") == task:
                phase = result.get("phase")
                if phase == AgentPhase.COMPLETE.value:
                    status = ArchivedTurnStatus.COMPLETE
                elif phase == AgentPhase.FAILED.value:
                    status = ArchivedTurnStatus.FAILED
                raw_final = result.get("final_response")
                if isinstance(raw_final, str) and raw_final.strip():
                    final_response = raw_final
                raw_failure = result.get("failure_message")
                if isinstance(raw_failure, str) and raw_failure.strip():
                    failure_message = raw_failure
            turns.append(
                ArchivedDialogueTurn(
                    session_id=session_id,
                    turn_index=turn_index,
                    recorded_at=recorded_at,
                    task=task,
                    status=status,
                    final_response=final_response,
                    failure_message=failure_message,
                )
            )
        return tuple(turns)


def _message_payload(message: Message) -> dict[str, Any]:
    return {
        "role": message.role.value,
        "content": message.content,
        "tool_calls": [_tool_call_payload(call) for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "reasoning_content": message.reasoning_content,
    }


def _assistant_turn_payload(turn: AssistantTurn) -> dict[str, Any]:
    return {
        "content": turn.content,
        "tool_calls": [_tool_call_payload(call) for call in turn.tool_calls],
        "reasoning_content": turn.reasoning_content,
    }


def _belongs_to_turn(
    payload: Mapping[str, Any],
    turn_index: int | None,
) -> bool:
    if turn_index is None:
        return False
    raw_turn_index = payload.get("turn_index")
    return (
        isinstance(raw_turn_index, int)
        and not isinstance(raw_turn_index, bool)
        and raw_turn_index == turn_index
    )


def _parse_messages(value: Any) -> tuple[Message, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        message
        for item in value
        if (message := _parse_message(item)) is not None
    )


def _conversation_messages(messages: Sequence[Message]) -> tuple[Message, ...]:
    """Normalize legacy archives to persistent history without System messages."""

    return tuple(
        message for message in messages if message.role is not MessageRole.SYSTEM
    )


def _parse_assistant_response(value: Any) -> Message | None:
    if not isinstance(value, dict):
        return None
    return _parse_message(
        {
            "role": MessageRole.ASSISTANT.value,
            "content": value.get("content"),
            "tool_calls": value.get("tool_calls", []),
            "tool_call_id": None,
            "reasoning_content": value.get("reasoning_content"),
        }
    )


def _parse_tool_result_message(value: Any) -> Message | None:
    if not isinstance(value, dict):
        return None
    call_id = value.get("call_id")
    tool_name = value.get("tool_name")
    ok = value.get("ok")
    content = value.get("content")
    error_code = value.get("error_code")
    metadata = value.get("metadata", {})
    if (
        not isinstance(call_id, str)
        or not isinstance(tool_name, str)
        or not isinstance(ok, bool)
        or not isinstance(content, str)
        or (error_code is not None and not isinstance(error_code, str))
        or not isinstance(metadata, Mapping)
    ):
        return None
    try:
        return ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            ok=ok,
            content=content,
            error_code=error_code,
            metadata=metadata,
        ).as_message()
    except (TypeError, ValueError):
        return None


def _tool_call_payload(call: ToolCall) -> dict[str, str]:
    return {
        "id": call.id,
        "name": call.name,
        "arguments_json": call.arguments_json,
    }


def _tool_definition_payload(definition: ToolDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "description": definition.description,
        "parameters_schema": _json_safe(definition.parameters_schema),
    }


def _parse_message(value: Any) -> Message | None:
    if not isinstance(value, dict):
        return None
    try:
        role = MessageRole(value.get("role"))
        raw_calls = value.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            return None
        calls = tuple(
            ToolCall(
                id=item["id"],
                name=item["name"],
                arguments_json=item["arguments_json"],
            )
            for item in raw_calls
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("arguments_json"), str)
        )
        return Message(
            role=role,
            content=value.get("content") if isinstance(value.get("content"), str) else None,
            tool_calls=calls,
            tool_call_id=(
                value.get("tool_call_id")
                if isinstance(value.get("tool_call_id"), str)
                else None
            ),
            reasoning_content=(
                value.get("reasoning_content")
                if isinstance(value.get("reasoning_content"), str)
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("rb") as archive:
            for raw_line in archive:
                try:
                    value = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if (
                    isinstance(value, dict)
                    and value.get("schema_version") == SESSION_SCHEMA_VERSION
                ):
                    records.append(value)
    except OSError as exc:
        raise SessionPersistenceError(
            "could not read a previous session archive"
        ) from exc
    return records


def _parse_context_checkpoint(
    payload: Mapping[str, Any],
) -> ContextCheckpoint | None:
    try:
        return ContextCheckpoint(
            summary=payload["summary"],
            covered_message_count=payload["covered_message_count"],
            source_hash=payload["source_hash"],
        )
    except (KeyError, TypeError, ValueError):
        return None


def _timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass
