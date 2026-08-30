import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pytest

from minicoder.adapters.console import ConsoleEventSink
from minicoder.adapters.jsonl_trace import JsonlTraceSink
from minicoder.application.event_bus import EventBus
from minicoder.domain.events import AgentEventKind

FIXED_TIME = datetime(
    2026,
    8,
    30,
    16,
    45,
    12,
    345000,
    tzinfo=timezone(timedelta(hours=8)),
)


def test_console_sink_renders_concise_event_lines() -> None:
    output = StringIO()
    bus = EventBus(
        (ConsoleEventSink(output),),
        run_id="run-console",
        clock=lambda: FIXED_TIME,
    )

    bus.publish(
        AgentEventKind.TASK_STARTED,
        model_step=0,
        details={"tool_count": 7, "max_steps": 20},
    )
    bus.publish(
        AgentEventKind.MODEL_REQUESTED,
        model_step=1,
        details={"message_count": 2},
    )
    bus.publish(
        AgentEventKind.MODEL_RETRY_SCHEDULED,
        model_step=1,
        details={
            "retry_number": 1,
            "error_type": "ModelConnectionError",
            "delay_seconds": 0.5,
        },
    )
    bus.publish(
        AgentEventKind.CONTEXT_COMPACTED,
        model_step=1,
        details={
            "original_chars": 2_000,
            "prepared_chars": 800,
            "omitted_message_count": 4,
        },
    )
    bus.publish(
        AgentEventKind.COMPLETION_REJECTED,
        model_step=1,
        details={
            "reason": "verification_required",
            "modified_file_count": 2,
        },
    )
    bus.publish(
        AgentEventKind.TOOL_CALLED,
        model_step=1,
        details={"tool_name": "read_file", "call_id": "call-1"},
    )
    bus.publish(
        AgentEventKind.TOOL_FINISHED,
        model_step=1,
        details={
            "tool_name": "read_file",
            "call_id": "call-1",
            "ok": False,
            "error_code": "FILE_NOT_FOUND",
        },
    )
    bus.publish(
        AgentEventKind.VERIFICATION_PASSED,
        model_step=1,
        details={"verification_kind": "pytest"},
    )
    bus.publish(
        AgentEventKind.TASK_FAILED,
        model_step=1,
        details={"reason": "model_error", "message": "network unavailable"},
    )

    assert output.getvalue().splitlines() == [
        "[TASK] started (7 tools, max 20 model steps)",
        "[MODEL] step 1 requested (2 messages)",
        "[MODEL] retry 1 after ModelConnectionError (0.5s)",
        "[CONTEXT] compacted 2000 -> 800 chars (4 messages omitted)",
        "[REVIEW] final response rejected (verification_required; 2 modified files)",
        "[TOOL] read_file called (call_id=call-1)",
        "[TOOL] read_file failed (FILE_NOT_FOUND) (call_id=call-1)",
        "[VERIFY] pytest passed",
        "[FAILED] model_error after 1 model steps: network unavailable",
    ]


def test_jsonl_sink_appends_parseable_versioned_records(tmp_path: Path) -> None:
    trace_path = tmp_path / "agent.jsonl"
    sink = JsonlTraceSink(trace_path)
    bus = EventBus(
        (sink,),
        run_id="run-jsonl",
        clock=lambda: FIXED_TIME,
    )

    bus.publish(
        AgentEventKind.TASK_STARTED,
        model_step=0,
        details={"tool_count": 7, "label": "中文"},
    )
    bus.publish(
        AgentEventKind.TASK_COMPLETED,
        model_step=2,
        details={"response_chars": 12},
    )

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["sequence"] for record in records] == [1, 2]
    assert records[0] == {
        "schema_version": 1,
        "run_id": "run-jsonl",
        "sequence": 1,
        "timestamp": "2026-08-30T08:45:12.345Z",
        "type": "task_started",
        "model_step": 0,
        "details": {"tool_count": 7, "label": "中文"},
    }
    assert records[1]["type"] == "task_completed"


def test_jsonl_sink_requires_an_existing_parent_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent directory"):
        JsonlTraceSink(tmp_path / "missing" / "trace.jsonl")
