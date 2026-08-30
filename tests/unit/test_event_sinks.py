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
        "[开始] 正在处理你的任务（本轮最多 20 个步骤）",
        "[分析] 正在规划下一步（步骤 1）",
        "[重试] 模型服务暂时不可用，0.5 秒后进行第 1 次重试",
        "[上下文] 对话较长，已整理早期内容",
        "[检查] 文件已经修改，正在补充验证…",
        "[操作] 正在读取文件…",
        "[注意] 读取文件未成功；结果已返回给 MiniCoder",
        "[验证] pytest 测试已通过",
        "[失败] 模型服务请求失败",
    ]


def test_console_sink_omits_success_noise_and_internal_call_ids() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-friendly")

    bus.publish(
        AgentEventKind.TOOL_CALLED,
        model_step=1,
        details={"tool_name": "create_file", "call_id": "private-call-id"},
    )
    bus.publish(
        AgentEventKind.TOOL_FINISHED,
        model_step=1,
        details={
            "tool_name": "create_file",
            "call_id": "private-call-id",
            "ok": True,
            "error_code": None,
        },
    )

    assert output.getvalue().splitlines() == ["[操作] 正在创建文件…"]
    assert "private-call-id" not in output.getvalue()


def test_console_sink_renders_planning_and_memory_as_optional_progress() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-plan-memory")

    bus.publish(AgentEventKind.PLANNING_STARTED, model_step=0)
    bus.publish(
        AgentEventKind.MODEL_REQUESTED,
        model_step=1,
        details={"request_kind": "planning"},
    )
    bus.publish(AgentEventKind.PLANNING_COMPLETED, model_step=1)
    bus.publish(
        AgentEventKind.MEMORY_LOADED,
        model_step=0,
        details={"record_count": 3},
    )
    bus.publish(AgentEventKind.MEMORY_SUMMARY_REQUESTED, model_step=2)
    bus.publish(AgentEventKind.MEMORY_SUMMARY_COMPLETED, model_step=2)
    bus.publish(AgentEventKind.MEMORY_SAVED, model_step=2)
    bus.publish(
        AgentEventKind.MEMORY_OPERATION_FAILED,
        model_step=2,
        details={"operation": "append"},
    )

    assert output.getvalue().splitlines() == [
        "[计划] 正在制定本轮执行计划…",
        "[计划] 已生成，开始按照计划处理",
        "[记忆] 已加载这个项目最近的 3 条记录",
        "[记忆] 正在整理本轮可供以后参考的项目摘要…",
        "[记忆] 本地记忆文件不可用；本次任务结果不受影响",
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
