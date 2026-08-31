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
        AgentEventKind.COMPLETION_REJECTED,
        model_step=1,
        details={
            "reason": "verification_unsupported",
            "modified_file_count": 2,
        },
    )
    bus.publish(
        AgentEventKind.TOOL_CALLED,
        model_step=1,
        details={
            "tool_name": "read_file",
            "call_id": "call-1",
            "display_path": "src/app.py",
            "display_offset": 0,
        },
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
        "[开始] 正在处理你的任务（本轮最多 20 次模型调用）",
        "[分析] 正在请求模型（第 1 次）",
        "[重试] 模型服务暂时不可用，0.5 秒后进行第 1 次重试",
        "[上下文] 对话较长，已整理早期内容",
        "[检查] 文件已经修改，正在补充验证…",
        "[检查] 当前验证方式未识别，正在改用受支持的验证命令…",
        "[操作] 使用 read_file 读取文件：src/app.py（从字符 0 开始）",
        "[错误] read_file 读取文件失败：目标文件或目录不存在；MiniCoder 将根据结果继续处理",
        "[验证] pytest 测试已通过",
        "[失败] 模型服务请求失败",
    ]


def test_console_sink_omits_success_noise_and_internal_call_ids() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-friendly")

    bus.publish(
        AgentEventKind.TOOL_CALLED,
        model_step=1,
        details={
            "tool_name": "create_file",
            "call_id": "private-call-id",
            "display_path": "created.py",
        },
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

    assert output.getvalue().splitlines() == [
        "[操作] 使用 create_file 创建文件：created.py"
    ]
    assert "private-call-id" not in output.getvalue()


def test_console_sink_shows_command_and_chinese_failure_reason() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-command-error")

    bus.publish(
        AgentEventKind.TOOL_CALLED,
        model_step=2,
        details={
            "tool_name": "run_command",
            "call_id": "call-command",
            "display_command": "python -m pytest -q",
        },
    )
    bus.publish(
        AgentEventKind.TOOL_FINISHED,
        model_step=2,
        details={
            "tool_name": "run_command",
            "call_id": "call-command",
            "ok": False,
            "error_code": "COMMAND_FAILED",
            "exit_code": 2,
        },
    )

    assert output.getvalue().splitlines() == [
        "[操作] 使用 run_command 执行命令：python -m pytest -q",
        "[错误] run_command 运行命令失败：命令返回了非零退出码"
        "（退出码 2）；MiniCoder 将根据结果继续处理",
    ]


def test_console_sink_explains_an_out_of_order_tool_without_claiming_execution() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-plan-order")

    bus.publish(
        AgentEventKind.PLAN_TOOL_REJECTED,
        model_step=3,
        details={
            "call_id": "call-too-early",
            "tool_name": "replace_text",
            "expected_plan_step": 2,
            "attempted_plan_step": 4,
            "plan_item_count": 6,
            "display_path": "README.md",
        },
    )

    assert output.getvalue().splitlines() == [
        "[顺序] 暂未执行 replace_text：当前应先处理计划 2/6，该操作更符合 4/6"
    ]
    assert "call-too-early" not in output.getvalue()


def test_console_sink_renders_planning_and_memory_as_optional_progress() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-plan-memory")

    bus.publish(AgentEventKind.PLANNING_STARTED, model_step=0)
    bus.publish(
        AgentEventKind.MODEL_REQUESTED,
        model_step=1,
        details={"request_kind": "planning"},
    )
    bus.publish(
        AgentEventKind.PLANNING_COMPLETED,
        model_step=1,
        details={
            "plan_item_count": 2,
            "display_plan": "1. 检查项目\n2. 输出结果",
        },
    )
    bus.publish(
        AgentEventKind.PLAN_STEP_STARTED,
        model_step=1,
        details={
            "plan_step": 1,
            "plan_item_count": 2,
            "display_plan_step": "检查项目",
        },
    )
    bus.publish(
        AgentEventKind.PLAN_STEP_COMPLETED,
        model_step=2,
        details={
            "plan_step": 1,
            "plan_item_count": 2,
            "display_plan_step": "检查项目",
        },
    )
    bus.publish(
        AgentEventKind.PLAN_STEPS_UNTRACKED,
        model_step=2,
        details={
            "first_plan_step": 2,
            "last_plan_step": 2,
            "untracked_plan_item_count": 1,
            "plan_item_count": 2,
        },
    )
    bus.publish(
        AgentEventKind.PLAN_COMPLETED,
        model_step=2,
        details={
            "plan_item_count": 2,
            "untracked_plan_item_count": 1,
        },
    )
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
        "[计划] 已生成，共 2 项：",
        "  1. 检查项目",
        "  2. 输出结果",
        "[进行中] 1/2 检查项目",
        "[已完成] 1/2 检查项目",
        "[未关联] 计划 2/2 没有对应到独立的工具操作；不推测其具体完成时间",
        "[计划] 任务已完成，计划进度结束（共 2 项，1 项未单独关联工具操作）",
        "[长期记忆] 已加载这个项目最近的 3 条记录",
        "[记忆] 正在更新会话摘要并判断是否记录长期记忆…",
        "[长期记忆] 已记录一条以后仍有价值的项目信息",
        "[记忆] 本地记忆文件不可用；本次任务结果不受影响",
    ]


def test_console_sink_distinguishes_short_term_restore_and_model_compaction() -> None:
    output = StringIO()
    bus = EventBus((ConsoleEventSink(output),), run_id="run-session-context")

    bus.publish(
        AgentEventKind.SESSION_CONTEXT_LOADED,
        model_step=0,
        details={
            "previous_status": "failed",
            "previous_stop_reason": "max_steps",
            "recent_message_count": 8,
        },
    )
    bus.publish(
        AgentEventKind.CONTEXT_SUMMARY_REQUESTED,
        model_step=4,
        details={"source_chars": 1000, "omitted_message_count": 6},
    )
    bus.publish(
        AgentEventKind.CONTEXT_SUMMARY_FAILED,
        model_step=4,
        details={"reason": "model_error", "fallback_chars": 300},
    )
    bus.publish(
        AgentEventKind.SESSION_ARCHIVE_FAILED,
        model_step=4,
        details={"operation": "append"},
    )

    assert output.getvalue().splitlines() == [
        "[短期记忆] 已恢复同一工作区最近一次会话（状态：failed）",
        "[上下文] 正在让模型压缩较早的对话内容…",
        "[上下文] 模型压缩未成功，已使用基础压缩结果",
        "[短期记忆] 完整会话档案暂时不可用；本次任务仍继续",
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
        details={
            "tool_count": 7,
            "label": "中文",
            "display_private_path": "private/source.py",
        },
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
    assert "private/source.py" not in trace_path.read_text(encoding="utf-8")
    assert records[1]["type"] == "task_completed"


def test_jsonl_sink_requires_an_existing_parent_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="parent directory"):
        JsonlTraceSink(tmp_path / "missing" / "trace.jsonl")
