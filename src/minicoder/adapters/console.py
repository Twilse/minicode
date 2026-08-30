"""Concise terminal observer for sanitized agent events."""

from __future__ import annotations

import sys
from typing import TextIO

from minicoder.domain.events import AgentEvent, AgentEventKind

_TOOL_ACTIONS = {
    "list_files": "查看项目文件",
    "read_file": "读取文件",
    "search_text": "搜索代码",
    "create_file": "创建文件",
    "replace_text": "修改文件",
    "run_command": "运行命令",
    "read_tool_output": "读取完整命令输出",
}

_VERIFICATION_LABELS = {
    "C/C++ compiler": "C/C++ 编译检查",
    "cmake build": "CMake 构建检查",
    "configured verifier": "项目自定义检查",
    "ctest": "CTest 测试",
    "ninja": "Ninja 构建检查",
    "pytest": "pytest 测试",
}


class ConsoleEventSink:
    """Render concise user-facing progress without internal protocol details."""

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = sys.stdout if output is None else output

    def handle(self, event: AgentEvent) -> None:
        message = _format_event(event)
        if message is not None:
            print(message, file=self._output, flush=True)


def _format_event(event: AgentEvent) -> str | None:
    details = event.details
    if event.kind is AgentEventKind.TASK_STARTED:
        return (
            "[开始] 正在处理你的任务"
            f"（本轮最多 {details['max_steps']} 个步骤）"
        )
    if event.kind is AgentEventKind.PLANNING_STARTED:
        return "[计划] 正在制定本轮执行计划…"
    if event.kind is AgentEventKind.PLANNING_COMPLETED:
        return "[计划] 已生成，开始按照计划处理"
    if event.kind is AgentEventKind.MODEL_REQUESTED:
        if details.get("request_kind") == "planning":
            return None
        return f"[分析] 正在规划下一步（步骤 {event.model_step}）"
    if event.kind is AgentEventKind.MEMORY_LOADED:
        count = int(details.get("record_count", 0))
        return f"[记忆] 已加载这个项目最近的 {count} 条记录"
    if event.kind is AgentEventKind.MEMORY_SUMMARY_REQUESTED:
        return "[记忆] 正在整理本轮可供以后参考的项目摘要…"
    if event.kind is AgentEventKind.MEMORY_SUMMARY_COMPLETED:
        return None
    if event.kind is AgentEventKind.MEMORY_SUMMARY_FAILED:
        return "[记忆] 模型总结未成功，已改用基础摘要"
    if event.kind is AgentEventKind.MEMORY_SAVED:
        return None
    if event.kind is AgentEventKind.MEMORY_OPERATION_FAILED:
        return "[记忆] 本地记忆文件不可用；本次任务结果不受影响"
    if event.kind is AgentEventKind.MODEL_RETRY_SCHEDULED:
        return (
            f"[重试] 模型服务暂时不可用，{details['delay_seconds']} 秒后"
            f"进行第 {details['retry_number']} 次重试"
        )
    if event.kind is AgentEventKind.CONTEXT_COMPACTED:
        return "[上下文] 对话较长，已整理早期内容"
    if event.kind is AgentEventKind.COMPLETION_REJECTED:
        return _completion_message(str(details["reason"]))
    if event.kind is AgentEventKind.TOOL_CALLED:
        action = _tool_action(str(details["tool_name"]))
        return f"[操作] 正在{action}…"
    if event.kind is AgentEventKind.TOOL_FINISHED:
        if details["ok"]:
            return None
        action = _tool_action(str(details["tool_name"]))
        return f"[注意] {action}未成功；结果已返回给 MiniCoder"
    if event.kind is AgentEventKind.VERIFICATION_PASSED:
        kind = str(details["verification_kind"])
        label = _VERIFICATION_LABELS.get(kind, kind)
        return f"[验证] {label}已通过"
    if event.kind is AgentEventKind.TASK_COMPLETED:
        return f"[完成] 任务已完成（共 {event.model_step} 个步骤）"
    if event.kind is AgentEventKind.TASK_FAILED:
        return _failure_message(str(details["reason"]), event.model_step)
    return None


def _tool_action(tool_name: str) -> str:
    return _TOOL_ACTIONS.get(tool_name, "执行操作")


def _completion_message(reason: str) -> str:
    if reason == "verification_required":
        return "[检查] 文件已经修改，正在补充验证…"
    if reason == "verification_failed":
        return "[检查] 验证未通过，正在继续修复…"
    return "[检查] 当前结果尚未满足完成条件，正在继续处理…"


def _failure_message(reason: str, model_step: int) -> str:
    if reason == "max_steps":
        return f"[未完成] 已达到本轮最大处理步骤（{model_step} 步）"
    if reason == "model_error":
        return "[失败] 模型服务请求失败"
    if reason == "planning_error":
        return "[失败] 模型没有返回可执行的计划"
    if reason == "user_interrupted":
        return "[停止] 用户已中断任务"
    if reason == "verification_unsupported":
        return "[未完成] 当前验证命令尚未被 MiniCoder 识别"
    return "[失败] 任务未能完成"
