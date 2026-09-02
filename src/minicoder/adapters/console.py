"""Concise terminal observer for sanitized agent events."""

from __future__ import annotations

import re
import sys
from typing import TextIO

from minicoder.domain.errors import (
    ConfigurationError,
    DomainValidationError,
    MemoryPersistenceError,
    MiniCoderError,
    ModelAccessError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponseError,
    ModelServiceError,
    ToolRegistrationError,
)
from minicoder.domain.events import AgentEvent, AgentEventKind
from minicoder.domain.state import AgentRunResult, AgentStopReason

_TOOL_ACTIONS = {
    "list_files": "查看项目文件",
    "read_file": "读取文件",
    "search_text": "搜索代码",
    "create_file": "创建文件",
    "write_file": "写入文件",
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

_TOOL_ERROR_MESSAGES = {
    "UNKNOWN_TOOL": "模型请求了不存在的工具",
    "INVALID_ARGUMENTS": "工具参数格式不正确",
    "TOOL_EXECUTION_ERROR": "工具执行时发生内部错误",
    "TOOL_CONTRACT_ERROR": "工具返回了不符合约定的结果",
    "INVALID_PATH": "路径格式无效",
    "PATH_OUTSIDE_WORKSPACE": "路径超出了允许访问的工作目录",
    "BINARY_FILE": "目标文件不是可处理的 UTF-8 文本文件",
    "FILE_ALREADY_EXISTS": "目标文件已经存在",
    "FILE_CONTENT_MISMATCH": "文件内容已经变化，请重新读取后再写入",
    "FILE_IO_ERROR": "读写文件时发生系统错误",
    "FILE_NOT_FOUND": "目标文件或目录不存在",
    "FILE_TOO_LARGE": "目标文件超过了允许处理的大小",
    "INVALID_OFFSET": "读取位置超出了有效范围",
    "NO_CHANGES": "替换内容没有产生变化",
    "NOT_A_DIRECTORY": "目标路径不是目录",
    "NOT_A_FILE": "目标路径不是普通文件",
    "PARENT_DIRECTORY_NOT_FOUND": "目标文件的父目录不存在",
    "PERMISSION_DENIED": "当前用户没有足够的文件访问权限",
    "TEXT_NOT_FOUND": "没有找到需要替换的原文字",
    "TEXT_NOT_UNIQUE": "需要替换的原文字出现了多次，无法确定唯一位置",
    "COMMAND_FAILED": "命令返回了非零退出码",
    "COMMAND_LAUNCH_FAILED": "命令无法启动",
    "COMMAND_NOT_FOUND": "系统中找不到这个命令",
    "COMMAND_PERMISSION_DENIED": "当前用户没有执行这个命令的权限",
    "COMMAND_TIMED_OUT": "命令执行超时",
    "OUTPUT_STORAGE_FAILED": "完整命令输出无法安全保存",
    "COMMAND_REJECTED": "命令被安全策略拒绝",
    "INVALID_COMMAND": "命令参数无效",
    "ARTIFACT_NOT_FOUND": "找不到对应的完整命令输出",
    "ARTIFACT_STORE_CLOSED": "本次会话的命令输出存储已经关闭",
    "INVALID_ARTIFACT_RANGE": "读取命令输出的位置无效",
}
_REQUIRED_ENV_ERROR = re.compile(r"^(MINICODER_[A-Z0-9_]+) is required$")


class ConsoleEventSink:
    """Render concise user-facing progress without internal protocol details."""

    def __init__(
        self,
        output: TextIO | None = None,
        *,
        defer_recovery_messages: bool = False,
    ) -> None:
        self._output = sys.stdout if output is None else output
        self._defer_recovery_messages = defer_recovery_messages
        self._deferred_recovery_messages: list[str] = []

    def handle(self, event: AgentEvent) -> None:
        message = _format_event(event)
        if message is None:
            return
        if self._defer_recovery_messages and event.kind in {
            AgentEventKind.SESSION_CONTEXT_LOADED,
            AgentEventKind.MEMORY_LOADED,
        }:
            self._deferred_recovery_messages.append(message)
            return
        print(message, file=self._output, flush=True)

    def flush_recovery_messages(self) -> None:
        """Print deferred startup recovery details after dialogue replay."""

        for message in self._deferred_recovery_messages:
            print(message, file=self._output, flush=True)
        self._deferred_recovery_messages.clear()


def _format_event(event: AgentEvent) -> str | None:
    details = event.details
    if event.kind is AgentEventKind.TASK_STARTED:
        return (
            "[开始] 正在处理你的任务"
            f"（本轮最多 {details['max_steps']} 次模型调用）"
        )
    if event.kind is AgentEventKind.PLANNING_STARTED:
        return "[计划] 正在制定本轮执行计划…"
    if event.kind is AgentEventKind.PLANNING_RETRY_REQUESTED:
        attempt = int(details["attempt"])
        maximum = int(details["maximum"])
        return (
            "[计划] 返回格式无法识别，正在要求模型重新生成"
            f"（第 {attempt}/{maximum} 次重试）…"
        )
    if event.kind is AgentEventKind.PLANNING_COMPLETED:
        count = int(details.get("plan_item_count", 0))
        plan = str(details.get("display_plan", "")).strip()
        if plan:
            indented = "\n".join(f"  {line}" for line in plan.splitlines())
            return f"[计划] 已生成，共 {count} 项：\n{indented}"
        return "[计划] 已生成，开始按照计划处理"
    if event.kind is AgentEventKind.PLAN_STEP_STARTED:
        index = int(details["plan_step"])
        count = int(details["plan_item_count"])
        step = str(details.get("display_plan_step", "当前步骤"))
        return f"[进行中] {index}/{count} {step}"
    if event.kind is AgentEventKind.PLAN_STEP_COMPLETED:
        index = int(details["plan_step"])
        count = int(details["plan_item_count"])
        step = str(details.get("display_plan_step", "当前步骤"))
        return f"[已完成] {index}/{count} {step}"
    if event.kind is AgentEventKind.PLAN_STEP_REPORT_REQUIRED:
        index = int(details["plan_step"])
        count = int(details["plan_item_count"])
        return (
            f"[计划] 模型尚未确认第 {index}/{count} 项完成，"
            "继续处理当前项"
        )
    if event.kind is AgentEventKind.PLAN_COMPLETED:
        count = int(details["plan_item_count"])
        return f"[计划] 全部 {count} 项已完成"
    if event.kind is AgentEventKind.MODEL_REQUESTED:
        if details.get("request_kind") == "planning":
            return None
        return f"[分析] 正在请求模型（第 {event.model_step} 次）"
    if event.kind is AgentEventKind.MEMORY_LOADED:
        count = int(details.get("record_count", 0))
        return f"[长期记忆] 已加载这个项目全部 {count} 条有效记录"
    if event.kind is AgentEventKind.SESSION_CONTEXT_LOADED:
        status = str(details.get("previous_status", "unknown"))
        return f"[上下文] 已从最近的会话档案恢复连续对话上下文（状态：{status}）"
    if event.kind is AgentEventKind.MEMORY_SUMMARY_REQUESTED:
        return "[长期记忆] 正在严格判断本轮是否产生了真正有价值的新记忆…"
    if event.kind is AgentEventKind.MEMORY_SUMMARY_COMPLETED:
        return None
    if event.kind is AgentEventKind.MEMORY_SUMMARY_FAILED:
        return "[长期记忆] 本轮未新增记录；完整会话仍已保存在本地档案中"
    if event.kind is AgentEventKind.MEMORY_SAVED:
        return "[长期记忆] 已记录一条以后仍有价值的项目信息"
    if event.kind is AgentEventKind.MEMORY_OPERATION_FAILED:
        return "[记忆] 本地记忆文件不可用；本次任务结果不受影响"
    if event.kind is AgentEventKind.SESSION_ARCHIVE_FAILED:
        return "[短期记忆] 完整会话档案暂时不可用；本次任务仍继续"
    if event.kind is AgentEventKind.MODEL_RETRY_SCHEDULED:
        return (
            f"[重试] 模型服务暂时不可用，{details['delay_seconds']} 秒后"
            f"进行第 {details['retry_number']} 次重试"
        )
    if event.kind is AgentEventKind.CONTEXT_COMPACTED:
        return "[上下文] 对话较长，已整理早期内容"
    if event.kind is AgentEventKind.CONTEXT_SUMMARY_REQUESTED:
        return "[上下文] 正在让模型压缩较早的对话内容…"
    if event.kind is AgentEventKind.CONTEXT_SUMMARY_COMPLETED:
        return None
    if event.kind is AgentEventKind.CONTEXT_SUMMARY_FAILED:
        return "[上下文] 模型压缩未成功，已使用基础压缩结果"
    if event.kind is AgentEventKind.COMPLETION_REJECTED:
        return _completion_message(str(details["reason"]))
    if event.kind is AgentEventKind.TOOL_CALLED:
        return _tool_call_message(event)
    if event.kind is AgentEventKind.TOOL_FINISHED:
        if details["ok"]:
            return None
        return _tool_failure_message(event)
    if event.kind is AgentEventKind.VERIFICATION_PASSED:
        kind = str(details["verification_kind"])
        label = _VERIFICATION_LABELS.get(kind, kind)
        return f"[验证] {label}已通过"
    if event.kind is AgentEventKind.TASK_COMPLETED:
        return f"[完成] 任务已完成（共 {event.model_step} 次模型调用）"
    if event.kind is AgentEventKind.TASK_FAILED:
        return _failure_message(str(details["reason"]), event.model_step)
    return None


def _tool_action(tool_name: str) -> str:
    return _TOOL_ACTIONS.get(tool_name, "执行操作")


def _tool_call_message(event: AgentEvent) -> str:
    details = event.details
    tool_name = str(details["tool_name"])
    path = str(details.get("display_path", ""))
    if tool_name == "list_files" and path:
        return f"[操作] 使用 list_files 查看目录：{path}"
    if tool_name == "read_file" and path:
        offset = int(details.get("display_offset", 0))
        return f"[操作] 使用 read_file 读取文件：{path}（从字符 {offset} 开始）"
    if tool_name == "search_text":
        query = str(details.get("display_query", ""))
        location = path or "."
        if query:
            return f"[操作] 使用 search_text 在 {location} 中搜索：{query!r}"
    if tool_name == "create_file" and path:
        return f"[操作] 使用 create_file 创建文件：{path}"
    if tool_name == "write_file" and path:
        return f"[操作] 使用 write_file 写入文件：{path}"
    if tool_name == "replace_text" and path:
        return f"[操作] 使用 replace_text 修改文件：{path}"
    if tool_name == "run_command":
        command = str(details.get("display_command", ""))
        if command:
            return f"[操作] 使用 run_command 执行命令：{command}"
    if tool_name == "read_tool_output":
        offset = int(details.get("display_offset", 0))
        return f"[操作] 使用 read_tool_output 继续读取命令输出（从字符 {offset} 开始）"
    return f"[操作] 使用 {tool_name} {_tool_action(tool_name)}…"


def _tool_failure_message(event: AgentEvent) -> str:
    details = event.details
    tool_name = str(details["tool_name"])
    error_code = str(details.get("error_code") or "")
    explanation = _TOOL_ERROR_MESSAGES.get(error_code, "工具操作未成功")
    exit_code = details.get("exit_code")
    if (
        error_code == "COMMAND_FAILED"
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
    ):
        explanation = f"{explanation}（退出码 {exit_code}）"
    return (
        f"[错误] {tool_name} {_tool_action(tool_name)}失败：{explanation}；"
        "MiniCoder 将根据结果继续处理"
    )


def _completion_message(reason: str) -> str:
    if reason == "verification_required":
        return "[检查] 文件已经修改，正在补充验证…"
    if reason == "verification_failed":
        return "[检查] 验证未通过，正在继续修复…"
    if reason == "verification_unsupported":
        return "[检查] 当前验证方式未识别，正在改用受支持的验证命令…"
    return "[检查] 当前结果尚未满足完成条件，正在继续处理…"


def _failure_message(reason: str, model_step: int) -> str:
    if reason == "max_steps":
        return f"[未完成] 已达到本轮模型调用上限（共 {model_step} 次）"
    if reason == "model_error":
        return "[失败] 模型服务请求失败"
    if reason == "planning_error":
        return "[失败] 模型没有返回可执行的计划"
    if reason == "context_budget_exceeded":
        return "[未完成] 当前请求的固定内容已经超过上下文预算"
    if reason == "user_interrupted":
        return "[停止] 用户已中断任务"
    if reason == "verification_unsupported":
        return "[未完成] 当前验证命令尚未被 MiniCoder 识别"
    return "[失败] 任务未能完成"


def format_agent_failure(result: AgentRunResult) -> str:
    """Return one actionable Chinese explanation for a terminal Agent result."""

    if result.stop_reason is AgentStopReason.MAX_STEPS:
        return (
            "MiniCoder 未完成任务：已达到本轮模型调用上限"
            f"（共 {result.model_steps} 次）。可以继续发送消息，或调整 "
            "MINICODER_MAX_STEPS。"
        )
    if result.stop_reason is AgentStopReason.MODEL_ERROR:
        return (
            "MiniCoder 未完成任务：模型服务请求失败。请检查 API Key、Base URL、"
            "网络连接、账户额度和模型权限。"
        )
    if result.stop_reason is AgentStopReason.PLANNING_ERROR:
        return (
            "MiniCoder 未完成任务：模型没有返回有效的纯文本计划，"
            "因此没有执行规划阶段请求的工具。请重新尝试。"
        )
    if result.stop_reason is AgentStopReason.CONTEXT_BUDGET_EXCEEDED:
        return (
            "MiniCoder 未发送模型请求：System、短期/长期记忆、当前输入、"
            "工具定义和回复预留已经超过上下文预算。请缩短当前输入、"
            "减少工具，或提高 "
            "MINICODER_CONTEXT_BUDGET_CHARS。"
        )
    if result.stop_reason is AgentStopReason.VERIFICATION_UNSUPPORTED:
        return (
            "MiniCoder 未完成任务：当前验证命令未被识别。请改用受支持的测试、"
            "编译或静态检查命令，或在 .minicoder.toml 中配置项目验证命令。"
        )
    if result.stop_reason is AgentStopReason.USER_INTERRUPTED:
        return "MiniCoder 已停止：任务被用户中断。"
    return "MiniCoder 未完成任务：运行过程中出现了未分类错误。"


def format_minicoder_error(error: MiniCoderError) -> str:
    """Translate exception categories into concise Chinese CLI guidance."""

    if isinstance(error, ModelAccessError):
        return "模型认证或权限校验失败。请检查 API Key、模型名称和账户权限。"
    if isinstance(error, ModelRateLimitError):
        return "模型服务当前限流。请稍后重试，或检查账户调用额度。"
    if isinstance(error, ModelConnectionError):
        return "无法连接模型服务。请检查网络连接和 Base URL。"
    if isinstance(error, ModelServiceError):
        return "模型服务暂时异常。请稍后重试。"
    if isinstance(error, ModelRequestError):
        return "模型请求被拒绝。请检查模型名称、接口兼容性和请求配置。"
    if isinstance(error, ModelResponseError):
        return "模型返回格式不符合工具调用协议。请重试或更换兼容模型。"
    if isinstance(error, ConfigurationError):
        return _configuration_error_message(str(error))
    if isinstance(error, MemoryPersistenceError):
        return "项目记忆文件无法读取或保存；请检查本地目录权限和磁盘空间。"
    if isinstance(error, ToolRegistrationError):
        return "本地工具初始化失败；工具名称或参数定义可能存在冲突。"
    if isinstance(error, DomainValidationError):
        return "MiniCoder 收到了不符合内部约定的数据，请检查输入后重试。"
    return "MiniCoder 运行失败，请检查配置并重试。"


def _configuration_error_message(message: str) -> str:
    required = _REQUIRED_ENV_ERROR.fullmatch(message)
    if required is not None:
        return f"配置错误：缺少必需的环境变量 {required.group(1)}。"
    if message.startswith("workspace does not exist: "):
        path = message.removeprefix("workspace does not exist: ")
        return f"配置错误：工作目录不存在：{path}"
    if message.startswith("workspace is not a directory: "):
        path = message.removeprefix("workspace is not a directory: ")
        return f"配置错误：指定的工作路径不是目录：{path}"
    if "verification.commands must be an array" in message:
        return "配置错误：.minicoder.toml 中的 verification.commands 必须是数组。"
    detail = " ".join(message.split())
    if len(detail) > 240:
        detail = f"{detail[:239]}…"
    return f"配置错误：启动配置缺失或无效。技术详情：{detail}"
