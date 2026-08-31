"""Bounded user-facing progress facts derived from model plans and tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.events import EventDetail
from minicoder.domain.models import ToolCall

_MAX_PLAN_ITEMS = 7
_MAX_PLAN_ITEM_CHARS = 240
_MAX_DISPLAY_PATH_CHARS = 300
_MAX_DISPLAY_QUERY_CHARS = 120
_MAX_DISPLAY_ARGUMENT_CHARS = 120
_MAX_DISPLAY_COMMAND_CHARS = 500

_NUMBERED_PLAN_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(\d{1,2})[.)、:：]\s*(.+?)\s*$"
)
_BULLET_PLAN_LINE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")
_SAFE_COMMAND_ARGUMENT = re.compile(r"[A-Za-z0-9_./:@%+=,-]+")
_ANSI_ESCAPE_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TERMINAL_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

_INSPECTION_KEYWORDS = (
    "inspect",
    "read",
    "review",
    "analy",
    "locate",
    "search",
    "understand",
    "explore",
    "examine",
    "检查",
    "读取",
    "查看",
    "分析",
    "定位",
    "搜索",
    "理解",
)
_MUTATION_KEYWORDS = (
    "implement",
    "modify",
    "update",
    "create",
    "write",
    "fix",
    "add",
    "change",
    "refactor",
    "实现",
    "修改",
    "更新",
    "创建",
    "编写",
    "修复",
    "新增",
    "重构",
)
_VERIFICATION_KEYWORDS = (
    "verify",
    "test",
    "compile",
    "build",
    "lint",
    "check",
    "验证",
    "测试",
    "编译",
    "构建",
    "检查结果",
)
_TOOL_KEYWORDS = {
    "list_files": _INSPECTION_KEYWORDS,
    "read_file": _INSPECTION_KEYWORDS,
    "search_text": _INSPECTION_KEYWORDS,
    "read_tool_output": _INSPECTION_KEYWORDS,
    "create_file": _MUTATION_KEYWORDS,
    "replace_text": _MUTATION_KEYWORDS,
}
_INSPECTION_COMMANDS = frozenset(
    {"cat", "find", "git", "grep", "head", "ls", "pwd", "rg", "sed", "tail"}
)
_VERIFICATION_COMMANDS = frozenset(
    {
        "c++",
        "cc",
        "clang",
        "clang++",
        "cl",
        "ctest",
        "g++",
        "gcc",
        "ninja",
        "pytest",
    }
)
_VERIFICATION_COMMAND_WORDS = frozenset(
    {
        "build",
        "check",
        "compileall",
        "ctest",
        "lint",
        "ninja",
        "py_compile",
        "pytest",
        "test",
        "tests",
        "unittest",
        "verify",
    }
)
_SENSITIVE_LABEL_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One bounded plan item selected for user-facing progress."""

    index: int  # One-based item index.
    total: int  # Total number of displayed plan items.
    text: str  # Bounded plan item text.


@dataclass(frozen=True, slots=True)
class PlanStepUpdate:
    """One ordered plan-step status change emitted to observers."""

    step: PlanStep  # Plan item whose status changed.
    completed: bool  # False when starting; True when finishing the item.


@dataclass(slots=True)
class PlanProgress:
    """Track the currently displayed item of one model-generated plan."""

    items: tuple[str, ...]
    current_index: int = 0

    def __post_init__(self) -> None:
        if (
            not self.items
            or len(self.items) > _MAX_PLAN_ITEMS
            or any(
                not isinstance(item, str) or not item.strip()
                for item in self.items
            )
        ):
            raise DomainValidationError(
                "plan progress requires one to seven non-blank text items"
            )
        if self.current_index < 0 or self.current_index > len(self.items):
            raise DomainValidationError("plan current index is outside the plan")

    @classmethod
    def from_model_text(cls, text: str) -> PlanProgress:
        """Parse a numbered or bulleted model plan without another model call."""

        return cls(items=_parse_plan_items(text))

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def display_text(self) -> str:
        return "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(self.items, start=1)
        )

    def begin(self) -> PlanStepUpdate:
        """Start the first plan item after planning completes."""

        self.current_index = 1
        return PlanStepUpdate(step=self._step(1), completed=False)

    def advance_for_tool(
        self,
        call: ToolCall,
        *,
        explicit_step: int | None,
    ) -> tuple[PlanStepUpdate, ...]:
        """Select a plan item from an explicit marker or a deterministic fallback."""

        if explicit_step is not None and 1 <= explicit_step <= self.total:
            return self._transition_to(explicit_step)

        keywords = _keywords_for_tool(call)
        inferred = _matching_step(
            self.items,
            keywords,
            start=max(self.current_index, 1),
        )
        if inferred is None:
            inferred = max(self.current_index, 1)
        return self._transition_to(inferred)

    def finish(self) -> tuple[PlanStepUpdate, ...]:
        """Complete the active item; PLAN_COMPLETED closes the whole plan."""

        if self.current_index == 0:
            self.begin()
        update = PlanStepUpdate(
            step=self._step(self.current_index),
            completed=True,
        )
        self.current_index = self.total
        return (update,)

    def _transition_to(self, index: int) -> tuple[PlanStepUpdate, ...]:
        if index <= self.current_index:
            return ()
        updates: list[PlanStepUpdate] = [
            PlanStepUpdate(
                step=self._step(self.current_index),
                completed=True,
            )
        ]
        for intermediate in range(self.current_index + 1, index):
            step = self._step(intermediate)
            updates.append(PlanStepUpdate(step=step, completed=False))
            updates.append(PlanStepUpdate(step=step, completed=True))
        self.current_index = index
        updates.append(PlanStepUpdate(step=self._step(index), completed=False))
        return tuple(updates)

    def _step(self, index: int) -> PlanStep:
        return PlanStep(index=index, total=self.total, text=self.items[index - 1])


def tool_display_details(call: ToolCall) -> dict[str, EventDetail]:
    """Return bounded display-only arguments without file bodies or replacement text."""

    arguments = _json_object(call.arguments_json)
    if arguments is None:
        return {}

    details: dict[str, EventDetail] = {}
    if call.name == "list_files":
        details["display_path"] = _bounded_text(
            arguments.get("path", "."),
            _MAX_DISPLAY_PATH_CHARS,
        )
    elif call.name in {"read_file", "create_file", "replace_text"}:
        path = _bounded_text(arguments.get("path"), _MAX_DISPLAY_PATH_CHARS)
        if path:
            details["display_path"] = path
        if call.name == "read_file":
            offset = arguments.get("offset", 0)
            if isinstance(offset, int) and not isinstance(offset, bool):
                details["display_offset"] = offset
    elif call.name == "search_text":
        details["display_path"] = _bounded_text(
            arguments.get("path", "."),
            _MAX_DISPLAY_PATH_CHARS,
        )
        query = _bounded_text(
            arguments.get("query"),
            _MAX_DISPLAY_QUERY_CHARS,
        )
        if query:
            details["display_query"] = query
    elif call.name == "run_command":
        command = _display_command(arguments.get("argv"))
        if command:
            details["display_command"] = command
    elif call.name == "read_tool_output":
        offset = arguments.get("offset", 0)
        if isinstance(offset, int) and not isinstance(offset, bool):
            details["display_offset"] = offset
    return details


def _parse_plan_items(text: str) -> tuple[str, ...]:
    numbered: list[str] = []
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _NUMBERED_PLAN_LINE.match(line)
        if match is not None:
            if current is not None:
                numbered.append(_clean_plan_item(current))
            current = match.group(2)
            if len(numbered) >= _MAX_PLAN_ITEMS:
                break
            continue
        if current is not None:
            current = f"{current} {line}"
    if current is not None and len(numbered) < _MAX_PLAN_ITEMS:
        numbered.append(_clean_plan_item(current))
    if numbered:
        return tuple(numbered[:_MAX_PLAN_ITEMS])

    fallback: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = _BULLET_PLAN_LINE.match(line)
        fallback.append(_clean_plan_item(bullet.group(1) if bullet else line))
        if len(fallback) == _MAX_PLAN_ITEMS:
            break
    if fallback:
        return tuple(fallback)
    return ("执行当前任务",)


def _clean_plan_item(text: str) -> str:
    compact = _remove_terminal_controls(" ".join(text.split())).strip(" *_`")
    return _truncate(compact or "执行当前步骤", _MAX_PLAN_ITEM_CHARS)


def _matching_step(
    items: tuple[str, ...],
    keywords: tuple[str, ...],
    *,
    start: int,
) -> int | None:
    if not keywords:
        return None
    best_index: int | None = None
    best_score = 0
    for index in range(start, len(items) + 1):
        folded = items[index - 1].casefold()
        score = sum(keyword in folded for keyword in keywords)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _keywords_for_tool(call: ToolCall) -> tuple[str, ...]:
    if call.name != "run_command":
        return _TOOL_KEYWORDS.get(call.name, ())

    arguments = _json_object(call.arguments_json)
    argv = None if arguments is None else arguments.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(argument, str) for argument in argv
    ):
        return ()
    program = argv[0].replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if program.endswith(suffix):
            program = program[: -len(suffix)]
            break
    if program in _INSPECTION_COMMANDS:
        return _INSPECTION_KEYWORDS
    if program in _VERIFICATION_COMMANDS:
        return _VERIFICATION_KEYWORDS
    words = {
        word
        for argument in argv
        for word in re.split(r"[^a-z0-9_+.-]+", argument.casefold())
        if word
    }
    if words & _VERIFICATION_COMMAND_WORDS:
        return _VERIFICATION_KEYWORDS
    return ()


def _json_object(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _display_command(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return ""
    if any(not isinstance(argument, str) for argument in value):
        return ""

    displayed: list[str] = []
    hide_next = False
    for raw_argument in value:
        if hide_next:
            argument = "<redacted>"
            hide_next = False
        else:
            argument, hide_next = _redact_argument(raw_argument)
        argument = _truncate(argument, _MAX_DISPLAY_ARGUMENT_CHARS)
        displayed.append(_quote_argument(argument))
    return _truncate(" ".join(displayed), _MAX_DISPLAY_COMMAND_CHARS)


def _redact_argument(argument: str) -> tuple[str, bool]:
    folded = argument.casefold()
    normalized_label = folded.lstrip("-/").replace("-", "_")
    if any(part == normalized_label for part in _SENSITIVE_LABEL_PARTS):
        return argument, True

    for separator in ("=", ":"):
        if separator not in argument:
            continue
        label, _, _ = argument.partition(separator)
        normalized = label.casefold().lstrip("-/").replace("-", "_")
        if any(part in normalized for part in _SENSITIVE_LABEL_PARTS):
            return f"{label}{separator}<redacted>", False

    if (folded.startswith("sk-") and len(argument) >= 16) or folded.startswith(
        "bearer "
    ):
        return "<redacted>", False
    return argument, False


def _quote_argument(argument: str) -> str:
    if _SAFE_COMMAND_ARGUMENT.fullmatch(argument):
        return argument
    return json.dumps(argument, ensure_ascii=False)


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    compact = _remove_terminal_controls(" ".join(value.split()))
    return _truncate(compact, limit)


def _remove_terminal_controls(text: str) -> str:
    without_ansi = _ANSI_ESCAPE_SEQUENCE.sub("", text)
    return _TERMINAL_CONTROL_CHARACTERS.sub("", without_ansi)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"
