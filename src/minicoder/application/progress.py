"""Bounded user-facing progress facts derived from model plans and tool calls."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
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
    "增加",
    "接入",
    "补充",
    "重构",
)
_TEST_AUTHORING_KEYWORDS = (
    "test",
    "tests",
    "testing",
    "测试",
    "用例",
)
_DOCUMENTATION_KEYWORDS = (
    "document",
    "documentation",
    "readme",
    "文档",
    "说明",
    "记录",
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


@dataclass(frozen=True, slots=True)
class PlanTransition:
    """One evidence-based plan movement produced by a real agent activity."""

    updates: tuple[PlanStepUpdate, ...] = ()  # Honest start/complete changes.
    untracked: tuple[PlanStep, ...] = ()  # Items lacking an individual activity.


@dataclass(frozen=True, slots=True)
class _ToolActivity:
    """Deterministic action and target facts derived from one ToolCall."""

    action_keywords: tuple[str, ...]
    target_keywords: tuple[str, ...] = ()

    @property
    def keywords(self) -> tuple[str, ...]:
        return _unique(self.action_keywords + self.target_keywords)


@dataclass(slots=True)
class PlanProgress:
    """Track the currently displayed item of one model-generated plan."""

    items: tuple[str, ...]
    current_index: int = 0
    _current_has_activity: bool = field(default=False, init=False, repr=False)
    _untracked_indices: set[int] = field(
        default_factory=set,
        init=False,
        repr=False,
    )

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

    @property
    def untracked_count(self) -> int:
        return len(self._untracked_indices)

    def begin(self) -> PlanStepUpdate:
        """Start the first plan item after planning completes."""

        self.current_index = 1
        self._current_has_activity = False
        return PlanStepUpdate(step=self._step(1), completed=False)

    def advance_for_tool(
        self,
        call: ToolCall,
        *,
        explicit_step: int | None,
    ) -> PlanTransition:
        """Associate one real tool activity with the best matching plan item."""

        activity = _activity_for_tool(call)
        inferred = _matching_step(
            self.items,
            activity.keywords,
            start=max(self.current_index, 1),
        )
        target = inferred if inferred is not None else max(self.current_index, 1)
        if (
            explicit_step is not None
            and 1 <= explicit_step <= self.total
            and (
                not activity.action_keywords
                or _step_matches_action(
                    self.items[explicit_step - 1],
                    activity.action_keywords,
                )
                or inferred is None
            )
        ):
            target = explicit_step

        target = max(target, self.current_index)
        if target == self.current_index:
            self._current_has_activity = True
            return PlanTransition()
        return self._transition_to(target)

    def finish(self) -> PlanTransition:
        """Complete the active item; PLAN_COMPLETED closes the whole plan."""

        if self.current_index == 0:
            self.begin()
        update = PlanStepUpdate(
            step=self._step(self.current_index),
            completed=True,
        )
        untracked = self._mark_untracked(
            range(self.current_index + 1, self.total + 1)
        )
        self.current_index = self.total
        self._current_has_activity = False
        return PlanTransition(updates=(update,), untracked=untracked)

    def _transition_to(self, index: int) -> PlanTransition:
        if index <= self.current_index:
            self._current_has_activity = True
            return PlanTransition()

        updates: list[PlanStepUpdate] = []
        untracked_indices: list[int] = []
        if self._current_has_activity:
            updates.append(
                PlanStepUpdate(
                    step=self._step(self.current_index),
                    completed=True,
                )
            )
        else:
            untracked_indices.append(self.current_index)
        untracked_indices.extend(range(self.current_index + 1, index))
        self.current_index = index
        updates.append(PlanStepUpdate(step=self._step(index), completed=False))
        self._current_has_activity = True
        return PlanTransition(
            updates=tuple(updates),
            untracked=self._mark_untracked(untracked_indices),
        )

    def _mark_untracked(self, indices: Iterable[int]) -> tuple[PlanStep, ...]:
        newly_untracked: list[PlanStep] = []
        for index in indices:
            if index in self._untracked_indices:
                continue
            self._untracked_indices.add(index)
            newly_untracked.append(self._step(index))
        return tuple(newly_untracked)

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


def _step_matches_action(item: str, keywords: tuple[str, ...]) -> bool:
    folded = item.casefold()
    return any(keyword in folded for keyword in keywords)


def _activity_for_tool(call: ToolCall) -> _ToolActivity:
    arguments = _json_object(call.arguments_json)
    if call.name != "run_command":
        actions = _TOOL_KEYWORDS.get(call.name, ())
        path = None if arguments is None else arguments.get("path")
        targets = _target_keywords(path)
        if call.name in {"create_file", "replace_text"}:
            if _looks_like_test_path(path):
                actions = _unique(actions + _TEST_AUTHORING_KEYWORDS)
            if _looks_like_documentation_path(path):
                actions = _unique(actions + _DOCUMENTATION_KEYWORDS)
        if call.name == "search_text" and arguments is not None:
            targets = _unique(
                targets + _target_keywords(arguments.get("query"))
            )
        return _ToolActivity(actions, targets)

    argv = None if arguments is None else arguments.get("argv")
    if not isinstance(argv, list) or not argv or any(
        not isinstance(argument, str) for argument in argv
    ):
        return _ToolActivity(())
    program = argv[0].replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if program.endswith(suffix):
            program = program[: -len(suffix)]
            break
    if program in _INSPECTION_COMMANDS:
        return _ToolActivity(_INSPECTION_KEYWORDS)
    if program in _VERIFICATION_COMMANDS:
        return _ToolActivity(_VERIFICATION_KEYWORDS)
    words = {
        word
        for argument in argv
        for word in re.split(r"[^a-z0-9_+.-]+", argument.casefold())
        if word
    }
    if words & _VERIFICATION_COMMAND_WORDS:
        return _ToolActivity(_VERIFICATION_KEYWORDS)
    return _ToolActivity(())


def _target_keywords(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value.strip():
        return ()
    normalized = value.strip().replace("\\", "/").casefold()
    candidates: list[str] = []
    for part in normalized.split("/"):
        if not part or part in {".", ".."}:
            continue
        candidates.append(part)
        stem = part.rsplit(".", maxsplit=1)[0]
        if stem != part:
            candidates.append(stem)
        candidates.extend(
            token
            for token in re.split(r"[^a-z0-9_+-]+", stem)
            if len(token) >= 2
        )
    return _unique(tuple(candidates))


def _looks_like_test_path(value: Any) -> bool:
    keywords = _target_keywords(value)
    return any(
        keyword == "test"
        or keyword == "tests"
        or keyword.startswith("test_")
        or keyword.endswith("_test")
        for keyword in keywords
    )


def _looks_like_documentation_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/").casefold()
    name = normalized.rsplit("/", maxsplit=1)[-1]
    return (
        name.startswith("readme")
        or name.endswith((".md", ".rst"))
        or "/docs/" in f"/{normalized}"
    )


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


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
