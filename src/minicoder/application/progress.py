"""Bounded user-facing progress facts derived from model plans and tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.events import EventDetail
from minicoder.domain.models import ToolCall

_MAX_PLAN_ITEMS = 5
_MAX_PLAN_ITEM_CHARS = 240
_MAX_DISPLAY_PATH_CHARS = 300
_MAX_DISPLAY_QUERY_CHARS = 120
_MAX_DISPLAY_ARGUMENT_CHARS = 120
_MAX_DISPLAY_COMMAND_CHARS = 500

_NUMBERED_PLAN_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?(\d{1,2})[.)、:：]\s*(.+?)\s*$"
)
_BULLET_PLAN_LINE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")
_MARKDOWN_HEADING_PREFIX = re.compile(r"^#{1,6}\s*")
_PLAN_HEADING_TERMS = frozenset(
    {
        "计划",
        "计划如下",
        "工作计划",
        "行动计划",
        "执行计划",
        "实施计划",
        "开发计划",
        "修改计划",
        "处理计划",
        "规划",
        "规划如下",
        "执行规划",
        "开发规划",
        "方案",
        "方案如下",
        "执行方案",
        "实施方案",
        "处理方案",
        "修改方案",
        "技术方案",
        "步骤",
        "步骤如下",
        "执行步骤",
        "实施步骤",
        "操作步骤",
        "处理步骤",
        "开发步骤",
        "实现思路",
        "处理思路",
        "解决思路",
        "plan",
        "the plan",
        "planning",
        "action plan",
        "execution plan",
        "implementation plan",
        "development plan",
        "work plan",
        "proposed plan",
        "approach",
        "proposed approach",
        "implementation approach",
        "solution",
        "roadmap",
        "steps",
        "action steps",
        "execution steps",
        "implementation steps",
        "next steps",
    }
)
_SAFE_COMMAND_ARGUMENT = re.compile(r"[A-Za-z0-9_./:@%+=,-]+")
_ANSI_ESCAPE_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TERMINAL_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

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
    """One ordered plan movement explicitly reported by the model."""

    updates: tuple[PlanStepUpdate, ...] = ()  # Ordered start/complete changes.


@dataclass(slots=True)
class PlanProgress:
    """Require explicit, sequential completion of a model-generated plan."""

    items: tuple[str, ...]
    current_index: int = 0
    _final_step_reported: bool = field(default=False, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

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
                "plan progress requires one to five non-blank text items"
            )
        if self.current_index < 0 or self.current_index > len(self.items):
            raise DomainValidationError("plan current index is outside the plan")

    @classmethod
    def from_model_text(cls, text: str) -> PlanProgress:
        """Parse a numbered or bulleted model plan without another model call."""

        return cls(items=_parse_plan_items(text))

    @classmethod
    def from_planning_response(cls, text: str) -> PlanProgress:
        """Accept a loose plan heading plus at least one explicit list item."""

        if not isinstance(text, str) or not text.strip():
            raise DomainValidationError("planning response must not be empty")
        if not any(_is_plan_heading(line) for line in text.splitlines()):
            raise DomainValidationError(
                "planning response requires a recognized standalone plan heading"
            )
        items = _parse_explicit_plan_items(text)
        if not items:
            raise DomainValidationError(
                "planning response requires a numbered or bulleted item"
            )
        return cls(items=items)

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
    def current_step(self) -> PlanStep:
        """Return the currently active item after begin() has been called."""

        if self.current_index == 0:
            raise DomainValidationError("plan progress has not started")
        return self._step(self.current_index)

    @property
    def all_steps_reported(self) -> bool:
        """Whether the model explicitly reported the final item complete."""

        return self._final_step_reported

    def begin(self) -> PlanStepUpdate:
        """Start the first plan item after planning completes."""

        if self.current_index != 0 or self._finished:
            raise DomainValidationError("plan progress has already started")
        self.current_index = 1
        return PlanStepUpdate(step=self._step(1), completed=False)

    def complete_current(self, step: int) -> PlanTransition:
        """Complete only the active item and activate its immediate successor."""

        if self.current_index == 0:
            raise DomainValidationError("plan progress has not started")
        if self._finished:
            raise DomainValidationError("plan progress is already finished")
        if not isinstance(step, int) or isinstance(step, bool):
            raise DomainValidationError("completed plan step must be an integer")
        if step != self.current_index:
            raise DomainValidationError(
                f"current plan step is {self.current_index}, not {step}"
            )
        if self._final_step_reported:
            raise DomainValidationError("final plan step was already reported")
        if self.current_index == self.total:
            self._final_step_reported = True
            return PlanTransition()

        completed = PlanStepUpdate(
            step=self._step(self.current_index),
            completed=True,
        )
        self.current_index += 1
        started = PlanStepUpdate(
            step=self._step(self.current_index),
            completed=False,
        )
        return PlanTransition(updates=(completed, started))

    def finish(self) -> PlanTransition:
        """Publish final completion only after the final report and acceptance."""

        if not self._final_step_reported:
            raise DomainValidationError(
                "cannot finish plan before the final step is reported"
            )
        if self._finished:
            raise DomainValidationError("plan progress is already finished")
        self._finished = True
        return PlanTransition(
            updates=(
                PlanStepUpdate(
                    step=self._step(self.current_index),
                    completed=True,
                ),
            )
        )

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
    elif call.name in {"read_file", "create_file", "write_file", "replace_text"}:
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
    explicit = _parse_explicit_plan_items(text)
    if explicit:
        return explicit

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


def _parse_explicit_plan_items(text: str) -> tuple[str, ...]:
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

    bullets: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        bullet = _BULLET_PLAN_LINE.match(line)
        if bullet is None:
            continue
        bullets.append(_clean_plan_item(bullet.group(1)))
        if len(bullets) == _MAX_PLAN_ITEMS:
            break
    return tuple(bullets)


def _is_plan_heading(raw_line: str) -> bool:
    line = _remove_terminal_controls(raw_line).strip()
    line = _MARKDOWN_HEADING_PREFIX.sub("", line).strip()
    line = line.strip(" *_`").rstrip(":：").strip(" *_`")
    normalized = " ".join(line.casefold().split())
    return normalized in _PLAN_HEADING_TERMS


def _clean_plan_item(text: str) -> str:
    compact = _remove_terminal_controls(" ".join(text.split())).strip(" *_`")
    return _truncate(compact or "执行当前步骤", _MAX_PLAN_ITEM_CHARS)


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
