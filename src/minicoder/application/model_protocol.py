"""Decode reserved host annotations at the application/model boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from minicoder.domain.errors import ModelResponseError
from minicoder.domain.models import AssistantTurn

_PLAN_STEP_ANNOTATION = re.compile(
    r"\[plan_step\s*=\s*(\d+)\][ \t]*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DecodedAssistantTurn:
    """A clean model turn plus host metadata removed from its visible text."""

    turn: AssistantTurn  # Turn safe to store in history or return to the user.
    plan_step: int | None  # Optional one-based progress hint for tool calls.


def decode_assistant_turn(turn: AssistantTurn) -> DecodedAssistantTurn:
    """Separate all recognized host annotations from user-visible content."""

    content = turn.content
    if content is None:
        return DecodedAssistantTurn(turn=turn, plan_step=None)
    matches = tuple(_PLAN_STEP_ANNOTATION.finditer(content))
    if not matches:
        return DecodedAssistantTurn(turn=turn, plan_step=None)

    cleaned_content = _PLAN_STEP_ANNOTATION.sub("", content).strip() or None
    if cleaned_content is None and not turn.tool_calls:
        raise ModelResponseError(
            "model final response contained only reserved host annotations"
        )
    clean_turn = AssistantTurn(
        content=cleaned_content,
        tool_calls=turn.tool_calls,
        reasoning_content=turn.reasoning_content,
    )
    plan_step = int(matches[0].group(1)) if turn.tool_calls else None
    return DecodedAssistantTurn(turn=clean_turn, plan_step=plan_step)
