from __future__ import annotations

import pytest

from minicoder.application.model_protocol import decode_assistant_turn
from minicoder.domain.errors import ModelResponseError
from minicoder.domain.models import AssistantTurn, ToolCall


def test_decoder_removes_host_annotations_from_tool_call_content() -> None:
    call = ToolCall(
        id="call-read",
        name="read_file",
        arguments_json='{"path":"app.py"}',
    )
    raw = AssistantTurn(
        content="[plan_step=2] Reading the file.",
        tool_calls=(call,),
        reasoning_content="private state",
    )

    decoded = decode_assistant_turn(raw)

    assert decoded.turn.content == "Reading the file."
    assert decoded.turn.tool_calls == (call,)
    assert decoded.turn.reasoning_content == "private state"
    assert decoded.plan_step == 2


def test_decoder_removes_all_host_annotations_from_a_final_answer() -> None:
    raw = AssistantTurn(
        content=(
            "[plan_step=1] Remembered the project.\n\n"
            "[plan_step=2] No files were changed."
        )
    )

    decoded = decode_assistant_turn(raw)

    assert decoded.turn.content == (
        "Remembered the project.\n\nNo files were changed."
    )
    assert decoded.plan_step is None


def test_decoder_leaves_ordinary_model_content_unchanged() -> None:
    raw = AssistantTurn(content="A normal final answer.")

    decoded = decode_assistant_turn(raw)

    assert decoded.turn is raw
    assert decoded.plan_step is None


def test_final_answer_cannot_consist_only_of_host_annotations() -> None:
    with pytest.raises(ModelResponseError):
        decode_assistant_turn(AssistantTurn(content="[plan_step=1]"))
