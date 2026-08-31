from datetime import datetime, timezone

from minicoder.application.session_context import format_recent_session_context
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
)
from minicoder.domain.session import ArchivedTurnStatus, RecentSessionContext


def test_recent_context_includes_status_summary_and_tool_pairs_without_reasoning() -> None:
    call = ToolCall(
        id="call-1",
        name="replace_text",
        arguments_json='{"path":"app.py"}',
    )
    context = RecentSessionContext(
        session_id="previous-session",
        recorded_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        context_summary="app.py was modified but tests were not run.",
        last_task="finish app.py",
        status=ArchivedTurnStatus.FAILED,
        stop_reason="max_steps",
        recent_messages=(
            Message(role=MessageRole.USER, content="finish app.py"),
            AssistantTurn(
                content=None,
                tool_calls=(call,),
                reasoning_content="private old reasoning",
            ).as_message(),
            ToolResult(
                call_id=call.id,
                tool_name=call.name,
                ok=True,
                content="replacement succeeded",
            ).as_message(),
        ),
    )

    restored = format_recent_session_context(context)

    assert "finish app.py" in restored
    assert "Last status: failed" in restored
    assert "Stop reason: max_steps" in restored
    assert "replace_text" in restored
    assert "replacement succeeded" in restored
    assert "private old reasoning" not in restored


def test_recent_context_keeps_summary_and_latest_tail_when_bounded() -> None:
    context = RecentSessionContext(
        session_id="previous-session",
        recorded_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        context_summary="important rolling summary",
        last_task="old task",
        status=ArchivedTurnStatus.COMPLETE,
        stop_reason="final_response",
        recent_messages=(
            Message(role=MessageRole.USER, content="x" * 500),
            Message(role=MessageRole.ASSISTANT, content="latest-tail"),
        ),
    )

    restored = format_recent_session_context(context, max_chars=300)

    assert len(restored) <= 300
    assert "important rolling summary" in restored
    assert restored.endswith("latest-tail")
