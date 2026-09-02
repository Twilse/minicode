from datetime import datetime, timezone

from minicoder.application.session_context import format_recent_session_boundary
from minicoder.domain.models import Message, MessageRole
from minicoder.domain.session import ArchivedTurnStatus, RecentSessionContext


def test_recent_boundary_contains_only_unfinished_process_metadata() -> None:
    context = RecentSessionContext(
        session_id="previous-session",
        recorded_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        last_task="finish app.py",
        status=ArchivedTurnStatus.FAILED,
        stop_reason="max_steps",
        messages=(Message(role=MessageRole.USER, content="private old message"),),
        failure_message="The task stopped before tests ran.",
    )

    restored = format_recent_session_boundary(context)

    assert "finish app.py" not in restored
    assert "Status: failed" in restored
    assert "Stop reason: max_steps" in restored
    assert "The task stopped before tests ran" not in restored
    assert "private old message" not in restored


def test_completed_process_has_no_redundant_boundary() -> None:
    messages = (Message(role=MessageRole.ASSISTANT, content="exact history"),)
    context = RecentSessionContext(
        session_id="previous-session",
        recorded_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
        last_task="old task",
        status=ArchivedTurnStatus.COMPLETE,
        stop_reason="final_response",
        messages=messages,
        final_response="x" * 500,
    )

    restored = format_recent_session_boundary(context, max_chars=200)

    assert restored == ""
    assert context.messages == messages
