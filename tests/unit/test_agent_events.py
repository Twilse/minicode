from datetime import datetime, timezone

import pytest

from minicoder.domain.errors import DomainValidationError
from minicoder.domain.events import AgentEvent, AgentEventKind


def _event(**overrides: object) -> AgentEvent:
    values: dict[str, object] = {
        "run_id": "run-test",
        "sequence": 1,
        "kind": AgentEventKind.TASK_STARTED,
        "occurred_at": datetime(2026, 8, 30, 8, 0, tzinfo=timezone.utc),
        "model_step": 0,
        "details": {"tool_count": 7},
    }
    values.update(overrides)
    return AgentEvent(**values)  # type: ignore[arg-type]


def test_event_copies_and_freezes_safe_scalar_details() -> None:
    source = {"tool_name": "read_file", "ok": True}

    event = _event(details=source)
    source["tool_name"] = "replace_text"

    assert event.details == {"tool_name": "read_file", "ok": True}
    with pytest.raises(TypeError):
        event.details["ok"] = False  # type: ignore[index]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"run_id": "   "}, "run_id"),
        ({"sequence": 0}, "sequence"),
        ({"kind": "task_started"}, "AgentEventKind"),
        ({"occurred_at": datetime(2026, 8, 30)}, "timezone-aware"),
        ({"model_step": -1}, "model_step"),
        ({"details": {"nested": {"secret": True}}}, "scalar JSON"),
        ({"details": {"ratio": float("inf")}}, "finite"),
    ],
)
def test_event_rejects_invalid_protocol_values(
    override: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(DomainValidationError, match=message):
        _event(**override)
