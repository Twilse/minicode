from datetime import datetime, timezone

from minicoder.application.event_bus import EventBus
from minicoder.domain.events import AgentEvent, AgentEventKind
from tests.fakes import MemoryEventSink

FIXED_TIME = datetime(2026, 8, 30, 8, 30, tzinfo=timezone.utc)


def test_event_bus_assigns_one_run_id_and_ordered_sequences() -> None:
    first = MemoryEventSink()
    second = MemoryEventSink()
    bus = EventBus(
        (first, second),
        run_id="run-123",
        clock=lambda: FIXED_TIME,
    )

    started = bus.publish(
        AgentEventKind.TASK_STARTED,
        model_step=0,
        details={"tool_count": 7},
    )
    requested = bus.publish(
        AgentEventKind.MODEL_REQUESTED,
        model_step=1,
        details={"message_count": 2},
    )

    assert [event.sequence for event in first.events] == [1, 2]
    assert first.events == second.events
    assert started.run_id == requested.run_id == "run-123"
    assert started.occurred_at == requested.occurred_at == FIXED_TIME
    assert bus.failures == ()


def test_event_bus_isolates_one_sink_failure_and_continues_delivery() -> None:
    class FailingSink:
        def handle(self, event: AgentEvent) -> None:
            raise OSError(f"disk unavailable at event {event.sequence}")

    memory = MemoryEventSink()
    bus = EventBus(
        (FailingSink(), memory),
        run_id="run-failure",
        clock=lambda: FIXED_TIME,
    )

    event = bus.publish(AgentEventKind.TASK_COMPLETED, model_step=2)

    assert memory.events == [event]
    assert len(bus.failures) == 1
    failure = bus.failures[0]
    assert failure.event_sequence == 1
    assert failure.sink_type == "FailingSink"
    assert failure.error_type == "OSError"
    assert "disk unavailable" in failure.message
