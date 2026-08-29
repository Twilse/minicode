"""In-process Observer that orders events and isolates output sink failures."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from minicoder.application.ports import EventSinkPort
from minicoder.domain.events import AgentEvent, AgentEventKind, EventDetail

EventClock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class EventDeliveryFailure:
    """A non-fatal failure from one observer while handling one event."""

    event_sequence: int  # Event that the observer failed to consume.
    sink_type: str  # Observer class name for host-side diagnostics.
    error_type: str  # Exception class name without a traceback.
    message: str  # Exception text retained outside the model conversation.


class EventBus:
    """Publish ordered events to zero or more synchronous observers."""

    def __init__(
        self,
        sinks: Iterable[EventSinkPort] = (),
        *,
        run_id: str | None = None,
        clock: EventClock | None = None,
    ) -> None:
        selected_run_id = uuid4().hex if run_id is None else run_id
        if not isinstance(selected_run_id, str) or not selected_run_id.strip():
            raise ValueError("event bus run_id must be non-blank text")
        self._sinks = tuple(sinks)
        self._run_id = selected_run_id
        self._clock = _utc_now if clock is None else clock
        self._sequence = 0
        self._failures: list[EventDeliveryFailure] = []

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def failures(self) -> tuple[EventDeliveryFailure, ...]:
        """Return an immutable snapshot of non-fatal observer failures."""

        return tuple(self._failures)

    def publish(
        self,
        kind: AgentEventKind,
        *,
        model_step: int,
        details: Mapping[str, EventDetail] | None = None,
    ) -> AgentEvent:
        """Create, order, and synchronously deliver one sanitized event."""

        event = AgentEvent(
            run_id=self._run_id,
            sequence=self._sequence + 1,
            kind=kind,
            occurred_at=self._clock(),
            model_step=model_step,
            details={} if details is None else details,
        )
        self._sequence = event.sequence
        for sink in self._sinks:
            try:
                sink.handle(event)
            except Exception as exc:
                self._failures.append(
                    EventDeliveryFailure(
                        event_sequence=event.sequence,
                        sink_type=type(sink).__name__,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return event


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
