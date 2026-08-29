"""In-memory event observer used by application tests."""

from minicoder.domain.events import AgentEvent


class MemoryEventSink:
    """Retain events in delivery order without rendering or persistence."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def handle(self, event: AgentEvent) -> None:
        self.events.append(event)
