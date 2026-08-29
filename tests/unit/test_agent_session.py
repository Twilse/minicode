from pathlib import Path

import pytest

from minicoder.application.agent_engine import AgentEngine
from minicoder.application.event_bus import EventBus
from minicoder.domain.events import AgentEvent
from minicoder.bootstrap import AgentSession
from minicoder.domain.models import AssistantTurn
from minicoder.domain.state import AgentPhase
from minicoder.tools.output import ToolOutputArtifactStore
from tests.fakes import FakeModelAdapter, FakeToolAdapter


def _session(tmp_path: Path, model: FakeModelAdapter) -> tuple[
    AgentSession,
    ToolOutputArtifactStore,
]:
    artifacts = ToolOutputArtifactStore(
        max_read_chars=100,
        temporary_parent=tmp_path,
    )
    events = EventBus()
    engine = AgentEngine(
        model=model,
        tools=FakeToolAdapter(),
        max_steps=2,
        events=events,
    )
    return (
        AgentSession(engine=engine, artifacts=artifacts, events=events),
        artifacts,
    )


def test_session_closes_artifacts_after_a_completed_task(tmp_path: Path) -> None:
    session, artifacts = _session(
        tmp_path,
        FakeModelAdapter([AssistantTurn(content="done")]),
    )
    root = artifacts.root

    result = session.run("Complete one task")

    assert result.phase is AgentPhase.COMPLETE
    assert session.closed is True
    assert not root.exists()
    with pytest.raises(RuntimeError, match="already closed"):
        session.run("A second task is not allowed")


def test_session_closes_artifacts_after_an_unexpected_error(tmp_path: Path) -> None:
    session, artifacts = _session(tmp_path, FakeModelAdapter([]))
    root = artifacts.root

    with pytest.raises(AssertionError, match="no scripted turn"):
        session.run("Trigger the exhausted fake")

    assert session.closed is True
    assert not root.exists()


def test_session_exposes_non_fatal_event_sink_failures(tmp_path: Path) -> None:
    class FailingSink:
        def handle(self, event: AgentEvent) -> None:
            raise OSError(f"trace failed at {event.sequence}")

    artifacts = ToolOutputArtifactStore(
        max_read_chars=100,
        temporary_parent=tmp_path,
    )
    events = EventBus((FailingSink(),))
    engine = AgentEngine(
        model=FakeModelAdapter([AssistantTurn(content="done")]),
        tools=FakeToolAdapter(),
        max_steps=2,
        events=events,
    )
    session = AgentSession(engine=engine, artifacts=artifacts, events=events)

    result = session.run("Complete despite the broken trace sink")

    assert result.phase is AgentPhase.COMPLETE
    assert len(session.event_failures) == 3
    assert {failure.event_sequence for failure in session.event_failures} == {1, 2, 3}
    assert all(failure.sink_type == "FailingSink" for failure in session.event_failures)
