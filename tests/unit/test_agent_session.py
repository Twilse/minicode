from pathlib import Path

import pytest

from minicoder.application.agent_engine import AgentEngine
from minicoder.application.event_bus import EventBus
from minicoder.domain.errors import MemoryPersistenceError, ModelConnectionError
from minicoder.domain.events import AgentEvent, AgentEventKind
from minicoder.domain.memory import ProjectMemoryRecord
from minicoder.bootstrap import AgentSession
from minicoder.domain.models import AssistantTurn
from minicoder.domain.state import AgentPhase
from minicoder.tools.output import ToolOutputArtifactStore
from tests.fakes import FakeModelAdapter, FakeToolAdapter, MemoryEventSink


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


def test_session_submits_multiple_turns_and_closes_after_the_context(
    tmp_path: Path,
) -> None:
    session, artifacts = _session(
        tmp_path,
        FakeModelAdapter(
            [
                AssistantTurn(content="The project uses Python."),
                AssistantTurn(content="It requires Python 3.11."),
            ]
        ),
    )
    root = artifacts.root
    output_id = artifacts.save("shared output")

    with session as active:
        first = active.submit("Which language does it use?")
        assert active.closed is False
        assert active.history == first.messages
        assert artifacts.read(output_id, offset=0, limit=100).content == (
            "shared output"
        )

        second = active.submit("What is the minimum version?")
        assert second.messages[: len(first.messages)] == first.messages
        assert active.history == second.messages

    assert session.closed is True
    assert not root.exists()


def test_session_context_closes_artifacts_after_an_unexpected_error(
    tmp_path: Path,
) -> None:
    session, artifacts = _session(tmp_path, FakeModelAdapter([]))
    root = artifacts.root

    with pytest.raises(AssertionError, match="no scripted turn"):
        with session:
            session.submit("Trigger the exhausted fake")

    assert session.closed is True
    assert not root.exists()


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


def test_session_keeps_completed_result_when_memory_append_fails(
    tmp_path: Path,
) -> None:
    class FailingMemoryStore:
        def load_recent(self) -> tuple[ProjectMemoryRecord, ...]:
            return ()

        def append(self, record: ProjectMemoryRecord) -> None:
            raise MemoryPersistenceError("disk unavailable")

    class StaticSummarizer:
        def summarize(
            self,
            *,
            task: str,
            outcome: str,
            model_step: int,
        ) -> str:
            return "Completed the requested work."

    artifacts = ToolOutputArtifactStore(
        max_read_chars=100,
        temporary_parent=tmp_path,
    )
    observed = MemoryEventSink()
    events = EventBus((observed,), run_id="append-failure")
    engine = AgentEngine(
        model=FakeModelAdapter([AssistantTurn(content="done")]),
        tools=FakeToolAdapter(),
        max_steps=2,
        events=events,
    )
    session = AgentSession(
        engine=engine,
        artifacts=artifacts,
        events=events,
        memory_store=FailingMemoryStore(),
        memory_summarizer=StaticSummarizer(),
    )

    result = session.run("Complete even if memory cannot be saved")

    assert result.phase is AgentPhase.COMPLETE
    assert observed.events[-1].kind is AgentEventKind.MEMORY_OPERATION_FAILED
    assert observed.events[-1].details["operation"] == "append"


def test_session_does_not_summarize_failed_turn(tmp_path: Path) -> None:
    class FailingModel:
        def complete(self, **_: object) -> AssistantTurn:
            raise ModelConnectionError("offline")

    class UnusedMemoryStore:
        def load_recent(self) -> tuple[ProjectMemoryRecord, ...]:
            return ()

        def append(self, record: ProjectMemoryRecord) -> None:
            raise AssertionError("failed turns must not be persisted")

    class UnusedSummarizer:
        def summarize(self, **_: object) -> str:
            raise AssertionError("failed turns must not be summarized")

    artifacts = ToolOutputArtifactStore(
        max_read_chars=100,
        temporary_parent=tmp_path,
    )
    events = EventBus()
    engine = AgentEngine(
        model=FailingModel(),
        tools=FakeToolAdapter(),
        max_steps=2,
        events=events,
    )
    session = AgentSession(
        engine=engine,
        artifacts=artifacts,
        events=events,
        memory_store=UnusedMemoryStore(),
        memory_summarizer=UnusedSummarizer(),
    )

    result = session.run("This turn will fail")

    assert result.phase is AgentPhase.FAILED
