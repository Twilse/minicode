from pathlib import Path

import pytest

from minicoder.application.agent_engine import AgentEngine
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
    engine = AgentEngine(
        model=model,
        tools=FakeToolAdapter(),
        max_steps=2,
    )
    return AgentSession(engine=engine, artifacts=artifacts), artifacts


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
