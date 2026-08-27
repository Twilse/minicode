from minicoder.application.ports import ModelPort
from minicoder.domain.models import AssistantTurn, Message, MessageRole
from tests.fakes import FakeModelAdapter


def test_fake_model_adapter_satisfies_the_model_port_contract() -> None:
    scripted_turn = AssistantTurn(content="done")
    adapter: ModelPort = FakeModelAdapter([scripted_turn])
    messages = (Message(role=MessageRole.USER, content="fix the tests"),)

    result = adapter.complete(messages=messages, tools=())

    assert result is scripted_turn
    assert adapter.requests[0].messages == messages  # type: ignore[attr-defined]


def test_fake_model_adapter_fails_when_script_is_exhausted() -> None:
    adapter = FakeModelAdapter([])

    try:
        adapter.complete(
            messages=(Message(role=MessageRole.USER, content="task"),),
            tools=(),
        )
    except AssertionError as exc:
        assert "no scripted turn" in str(exc)
    else:
        raise AssertionError("expected an exhausted fake model to fail")
