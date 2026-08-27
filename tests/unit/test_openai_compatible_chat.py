from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.domain.errors import (
    ModelAccessError,
    ModelConnectionError,
    ModelRequestError,
    ModelResponseError,
    ModelRateLimitError,
    ModelServiceError,
)
from minicoder.domain.models import (
    AssistantTurn,
    Message,
    MessageRole,
    ToolCall,
    ToolDefinition,
)


class RecordingCompletions:
    def __init__(self, outcomes: Iterable[Any]) -> None:
        self._outcomes = deque(outcomes)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _adapter_with(outcomes: Iterable[Any]) -> tuple[
    OpenAICompatibleChatAdapter,
    RecordingCompletions,
]:
    completions = RecordingCompletions(outcomes)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return (
        OpenAICompatibleChatAdapter(client=client, model="coding-model"),
        completions,
    )


def _response(
    *,
    content: str | None,
    tool_calls: list[Any] | None = None,
    reasoning_content: str | None = None,
) -> Any:
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _function_call(
    *,
    call_id: str = "call-1",
    name: str = "read_file",
    arguments: str = '{"path":"main.py"}',
) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_adapter_serializes_provider_neutral_tools_and_parses_tool_calls() -> None:
    adapter, completions = _adapter_with(
        [
            _response(
                content=None,
                tool_calls=[_function_call()],
                reasoning_content="private protocol state",
            )
        ]
    )
    tool = ToolDefinition(
        name="read_file",
        description="Read one text file.",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    turn = adapter.complete(
        messages=(Message(role=MessageRole.USER, content="read main.py"),),
        tools=(tool,),
    )

    request = completions.requests[0]
    assert request["model"] == "coding-model"
    assert request["messages"] == [{"role": "user", "content": "read main.py"}]
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one text file.",
                "parameters": dict(tool.parameters_schema),
            },
        }
    ]
    assert "tool_choice" not in request
    assert turn == AssistantTurn(
        content=None,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="read_file",
                arguments_json='{"path":"main.py"}',
            ),
        ),
        reasoning_content="private protocol state",
    )


def test_adapter_replays_reasoning_only_for_assistant_tool_call_messages() -> None:
    adapter, completions = _adapter_with(
        [_response(content="done", reasoning_content="discard after final answer")]
    )
    call = ToolCall(id="call-1", name="read_file", arguments_json="{}")
    assistant_message = AssistantTurn(
        content=None,
        tool_calls=(call,),
        reasoning_content="required continuation state",
    ).as_message()
    tool_message = Message(
        role=MessageRole.TOOL,
        content="file contents",
        tool_call_id="call-1",
    )

    turn = adapter.complete(
        messages=(assistant_message, tool_message),
        tools=(),
    )

    sent_messages = completions.requests[0]["messages"]
    assert sent_messages[0]["content"] == ""
    assert sent_messages[0]["reasoning_content"] == "required continuation state"
    assert sent_messages[0]["tool_calls"][0]["id"] == "call-1"
    assert sent_messages[1] == {
        "role": "tool",
        "content": "file contents",
        "tool_call_id": "call-1",
    }
    assert "tools" not in completions.requests[0]
    assert turn == AssistantTurn(content="done")


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(choices=[]),
        _response(content=None),
        _response(
            content=None,
            tool_calls=[SimpleNamespace(type="custom")],
        ),
    ],
)
def test_adapter_rejects_malformed_model_responses(response: Any) -> None:
    adapter, _ = _adapter_with([response])

    with pytest.raises(ModelResponseError):
        adapter.complete(
            messages=(Message(role=MessageRole.USER, content="task"),),
            tools=(),
        )


def _request() -> httpx.Request:
    return httpx.Request(
        "POST",
        "https://models.example.com/v1/chat/completions",
    )


def _status_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=_request(),
        headers={"x-request-id": "request-123"},
    )


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            openai.AuthenticationError(
                "unauthorized",
                response=_status_response(401),
                body=None,
            ),
            ModelAccessError,
        ),
        (
            openai.RateLimitError(
                "too many requests",
                response=_status_response(429),
                body=None,
            ),
            ModelRateLimitError,
        ),
        (
            openai.APIConnectionError(request=_request()),
            ModelConnectionError,
        ),
        (
            openai.APIStatusError(
                "server failure",
                response=_status_response(503),
                body=None,
            ),
            ModelServiceError,
        ),
        (
            openai.APIStatusError(
                "bad request",
                response=_status_response(400),
                body=None,
            ),
            ModelRequestError,
        ),
    ],
)
def test_adapter_translates_sdk_errors(
    sdk_error: BaseException,
    expected_error: type[Exception],
) -> None:
    adapter, _ = _adapter_with([sdk_error])

    with pytest.raises(expected_error):
        adapter.complete(
            messages=(Message(role=MessageRole.USER, content="task"),),
            tools=(),
        )


def test_adapter_rejects_an_empty_conversation_before_calling_the_sdk() -> None:
    adapter, completions = _adapter_with([])

    with pytest.raises(ModelRequestError, match="at least one message"):
        adapter.complete(messages=(), tools=())

    assert completions.requests == []
