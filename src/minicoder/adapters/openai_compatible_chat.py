"""OpenAI-compatible Chat Completions adapter for the model port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import openai

from minicoder.domain.errors import (
    DomainValidationError,
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


class _ChatCompletionsResource(Protocol):
    """The narrow SDK operation used by this adapter."""

    def create(self, **kwargs: Any) -> Any:
        """Create one non-streaming chat completion."""

        ...


class _ChatResource(Protocol):
    """The SDK namespace containing Chat Completions."""

    completions: _ChatCompletionsResource


class ChatCompletionClient(Protocol):
    """The small part of an OpenAI-compatible client required by the adapter."""

    chat: _ChatResource


class OpenAICompatibleChatAdapter:
    """Translate between MiniCoder values and Chat Completions SDK objects."""

    def __init__(self, *, client: ChatCompletionClient, model: str) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ModelRequestError("model name must not be empty")
        self._client = client
        self._model = normalized_model

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> AssistantTurn:
        """Send one request and normalize the first returned choice."""

        if not messages:
            raise ModelRequestError("at least one message is required")

        request: dict[str, Any] = {
            "model": self._model,
            "messages": [_serialize_message(message) for message in messages],
        }
        if tools:
            request["tools"] = [_serialize_tool(tool) for tool in tools]

        try:
            response = self._client.chat.completions.create(**request)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ModelAccessError(
                "model service rejected the configured credentials or permissions"
            ) from exc
        except openai.RateLimitError as exc:
            raise ModelRateLimitError("model service rate limit exceeded") from exc
        except openai.APIConnectionError as exc:
            raise ModelConnectionError("could not connect to the model service") from exc
        except openai.APIStatusError as exc:
            status_code = exc.status_code
            suffix = _request_id_suffix(exc)
            if status_code >= 500:
                raise ModelServiceError(
                    f"model service returned HTTP {status_code}{suffix}"
                ) from exc
            raise ModelRequestError(
                f"model service rejected the request with HTTP {status_code}{suffix}"
            ) from exc
        except openai.APIError as exc:
            raise ModelServiceError("model API request failed") from exc

        return _parse_response(response)


def _serialize_message(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role.value}

    if message.role is MessageRole.ASSISTANT:
        if message.content is None and not message.tool_calls:
            raise ModelRequestError(
                "assistant messages without tool calls require content"
            )
        payload["content"] = "" if message.content is None else message.content
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        if message.reasoning_content is not None:
            payload["reasoning_content"] = message.reasoning_content
        return payload

    if message.content is None:
        raise ModelRequestError(f"{message.role.value} messages require content")
    payload["content"] = message.content

    if message.role is MessageRole.TOOL:
        payload["tool_call_id"] = message.tool_call_id
    return payload


def _serialize_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters_schema),
        },
    }


def _parse_response(response: Any) -> AssistantTurn:
    choices = getattr(response, "choices", None)
    if not choices:
        raise ModelResponseError("model response did not contain a choice")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ModelResponseError("model response choice did not contain a message")

    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ModelResponseError("model response content must be text or null")

    raw_tool_calls = getattr(message, "tool_calls", None) or ()
    tool_calls = tuple(_parse_tool_call(raw_call) for raw_call in raw_tool_calls)
    if not tool_calls and (content is None or not content.strip()):
        raise ModelResponseError(
            "model response contained neither final text nor tool calls"
        )

    raw_reasoning = getattr(message, "reasoning_content", None)
    if raw_reasoning is not None and not isinstance(raw_reasoning, str):
        raise ModelResponseError(
            "model response reasoning_content must be text or null"
        )

    return AssistantTurn(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=raw_reasoning,
    )


def _parse_tool_call(raw_call: Any) -> ToolCall:
    if getattr(raw_call, "type", None) != "function":
        raise ModelResponseError("model returned an unsupported tool call type")

    function = getattr(raw_call, "function", None)
    if function is None:
        raise ModelResponseError("model tool call did not contain a function")

    try:
        return ToolCall(
            id=_required_response_text(raw_call, "id"),
            name=_required_response_text(function, "name"),
            arguments_json=_required_response_text(function, "arguments"),
        )
    except DomainValidationError as exc:
        raise ModelResponseError("model returned an invalid tool call") from exc


def _required_response_text(value: Any, attribute: str) -> str:
    text = getattr(value, attribute, None)
    if not isinstance(text, str) or not text.strip():
        raise ModelResponseError(
            f"model response field {attribute!r} must be non-empty text"
        )
    return text


def _request_id_suffix(exc: openai.APIStatusError) -> str:
    request_id = getattr(exc, "request_id", None)
    return f" (request_id={request_id})" if request_id else ""
