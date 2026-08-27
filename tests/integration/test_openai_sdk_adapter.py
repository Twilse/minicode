import json
from typing import Any

import httpx
from openai import OpenAI

from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.domain.models import AssistantTurn, Message, MessageRole, ToolCall


def test_real_sdk_preserves_reasoning_content_for_tool_call_continuation() -> None:
    captured_request: dict[str, Any] = {}

    def handle_request(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1,
                "model": "compatible-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "next private protocol state",
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "search_text",
                                        "arguments": '{"query":"Todo"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handle_request))
    sdk_client = OpenAI(
        api_key="test-key",
        base_url="https://models.example.com/v1",
        http_client=http_client,
        max_retries=0,
    )
    adapter = OpenAICompatibleChatAdapter(
        client=sdk_client,
        model="compatible-model",
    )
    previous_call = ToolCall(
        id="call-1",
        name="read_file",
        arguments_json='{"path":"todo.py"}',
    )

    try:
        turn = adapter.complete(
            messages=(
                AssistantTurn(
                    content="",
                    tool_calls=(previous_call,),
                    reasoning_content="previous private protocol state",
                ).as_message(),
                Message(
                    role=MessageRole.TOOL,
                    content="file contents",
                    tool_call_id="call-1",
                ),
            ),
            tools=(),
        )
    finally:
        sdk_client.close()

    assert captured_request["messages"][0]["reasoning_content"] == (
        "previous private protocol state"
    )
    assert turn.reasoning_content == "next private protocol state"
    assert turn.tool_calls[0].name == "search_text"
