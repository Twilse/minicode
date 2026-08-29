from __future__ import annotations

import json
from collections.abc import Callable, Sequence

import pytest

from minicoder.application.ports import ToolPort
from minicoder.domain.models import ToolCall, ToolDefinition, ToolResult
from minicoder.tools.pipeline import (
    TOOL_CONTRACT_ERROR,
    TOOL_EXECUTION_ERROR,
    ToolContractMiddleware,
    ToolExceptionBoundary,
    ToolMiddleware,
    ToolPipeline,
    ToolResultEnvelopeMiddleware,
)


CALL = ToolCall(id="call-1", name="run_command", arguments_json="{}")
DEFINITION = ToolDefinition(
    name="run_command",
    description="Run one test command.",
    parameters_schema={"type": "object"},
)


class StubToolPort:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[ToolCall] = []

    def definitions(self) -> Sequence[ToolDefinition]:
        return (DEFINITION,)

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result  # type: ignore[return-value]


class RecordingMiddleware:
    def __init__(self, name: str, order: list[str]) -> None:
        self._name = name
        self._order = order

    def handle(
        self,
        call: ToolCall,
        next_handler: Callable[[ToolCall], ToolResult],
    ) -> ToolResult:
        self._order.append(f"{self._name}:before")
        result = next_handler(call)
        self._order.append(f"{self._name}:after")
        return result


def _result(
    *,
    content: str = "ok",
    metadata: dict[str, object] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=CALL.id,
        tool_name=CALL.name,
        ok=True,
        content=content,
        metadata={} if metadata is None else metadata,
    )


def _production_pipeline(backend: ToolPort, *, max_chars: int = 512) -> ToolPipeline:
    return ToolPipeline(
        backend,
        middleware=(
            ToolExceptionBoundary(),
            ToolResultEnvelopeMiddleware(max_model_chars=max_chars),
            ToolContractMiddleware(),
        ),
    )


def test_pipeline_delegates_definitions_and_nests_middleware_in_order() -> None:
    order: list[str] = []

    class OrderedPort(StubToolPort):
        def execute(self, call: ToolCall) -> ToolResult:
            order.append("backend")
            return super().execute(call)

    backend = OrderedPort(_result())
    middleware: tuple[ToolMiddleware, ...] = (
        RecordingMiddleware("outer", order),
        RecordingMiddleware("inner", order),
    )
    pipeline = ToolPipeline(backend, middleware=middleware)

    result = pipeline.execute(CALL)

    assert pipeline.definitions() == (DEFINITION,)
    assert result.ok is True
    assert order == [
        "outer:before",
        "inner:before",
        "backend",
        "inner:after",
        "outer:after",
    ]


def test_exception_boundary_normalizes_unexpected_errors_without_leaking_message() -> None:
    pipeline = _production_pipeline(
        StubToolPort(RuntimeError("secret implementation details"))
    )

    result = pipeline.execute(CALL)

    assert result.ok is False
    assert result.error_code == TOOL_EXECUTION_ERROR
    assert result.metadata == {"exception_type": "RuntimeError"}
    assert "secret implementation details" not in result.content


def test_exception_boundary_does_not_swallow_keyboard_interrupt() -> None:
    pipeline = _production_pipeline(StubToolPort(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        pipeline.execute(CALL)


@pytest.mark.parametrize(
    "bad_result",
    [
        "not a ToolResult",
        ToolResult(
            call_id="wrong-call",
            tool_name=CALL.name,
            ok=True,
            content="wrong correlation",
        ),
        ToolResult(
            call_id=CALL.id,
            tool_name="wrong-tool",
            ok=True,
            content="wrong correlation",
        ),
    ],
)
def test_contract_middleware_replaces_invalid_or_uncorrelated_results(
    bad_result: object,
) -> None:
    result = _production_pipeline(StubToolPort(bad_result)).execute(CALL)

    assert result.ok is False
    assert result.error_code == TOOL_CONTRACT_ERROR
    assert result.call_id == CALL.id
    assert result.tool_name == CALL.name


def test_envelope_exposes_only_bounded_whitelisted_metadata() -> None:
    output_id = "out_123"
    result = _production_pipeline(
        StubToolPort(
            _result(
                metadata={
                    "output_id": output_id,
                    "original_chars": 2_000,
                    "returned_chars": 200,
                    "truncated": True,
                    "exit_code": 1,
                    "timed_out": False,
                    "included_ranges": ((0, 100), (1_900, 2_000)),
                    "argv": ["private", "arguments"],
                    "exception_type": "SensitiveError",
                    "oversized_id": "x" * 10_000,
                }
            )
        )
    ).execute(CALL)

    payload = json.loads(result.model_content())

    assert payload["metadata"] == {
        "exit_code": 1,
        "included_ranges": [[0, 100], [1_900, 2_000]],
        "original_chars": 2_000,
        "output_id": output_id,
        "returned_chars": 200,
        "timed_out": False,
        "truncated": True,
    }
    assert result.metadata["argv"] == ["private", "arguments"]
    assert "argv" not in payload["metadata"]
    assert "exception_type" not in payload["metadata"]


def test_envelope_applies_a_hard_budget_to_the_complete_json_message() -> None:
    content = (
        "command header\n"
        + "noise\\\"\n" * 500
        + "Traceback: important failure\n"
        + "tail summary\n" * 200
    )
    result = _production_pipeline(
        StubToolPort(
            _result(
                content=content,
                metadata={
                    "output_id": "out_complete_output",
                    "original_chars": len(content),
                    "returned_chars": len(content),
                    "truncated": True,
                    "exit_code": 1,
                },
            )
        )
    ).execute(CALL)

    payload = json.loads(result.model_content())

    assert len(result.model_content()) <= 512
    assert payload["metadata"]["content_truncated"] is True
    assert payload["metadata"]["content_original_chars"] == len(content)
    assert payload["metadata"]["content_returned_chars"] == len(result.content)
    assert payload["metadata"]["output_id"] == "out_complete_output"
    assert "command header" in result.content
    assert "tail summary" in result.content


def test_short_result_is_not_changed_except_for_safe_model_metadata() -> None:
    original = _result(content="all tests passed", metadata={"exit_code": 0})

    result = _production_pipeline(StubToolPort(original)).execute(CALL)

    assert result.content == original.content
    assert result.metadata == original.metadata
    assert result.model_metadata == {"exit_code": 0}
    assert len(result.model_content()) <= 512
