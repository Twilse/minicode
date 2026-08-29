"""Composable middleware around the raw tool registry boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any, Protocol

from minicoder.application.ports import ToolPort
from minicoder.domain.models import ToolCall, ToolDefinition, ToolResult
from minicoder.tools.output import DiagnosticOutputCompactor, TextCompactionStrategy

TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
TOOL_CONTRACT_ERROR = "TOOL_CONTRACT_ERROR"

_MIN_MODEL_CHARS = 512
_MAX_MODEL_STRING_CHARS = 128
_MAX_MODEL_INTEGER = 10**15
_MAX_INCLUDED_RANGES = 16
_MIN_COMPACTED_CONTENT_CHARS = 1

ToolHandler = Callable[[ToolCall], ToolResult]


class ToolMiddleware(Protocol):
    """Wrap one downstream tool handler with one reusable processing concern."""

    def handle(self, call: ToolCall, next_handler: ToolHandler) -> ToolResult:
        """Process a call, delegating to the next link when appropriate."""

        ...


class ToolPipeline:
    """Expose one ToolPort while applying middleware in declared outer-first order."""

    def __init__(
        self,
        backend: ToolPort,
        *,
        middleware: Iterable[ToolMiddleware],
    ) -> None:
        self._backend = backend
        self._middleware = tuple(middleware)

    def definitions(self) -> Sequence[ToolDefinition]:
        """Delegate the model-visible schemas without changing their stable order."""

        return self._backend.definitions()

    def execute(self, call: ToolCall) -> ToolResult:
        """Build and invoke the small synchronous responsibility chain."""

        handler = self._backend.execute
        for current in reversed(self._middleware):
            handler = _bind(current, handler)
        return handler(call)


class ToolExceptionBoundary:
    """Convert unexpected ordinary exceptions into recoverable tool feedback."""

    def handle(self, call: ToolCall, next_handler: ToolHandler) -> ToolResult:
        try:
            return next_handler(call)
        except Exception as exc:
            return _failure_result(
                call,
                error_code=TOOL_EXECUTION_ERROR,
                content="The requested tool failed unexpectedly.",
                metadata={"exception_type": type(exc).__name__},
            )


class ToolContractMiddleware:
    """Reject invalid or uncorrelated values returned by a tool backend."""

    def handle(self, call: ToolCall, next_handler: ToolHandler) -> ToolResult:
        result = next_handler(call)
        if not isinstance(result, ToolResult):
            return _failure_result(
                call,
                error_code=TOOL_CONTRACT_ERROR,
                content=f"Tool {call.name!r} returned an invalid result.",
            )
        if result.call_id != call.id or result.tool_name != call.name:
            return _failure_result(
                call,
                error_code=TOOL_CONTRACT_ERROR,
                content=f"Tool {call.name!r} returned an uncorrelated result.",
            )
        return result


class ToolResultEnvelopeMiddleware:
    """Whitelist metadata and bound the complete JSON message sent to a model."""

    def __init__(
        self,
        *,
        max_model_chars: int,
        compactor: TextCompactionStrategy | None = None,
    ) -> None:
        if max_model_chars < _MIN_MODEL_CHARS:
            raise ValueError(
                f"max_model_chars must be at least {_MIN_MODEL_CHARS}"
            )
        self._max_model_chars = max_model_chars
        self._compactor = (
            DiagnosticOutputCompactor() if compactor is None else compactor
        )

    def handle(self, call: ToolCall, next_handler: ToolHandler) -> ToolResult:
        result = next_handler(call)
        visible_metadata = _select_model_metadata(result.metadata)
        candidate = replace(result, model_metadata=visible_metadata)
        if len(candidate.model_content()) <= self._max_model_chars:
            return candidate

        candidate = _fit_metadata_without_changing_content(
            result,
            visible_metadata,
            max_chars=self._max_model_chars,
        )
        if len(candidate.model_content()) <= self._max_model_chars:
            return candidate
        return self._compact_result(result)

    def _compact_result(self, result: ToolResult) -> ToolResult:
        original_chars = len(result.content)
        available = min(
            max(_MIN_COMPACTED_CONTENT_CHARS, original_chars - 1),
            self._max_model_chars,
        )

        while available >= _MIN_COMPACTED_CONTENT_CHARS:
            compacted = self._compactor.compact(
                result.content,
                max_chars=available,
            )
            host_metadata = {
                **result.metadata,
                "content_truncated": True,
                "content_original_chars": original_chars,
                "content_returned_chars": len(compacted.content),
            }
            model_metadata = _select_model_metadata(host_metadata)
            model_metadata = _fit_metadata_for_compacted_content(
                result,
                model_metadata,
                max_chars=self._max_model_chars,
            )
            candidate = replace(
                result,
                content=compacted.content,
                metadata=host_metadata,
                model_metadata=model_metadata,
            )
            encoded_chars = len(candidate.model_content())
            if encoded_chars <= self._max_model_chars:
                return candidate
            available -= max(1, encoded_chars - self._max_model_chars)

        raise RuntimeError("tool result envelope could not fit the configured budget")


def _bind(middleware: ToolMiddleware, next_handler: ToolHandler) -> ToolHandler:
    def bound(call: ToolCall) -> ToolResult:
        return middleware.handle(call, next_handler)

    return bound


def _fit_metadata_without_changing_content(
    result: ToolResult,
    metadata: Mapping[str, Any],
    *,
    max_chars: int,
) -> ToolResult:
    selected: dict[str, Any] = {}
    for key, value in metadata.items():
        proposed = {**selected, key: value}
        candidate = replace(result, model_metadata=proposed)
        if len(candidate.model_content()) <= max_chars:
            selected = proposed
    return replace(result, model_metadata=selected)


def _fit_metadata_for_compacted_content(
    result: ToolResult,
    metadata: Mapping[str, Any],
    *,
    max_chars: int,
) -> Mapping[str, Any]:
    required_keys = {
        "content_truncated",
        "content_original_chars",
        "content_returned_chars",
    }
    selected = {
        key: value for key, value in metadata.items() if key in required_keys
    }
    for key, value in metadata.items():
        if key in required_keys:
            continue
        proposed = {**selected, key: value}
        probe = replace(result, content="", model_metadata=proposed)
        if len(probe.model_content()) < max_chars:
            selected = proposed
    return selected


def _select_model_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in _MODEL_METADATA_ORDER:
        if key not in metadata:
            continue
        normalized = _normalize_model_metadata(key, metadata[key])
        if normalized is not _INVALID_METADATA:
            selected[key] = normalized
    return selected


_INVALID_METADATA = object()
_MODEL_METADATA_ORDER = (
    "content_truncated",
    "content_original_chars",
    "content_returned_chars",
    "output_id",
    "truncated",
    "original_chars",
    "returned_chars",
    "included_ranges",
    "exit_code",
    "timed_out",
    "duration_seconds",
    "offset",
    "end",
    "total_chars",
    "has_more",
    "next_offset",
)


def _normalize_model_metadata(key: str, value: Any) -> Any:
    if key == "output_id":
        if isinstance(value, str) and 0 < len(value) <= _MAX_MODEL_STRING_CHARS:
            return value
        return _INVALID_METADATA
    if key in {"truncated", "timed_out", "has_more", "content_truncated"}:
        return value if isinstance(value, bool) else _INVALID_METADATA
    if key == "exit_code":
        if value is None or _is_bounded_integer(value, allow_negative=True):
            return value
        return _INVALID_METADATA
    if key == "next_offset":
        if value is None or _is_bounded_integer(value):
            return value
        return _INVALID_METADATA
    if key == "duration_seconds":
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and 0 <= value <= _MAX_MODEL_INTEGER
        ):
            return round(float(value), 6)
        return _INVALID_METADATA
    if key == "included_ranges":
        return _normalize_ranges(value)
    return value if _is_bounded_integer(value) else _INVALID_METADATA


def _is_bounded_integer(value: Any, *, allow_negative: bool = False) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    lower = -_MAX_MODEL_INTEGER if allow_negative else 0
    return lower <= value <= _MAX_MODEL_INTEGER


def _normalize_ranges(value: Any) -> Any:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_INCLUDED_RANGES:
        return _INVALID_METADATA
    normalized: list[tuple[int, int]] = []
    for current in value:
        if not isinstance(current, (list, tuple)) or len(current) != 2:
            return _INVALID_METADATA
        start, end = current
        if (
            not _is_bounded_integer(start)
            or not _is_bounded_integer(end)
            or start > end
        ):
            return _INVALID_METADATA
        normalized.append((start, end))
    return tuple(normalized)


def _failure_result(
    call: ToolCall,
    *,
    error_code: str,
    content: str,
    metadata: Mapping[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=content,
        error_code=error_code,
        metadata={} if metadata is None else metadata,
    )
