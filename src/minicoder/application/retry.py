"""Explicit bounded retry policy for transient model-service failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, TypeVar

from minicoder.domain.errors import (
    DomainValidationError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelServiceError,
)

ResultT = TypeVar("ResultT")
RetryObserver = Callable[["RetryAttempt"], None]
Sleeper = Callable[[float], None]

_RETRYABLE_MODEL_ERRORS = (
    ModelConnectionError,
    ModelRateLimitError,
    ModelServiceError,
)


@dataclass(frozen=True, slots=True)
class RetryAttempt:
    """One scheduled retry after a classified transient failure."""

    retry_number: int  # One-based retry number; the initial call is not a retry.
    delay_seconds: float  # Backoff delay before the next operation attempt.
    error_type: str  # Stable exception class name without sensitive message text.


class RetryStrategy(Protocol):
    """Run an operation according to one replaceable retry policy."""

    def run(
        self,
        operation: Callable[[], ResultT],
        *,
        on_retry: RetryObserver | None = None,
    ) -> ResultT:
        """Return the operation value or re-raise its final exception."""

        ...


class ExponentialBackoffRetryStrategy:
    """Retry only connection, rate-limit, and server-side model errors."""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        initial_delay_seconds: float = 0.5,
        multiplier: float = 2.0,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise DomainValidationError(
                "retry max_retries must be a non-negative integer"
            )
        for name, value in (
            ("initial_delay_seconds", initial_delay_seconds),
            ("multiplier", multiplier),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
                or value <= 0
            ):
                raise DomainValidationError(
                    f"retry {name} must be finite and greater than zero"
                )
        self._max_retries = max_retries
        self._initial_delay_seconds = float(initial_delay_seconds)
        self._multiplier = float(multiplier)
        self._sleeper = sleeper

    def run(
        self,
        operation: Callable[[], ResultT],
        *,
        on_retry: RetryObserver | None = None,
    ) -> ResultT:
        retry_number = 0
        while True:
            try:
                return operation()
            except _RETRYABLE_MODEL_ERRORS as exc:
                if retry_number >= self._max_retries:
                    raise
                retry_number += 1
                delay = self._initial_delay_seconds * (
                    self._multiplier ** (retry_number - 1)
                )
                if on_retry is not None:
                    on_retry(
                        RetryAttempt(
                            retry_number=retry_number,
                            delay_seconds=delay,
                            error_type=type(exc).__name__,
                        )
                    )
                self._sleeper(delay)
