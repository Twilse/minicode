import pytest

from minicoder.application.retry import (
    ExponentialBackoffRetryStrategy,
    RetryAttempt,
)
from minicoder.domain.errors import (
    DomainValidationError,
    ModelAccessError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelRequestError,
    ModelResponseError,
    ModelServiceError,
)


def test_retry_strategy_uses_bounded_exponential_delays() -> None:
    outcomes: list[object] = [
        ModelConnectionError("offline"),
        ModelRateLimitError("slow down"),
        "done",
    ]
    sleeps: list[float] = []
    attempts: list[RetryAttempt] = []

    def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)

    strategy = ExponentialBackoffRetryStrategy(
        max_retries=2,
        initial_delay_seconds=0.25,
        multiplier=2,
        sleeper=sleeps.append,
    )

    result = strategy.run(operation, on_retry=attempts.append)

    assert result == "done"
    assert sleeps == [0.25, 0.5]
    assert attempts == [
        RetryAttempt(1, 0.25, "ModelConnectionError"),
        RetryAttempt(2, 0.5, "ModelRateLimitError"),
    ]


def test_retry_strategy_reraises_the_last_transient_error() -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise ModelServiceError("still unavailable")

    strategy = ExponentialBackoffRetryStrategy(
        max_retries=2,
        initial_delay_seconds=0.1,
        sleeper=sleeps.append,
    )

    with pytest.raises(ModelServiceError, match="still unavailable"):
        strategy.run(operation)

    assert calls == 3
    assert sleeps == [0.1, 0.2]


@pytest.mark.parametrize(
    "error",
    [
        ModelAccessError("bad credentials"),
        ModelRequestError("invalid request"),
        ModelResponseError("invalid response"),
    ],
)
def test_retry_strategy_does_not_retry_permanent_errors(error: Exception) -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise error

    strategy = ExponentialBackoffRetryStrategy(
        max_retries=2,
        sleeper=sleeps.append,
    )

    with pytest.raises(type(error), match=str(error)):
        strategy.run(operation)

    assert calls == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_retries": -1},
        {"initial_delay_seconds": 0},
        {"multiplier": float("inf")},
    ],
)
def test_retry_strategy_rejects_invalid_policy_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(DomainValidationError):
        ExponentialBackoffRetryStrategy(**kwargs)  # type: ignore[arg-type]
