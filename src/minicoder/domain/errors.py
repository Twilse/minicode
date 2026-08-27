"""Application-specific exception hierarchy."""


class MiniCoderError(Exception):
    """Base class for errors that MiniCoder can present to a user."""


class ConfigurationError(MiniCoderError):
    """Raised when startup configuration is missing or invalid."""


class DomainValidationError(MiniCoderError, ValueError):
    """Raised when an internal domain value violates an invariant."""


class UnsupportedPlatformError(ConfigurationError):
    """Raised when no safe process adapter exists for the current platform."""
