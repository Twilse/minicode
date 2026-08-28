"""Application-specific exception hierarchy."""


class MiniCoderError(Exception):
    """Base class for errors that MiniCoder can present to a user."""


class ConfigurationError(MiniCoderError):
    """Raised when startup configuration is missing or invalid."""


class DomainValidationError(MiniCoderError, ValueError):
    """Raised when an internal domain value violates an invariant."""


class ToolRegistrationError(MiniCoderError, ValueError):
    """Raised when a local tool cannot be safely added to the registry."""


class UnsupportedPlatformError(ConfigurationError):
    """Raised when no safe process adapter exists for the current platform."""


class ModelError(MiniCoderError):
    """Base class for failures at the language-model boundary."""


class ModelAccessError(ModelError):
    """Raised when model credentials or permissions are rejected."""


class ModelRateLimitError(ModelError):
    """Raised when the model service asks the client to slow down."""


class ModelConnectionError(ModelError):
    """Raised when the model service cannot be reached."""


class ModelServiceError(ModelError):
    """Raised for retryable server-side model service failures."""


class ModelRequestError(ModelError):
    """Raised when a model request is invalid or rejected without retry."""


class ModelResponseError(ModelError):
    """Raised when a model response violates the expected protocol shape."""
