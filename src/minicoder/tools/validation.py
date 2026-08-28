"""Strict JSON parsing and JSON Schema validation for model tool arguments."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, NoReturn

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from minicoder.domain.errors import ToolRegistrationError
from minicoder.domain.models import ToolDefinition


class ToolArgumentsError(ValueError):
    """Describe model-provided tool arguments that cannot be executed safely."""


def compile_arguments_validator(
    definition: ToolDefinition,
) -> Draft202012Validator:
    """Compile and validate one tool's schema when it is registered."""

    schema = deepcopy(dict(definition.parameters_schema))
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ToolRegistrationError(
            f"tool {definition.name!r} has an invalid parameters schema"
        ) from exc
    return Draft202012Validator(schema)


def parse_and_validate_arguments(
    arguments_json: str,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    """Return one valid JSON object or raise a concise model-facing error."""

    try:
        value = json.loads(
            arguments_json,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ToolArgumentsError(
            f"arguments are not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(value, dict):
        raise ToolArgumentsError("arguments JSON must be an object")

    errors = sorted(validator.iter_errors(value), key=_validation_error_sort_key)
    if errors:
        error = errors[0]
        raise ToolArgumentsError(
            f"arguments failed schema validation at "
            f"{_format_json_path(error.absolute_path)}: {error.message}"
        )
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolArgumentsError(f"arguments contain duplicate field {key!r}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> NoReturn:
    raise ToolArgumentsError(f"arguments contain non-JSON value {value}")


def _validation_error_sort_key(error: ValidationError) -> tuple[str, str]:
    path = "/".join(str(part) for part in error.absolute_path)
    return path, error.message


def _format_json_path(path: Iterable[str | int]) -> str:
    formatted = "$"
    for part in path:
        if isinstance(part, int):
            formatted += f"[{part}]"
        else:
            formatted += f"[{part!r}]"
    return formatted
