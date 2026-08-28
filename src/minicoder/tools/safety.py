"""Workspace capability boundaries shared by every local file tool."""

from __future__ import annotations

from pathlib import Path

from minicoder.domain.errors import ConfigurationError

INVALID_PATH = "INVALID_PATH"
PATH_OUTSIDE_WORKSPACE = "PATH_OUTSIDE_WORKSPACE"


class WorkspacePathError(ValueError):
    """A model-provided path that cannot be resolved inside the workspace."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class WorkspacePathPolicy:
    """Resolve untrusted relative paths under one canonical workspace root."""

    def __init__(self, workspace: str | Path) -> None:
        try:
            resolved = Path(workspace).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ConfigurationError("workspace cannot be resolved") from exc
        if not resolved.is_dir():
            raise ConfigurationError("workspace must be a directory")
        self._workspace = resolved

    @property
    def workspace(self) -> Path:
        """Return the canonical root used by every path check."""

        return self._workspace

    def resolve(self, raw_path: str) -> Path:
        """Return a canonical in-workspace path or reject the untrusted input."""

        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise WorkspacePathError(
                INVALID_PATH,
                "path must be non-empty text without null bytes",
            )

        supplied = Path(raw_path)
        if supplied.is_absolute():
            raise WorkspacePathError(
                PATH_OUTSIDE_WORKSPACE,
                f"absolute path {raw_path!r} is not allowed",
            )

        try:
            resolved = (self._workspace / supplied).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise WorkspacePathError(
                INVALID_PATH,
                f"path {raw_path!r} cannot be resolved",
            ) from exc

        try:
            resolved.relative_to(self._workspace)
        except ValueError as exc:
            raise WorkspacePathError(
                PATH_OUTSIDE_WORKSPACE,
                f"path {raw_path!r} resolves outside the workspace",
            ) from exc
        return resolved

    def display(self, path: Path) -> str:
        """Return one stable POSIX-style path relative to the workspace."""

        try:
            relative = path.resolve(strict=False).relative_to(self._workspace)
        except (OSError, RuntimeError, ValueError) as exc:
            raise WorkspacePathError(
                PATH_OUTSIDE_WORKSPACE,
                "resolved path is outside the workspace",
            ) from exc
        return "." if relative == Path(".") else relative.as_posix()
