"""Deterministic, workspace-scoped file tools implemented with the standard library."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from minicoder.domain.errors import ConfigurationError
from minicoder.domain.models import ToolDefinition, ToolResult
from minicoder.tools.base import ToolCommand
from minicoder.tools.safety import WorkspacePathError, WorkspacePathPolicy

BINARY_FILE = "BINARY_FILE"
FILE_ALREADY_EXISTS = "FILE_ALREADY_EXISTS"
FILE_IO_ERROR = "FILE_IO_ERROR"
FILE_NOT_FOUND = "FILE_NOT_FOUND"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
INVALID_OFFSET = "INVALID_OFFSET"
NO_CHANGES = "NO_CHANGES"
NOT_A_DIRECTORY = "NOT_A_DIRECTORY"
NOT_A_FILE = "NOT_A_FILE"
PARENT_DIRECTORY_NOT_FOUND = "PARENT_DIRECTORY_NOT_FOUND"
PERMISSION_DENIED = "PERMISSION_DENIED"
TEXT_NOT_FOUND = "TEXT_NOT_FOUND"
TEXT_NOT_UNIQUE = "TEXT_NOT_UNIQUE"

_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_BINARY_SAMPLE_BYTES = 8192  # Bytes sampled to detect nulls in binary files.
_MAX_LIST_DEPTH = 8  # Maximum recursive directory-listing depth.
_MAX_LIST_ENTRIES = 500  # Maximum entries returned by one directory listing.
_MAX_SEARCH_LINE_CHARS = 240  # Maximum preview characters per matching line.
_MAX_SEARCH_FILE_BYTES = 1_000_000  # Largest file considered by text search.
_MAX_SEARCH_MATCHES = 100  # Maximum matches returned by one text search.
_MAX_MUTATION_FILE_BYTES = 1_000_000  # Largest file eligible for replacement.
_MAX_WRITE_CHARS = 200_000  # Maximum text accepted by one write argument.


class ListFilesTool:
    """List a bounded, deterministic view of one workspace directory."""

    def __init__(self, paths: WorkspacePathPolicy) -> None:
        self._paths = paths
        self._definition = ToolDefinition(
            name="list_files",
            description=(
                "List files and directories below a workspace-relative path. "
                "Results are sorted and common generated directories are ignored."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1, "default": "."},
                    "max_depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_LIST_DEPTH,
                        "default": 3,
                    },
                    "max_entries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_LIST_ENTRIES,
                        "default": 200,
                    },
                },
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        raw_path = command.arguments.get("path", ".")
        max_depth = command.arguments.get("max_depth", 3)
        max_entries = command.arguments.get("max_entries", 200)
        try:
            root = self._paths.resolve(raw_path)
            display_root = self._paths.display(root)
            if not root.exists():
                return _failure(
                    command,
                    FILE_NOT_FOUND,
                    f"Path {display_root!r} does not exist.",
                )
            if not root.is_dir():
                return _failure(
                    command,
                    NOT_A_DIRECTORY,
                    f"Path {display_root!r} is not a directory.",
                )
            entries, truncated = _list_entries(
                self._paths,
                root,
                max_depth=max_depth,
                max_entries=max_entries,
            )
        except WorkspacePathError as exc:
            return _failure(command, exc.error_code, str(exc))
        except PermissionError:
            return _failure(
                command,
                PERMISSION_DENIED,
                f"Cannot list path {raw_path!r}.",
            )
        except OSError as exc:
            return _io_failure(command, "list files", exc)

        content = "\n".join(entries) if entries else "(no entries)"
        if truncated:
            content += f"\n... {max_entries} entry limit reached; listing truncated."
        return _success(
            command,
            content,
            metadata={
                "path": display_root,
                "entries": len(entries),
                "max_depth": max_depth,
                "truncated": truncated,
            },
        )


class ReadFileTool:
    """Read one UTF-8 text file using bounded character ranges."""

    def __init__(self, paths: WorkspacePathPolicy, *, max_chars: int = 12_000) -> None:
        if max_chars <= 0:
            raise ConfigurationError("read_file max_chars must be greater than zero")
        self._paths = paths
        self._max_chars = max_chars
        self._definition = ToolDefinition(
            name="read_file",
            description=(
                "Read a UTF-8 text file by character offset. Use next_offset when "
                "has_more is true to continue reading."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_chars,
                        "default": max_chars,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        raw_path = command.arguments["path"]
        offset = command.arguments.get("offset", 0)
        limit = command.arguments.get("limit", self._max_chars)
        try:
            path = self._paths.resolve(raw_path)
            display_path = self._paths.display(path)
            if not path.exists():
                return _failure(
                    command,
                    FILE_NOT_FOUND,
                    f"File {display_path!r} does not exist.",
                )
            if not path.is_file():
                return _failure(
                    command,
                    NOT_A_FILE,
                    f"Path {display_path!r} is not a file.",
                )
            if _looks_binary(path):
                return _failure(
                    command,
                    BINARY_FILE,
                    f"File {display_path!r} is not UTF-8 text.",
                )
            chunk, has_more, offset_valid = _read_text_range(path, offset, limit)
        except WorkspacePathError as exc:
            return _failure(command, exc.error_code, str(exc))
        except UnicodeDecodeError:
            return _failure(
                command,
                BINARY_FILE,
                f"File {raw_path!r} is not UTF-8 text.",
            )
        except PermissionError:
            return _failure(
                command,
                PERMISSION_DENIED,
                f"Cannot read file {raw_path!r}.",
            )
        except OSError as exc:
            return _io_failure(command, "read file", exc)

        if not offset_valid:
            return _failure(
                command,
                INVALID_OFFSET,
                f"Offset {offset} is beyond the end of file {display_path!r}.",
            )

        end = offset + len(chunk)
        next_offset = end if has_more else None
        header = (
            f"[path={display_path!r} chars={offset}:{end} "
            f"has_more={'true' if has_more else 'false'}"
        )
        if next_offset is not None:
            header += f" next_offset={next_offset}"
        content = f"{header}]\n{chunk}"
        return _success(
            command,
            content,
            metadata={
                "path": display_path,
                "offset": offset,
                "returned_chars": len(chunk),
                "has_more": has_more,
                "next_offset": next_offset,
            },
        )


class SearchTextTool:
    """Search UTF-8 workspace files for a bounded literal string."""

    def __init__(self, paths: WorkspacePathPolicy) -> None:
        self._paths = paths
        self._definition = ToolDefinition(
            name="search_text",
            description=(
                "Search UTF-8 workspace files for a literal string and return "
                "sorted path:line matches."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string", "minLength": 1, "default": "."},
                    "case_sensitive": {"type": "boolean", "default": True},
                    "max_matches": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_SEARCH_MATCHES,
                        "default": 50,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        query = command.arguments["query"]
        raw_path = command.arguments.get("path", ".")
        case_sensitive = command.arguments.get("case_sensitive", True)
        max_matches = command.arguments.get("max_matches", 50)
        try:
            root = self._paths.resolve(raw_path)
            display_root = self._paths.display(root)
            if not root.exists():
                return _failure(
                    command,
                    FILE_NOT_FOUND,
                    f"Path {display_root!r} does not exist.",
                )
            if not (root.is_file() or root.is_dir()):
                return _failure(
                    command,
                    NOT_A_FILE,
                    f"Path {display_root!r} cannot be searched.",
                )
            matches, files_scanned, files_skipped, truncated = _search_files(
                self._paths,
                root,
                query=query,
                case_sensitive=case_sensitive,
                max_matches=max_matches,
            )
        except WorkspacePathError as exc:
            return _failure(command, exc.error_code, str(exc))
        except PermissionError:
            return _failure(
                command,
                PERMISSION_DENIED,
                f"Cannot search path {raw_path!r}.",
            )
        except OSError as exc:
            return _io_failure(command, "search files", exc)

        if matches:
            content = "\n".join(matches)
            if truncated:
                content += f"\n... {max_matches} match limit reached; search truncated."
        else:
            content = f"No matches for {query!r} under {display_root!r}."
        content += (
            f"\n[search files_scanned={files_scanned} "
            f"files_skipped={files_skipped} matches={len(matches)} "
            f"truncated={'true' if truncated else 'false'}]"
        )
        return _success(
            command,
            content,
            metadata={
                "path": display_root,
                "matches": len(matches),
                "files_scanned": files_scanned,
                "files_skipped": files_skipped,
                "truncated": truncated,
            },
        )


class CreateFileTool:
    """Create one new UTF-8 file without overwriting an existing path."""

    def __init__(self, paths: WorkspacePathPolicy) -> None:
        self._paths = paths
        self._definition = ToolDefinition(
            name="create_file",
            description=(
                "Create a new UTF-8 text file inside the workspace. Existing paths "
                "are never overwritten."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {
                        "type": "string",
                        "maxLength": _MAX_WRITE_CHARS,
                    },
                    "create_parents": {"type": "boolean", "default": True},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        raw_path = command.arguments["path"]
        content = command.arguments["content"]
        create_parents = command.arguments.get("create_parents", True)
        created = False
        try:
            path = self._paths.resolve(raw_path)
            display_path = self._paths.display(path)
            if path.exists() or path.is_symlink():
                return _failure(
                    command,
                    FILE_ALREADY_EXISTS,
                    f"Path {display_path!r} already exists; no content was changed.",
                )

            parent = path.parent
            parent_existed = parent.exists()
            if not parent_existed and not create_parents:
                return _failure(
                    command,
                    PARENT_DIRECTORY_NOT_FOUND,
                    f"Parent directory for {display_path!r} does not exist.",
                )
            if create_parents:
                parent.mkdir(parents=True, exist_ok=True)
            if not parent.is_dir():
                return _failure(
                    command,
                    NOT_A_DIRECTORY,
                    f"Parent of {display_path!r} is not a directory.",
                )

            with path.open("x", encoding="utf-8", newline="") as stream:
                created = True
                stream.write(content)
        except WorkspacePathError as exc:
            return _failure(command, exc.error_code, str(exc))
        except FileExistsError:
            return _failure(
                command,
                FILE_ALREADY_EXISTS,
                f"Path {raw_path!r} already exists; no content was changed.",
            )
        except PermissionError:
            if created:
                _best_effort_unlink(path)
            return _failure(
                command,
                PERMISSION_DENIED,
                f"Cannot create file {raw_path!r}.",
            )
        except OSError as exc:
            if created:
                _best_effort_unlink(path)
            return _io_failure(command, "create file", exc)

        return _success(
            command,
            f"Created {display_path!r} ({len(content)} characters).",
            metadata={
                "path": display_path,
                "characters_written": len(content),
                "parent_directories_created": not parent_existed,
            },
        )


class ReplaceTextTool:
    """Replace one uniquely matching literal string using an atomic file swap."""

    def __init__(self, paths: WorkspacePathPolicy) -> None:
        self._paths = paths
        self._definition = ToolDefinition(
            name="replace_text",
            description=(
                "Replace one exact literal string in a UTF-8 file. The operation "
                "is rejected unless old_text occurs exactly once."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "old_text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _MAX_WRITE_CHARS,
                    },
                    "new_text": {
                        "type": "string",
                        "maxLength": _MAX_WRITE_CHARS,
                    },
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, command: ToolCommand) -> ToolResult:
        raw_path = command.arguments["path"]
        old_text = command.arguments["old_text"]
        new_text = command.arguments["new_text"]
        try:
            path = self._paths.resolve(raw_path)
            display_path = self._paths.display(path)
            if not path.exists():
                return _failure(
                    command,
                    FILE_NOT_FOUND,
                    f"File {display_path!r} does not exist.",
                )
            if not path.is_file():
                return _failure(
                    command,
                    NOT_A_FILE,
                    f"Path {display_path!r} is not a file.",
                )
            if path.stat().st_size > _MAX_MUTATION_FILE_BYTES:
                return _failure(
                    command,
                    FILE_TOO_LARGE,
                    f"File {display_path!r} is too large for exact replacement.",
                )
            if _looks_binary(path):
                return _failure(
                    command,
                    BINARY_FILE,
                    f"File {display_path!r} is not UTF-8 text.",
                )

            original = _read_utf8_text(path)
            match_count = _count_overlapping_occurrences(original, old_text)
            if match_count == 0:
                return _failure(
                    command,
                    TEXT_NOT_FOUND,
                    (
                        f"old_text was not found in {display_path!r}; "
                        "no content was changed."
                    ),
                )
            if match_count > 1:
                return _failure(
                    command,
                    TEXT_NOT_UNIQUE,
                    (
                        f"old_text matched {match_count} times in {display_path!r}; "
                        "no content was changed."
                    ),
                )
            if old_text == new_text:
                return _failure(
                    command,
                    NO_CHANGES,
                    "old_text and new_text are identical; no content was changed.",
                )

            updated = original.replace(old_text, new_text, 1)
            _atomic_write_text(path, updated)
        except WorkspacePathError as exc:
            return _failure(command, exc.error_code, str(exc))
        except UnicodeDecodeError:
            return _failure(
                command,
                BINARY_FILE,
                f"File {raw_path!r} is not UTF-8 text.",
            )
        except PermissionError:
            return _failure(
                command,
                PERMISSION_DENIED,
                f"Cannot modify file {raw_path!r}.",
            )
        except OSError as exc:
            return _io_failure(command, "replace text", exc)

        return _success(
            command,
            f"Replaced one occurrence in {display_path!r}.",
            metadata={
                "path": display_path,
                "old_characters": len(old_text),
                "new_characters": len(new_text),
            },
        )


def _list_entries(
    paths: WorkspacePathPolicy,
    root: Path,
    *,
    max_depth: int,
    max_entries: int,
) -> tuple[list[str], bool]:
    entries: list[str] = []
    truncated = False

    def visit(directory: Path, depth: int) -> None:
        nonlocal truncated
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if len(entries) >= max_entries:
                truncated = True
                return
            raw_relative = child.relative_to(paths.workspace).as_posix()
            try:
                resolved = paths.resolve(raw_relative)
            except WorkspacePathError:
                continue
            if child.name in _IGNORED_DIRECTORIES and resolved.is_dir():
                continue
            if child.is_symlink():
                entries.append(f"{raw_relative}@")
            elif child.is_dir():
                entries.append(f"{raw_relative}/")
                if depth < max_depth:
                    visit(child, depth + 1)
                    if truncated:
                        return
            elif child.is_file():
                entries.append(raw_relative)

    visit(root, 1)
    return entries, truncated


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as stream:
        return b"\x00" in stream.read(_BINARY_SAMPLE_BYTES)


def _read_text_range(path: Path, offset: int, limit: int) -> tuple[str, bool, bool]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        remaining = offset
        while remaining:
            skipped = stream.read(min(remaining, 8192))
            if not skipped:
                return "", False, False
            remaining -= len(skipped)
        value = stream.read(limit + 1)
    return value[:limit], len(value) > limit, True


def _read_utf8_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _count_overlapping_occurrences(content: str, target: str) -> int:
    count = 0
    start = 0
    while True:
        match = content.find(target, start)
        if match < 0:
            return count
        count += 1
        start = match + 1


def _search_files(
    paths: WorkspacePathPolicy,
    root: Path,
    *,
    query: str,
    case_sensitive: bool,
    max_matches: int,
) -> tuple[list[str], int, int, bool]:
    candidates: Iterable[Path] = [root] if root.is_file() else _iter_files(paths, root)
    needle = query if case_sensitive else query.casefold()
    matches: list[str] = []
    files_scanned = 0
    files_skipped = 0

    for path in candidates:
        try:
            if path.stat().st_size > _MAX_SEARCH_FILE_BYTES or _looks_binary(path):
                files_skipped += 1
                continue
            content = _read_utf8_text(path)
            files_scanned += 1
            for line_number, line in enumerate(content.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                snippet = line
                if len(snippet) > _MAX_SEARCH_LINE_CHARS:
                    snippet = snippet[: _MAX_SEARCH_LINE_CHARS - 3] + "..."
                display_path = path.relative_to(paths.workspace).as_posix()
                matches.append(f"{display_path}:{line_number}: {snippet}")
                if len(matches) > max_matches:
                    return (
                        matches[:max_matches],
                        files_scanned,
                        files_skipped,
                        True,
                    )
        except (UnicodeDecodeError, OSError):
            files_skipped += 1
    return matches, files_scanned, files_skipped, False


def _iter_files(paths: WorkspacePathPolicy, root: Path) -> Iterator[Path]:
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        raw_relative = child.relative_to(paths.workspace).as_posix()
        try:
            resolved = paths.resolve(raw_relative)
        except WorkspacePathError:
            continue
        if child.name in _IGNORED_DIRECTORIES and resolved.is_dir():
            continue
        if child.is_symlink():
            if resolved.is_file():
                yield child
        elif child.is_dir():
            yield from _iter_files(paths, child)
        elif child.is_file():
            yield child


def _atomic_write_text(path: Path, content: str) -> None:
    """Write beside the target, then atomically replace it without a partial state."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
        os.chmod(temporary_path, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            _best_effort_unlink(temporary_path)


def _best_effort_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _success(
    command: ToolCommand,
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    return ToolResult(
        call_id=command.call_id,
        tool_name=command.tool_name,
        ok=True,
        content=content,
        metadata={} if metadata is None else metadata,
    )


def _failure(command: ToolCommand, error_code: str, content: str) -> ToolResult:
    return ToolResult(
        call_id=command.call_id,
        tool_name=command.tool_name,
        ok=False,
        content=content,
        error_code=error_code,
    )


def _io_failure(command: ToolCommand, operation: str, exc: OSError) -> ToolResult:
    return ToolResult(
        call_id=command.call_id,
        tool_name=command.tool_name,
        ok=False,
        content=f"Could not {operation} because of an operating-system error.",
        error_code=FILE_IO_ERROR,
        metadata={"exception_type": type(exc).__name__},
    )
