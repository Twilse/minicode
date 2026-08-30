"""Workspace-scoped JSON Lines persistence for bounded project memories."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from minicoder.domain.errors import MemoryPersistenceError
from minicoder.domain.memory import ProjectMemoryRecord

MEMORY_SCHEMA_VERSION = 1
DEFAULT_MAX_RECORDS = 8
DEFAULT_MAX_SUMMARY_CHARS = 1_200


class JsonlProjectMemoryStore:
    """Persist recent semantic summaries outside the inspected workspace."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        storage_root: str | Path | None = None,
        sensitive_values: Sequence[str] = (),
        max_records: int = DEFAULT_MAX_RECORDS,
        max_summary_chars: int = DEFAULT_MAX_SUMMARY_CHARS,
    ) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool):
            raise ValueError("memory max_records must be a positive integer")
        if max_records <= 0:
            raise ValueError("memory max_records must be a positive integer")
        if (
            not isinstance(max_summary_chars, int)
            or isinstance(max_summary_chars, bool)
            or max_summary_chars <= 0
        ):
            raise ValueError(
                "memory max_summary_chars must be a positive integer"
            )

        resolved_workspace = Path(workspace).expanduser().resolve()
        root = (
            Path.home() / ".minicoder" / "memory"
            if storage_root is None
            else Path(storage_root).expanduser()
        )
        digest = hashlib.sha256(
            str(resolved_workspace).encode("utf-8")
        ).hexdigest()[:24]

        try:
            root.mkdir(parents=True, exist_ok=True)
            if not root.is_dir():
                raise OSError("memory storage root is not a directory")
            _best_effort_chmod(root, 0o700)
        except OSError as exc:
            raise MemoryPersistenceError(
                "could not prepare the project memory directory"
            ) from exc

        self._path = root / f"{digest}.jsonl"
        if self._path.exists() and not self._path.is_file():
            raise MemoryPersistenceError(
                "project memory path is not a regular file"
            )
        if self._path.exists():
            _best_effort_chmod(self._path, 0o600)
        self._sensitive_values = tuple(
            value for value in sensitive_values if isinstance(value, str) and value
        )
        self._max_records = max_records
        self._max_summary_chars = max_summary_chars

    @property
    def path(self) -> Path:
        return self._path

    def load_recent(self) -> Sequence[ProjectMemoryRecord]:
        """Load recent valid lines while ignoring independent corrupt records."""

        if not self._path.exists():
            return ()

        records: deque[ProjectMemoryRecord] = deque(maxlen=self._max_records)
        try:
            with self._path.open("rb") as memory_file:
                for raw_line in memory_file:
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                    record = self._parse_record(line)
                    if record is not None:
                        records.append(record)
        except OSError as exc:
            raise MemoryPersistenceError(
                "could not read project memory"
            ) from exc
        return tuple(records)

    def append(self, record: ProjectMemoryRecord) -> None:
        """Append one sanitized record as an independently parseable line."""

        if not isinstance(record, ProjectMemoryRecord):
            raise TypeError("project memory store requires ProjectMemoryRecord")
        safe_summary = self._bound_and_redact(record.summary)
        timestamp = (
            record.recorded_at.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "recorded_at": timestamp,
            "summary": safe_summary,
        }
        line = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        existed = self._path.exists()
        try:
            with self._path.open("a", encoding="utf-8", newline="") as memory_file:
                memory_file.write(f"{line}\n")
            if not existed:
                _best_effort_chmod(self._path, 0o600)
        except OSError as exc:
            raise MemoryPersistenceError(
                "could not persist project memory"
            ) from exc

    def _parse_record(self, line: str) -> ProjectMemoryRecord | None:
        try:
            payload: Any = json.loads(line)
            if not isinstance(payload, dict):
                return None
            if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
                return None
            raw_timestamp = payload.get("recorded_at")
            raw_summary = payload.get("summary")
            if not isinstance(raw_timestamp, str) or not isinstance(
                raw_summary,
                str,
            ):
                return None
            normalized_timestamp = (
                f"{raw_timestamp[:-1]}+00:00"
                if raw_timestamp.endswith("Z")
                else raw_timestamp
            )
            recorded_at = datetime.fromisoformat(normalized_timestamp)
            return ProjectMemoryRecord(
                recorded_at=recorded_at,
                summary=self._bound_and_redact(raw_summary),
            )
        except (json.JSONDecodeError, ValueError):
            return None

    def _bound_and_redact(self, text: str) -> str:
        redacted = text
        for sensitive_value in self._sensitive_values:
            redacted = redacted.replace(sensitive_value, "<redacted>")
        return _bounded_text(redacted, self._max_summary_chars)


def _bounded_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n...[memory truncated]...\n"
    remaining = limit - len(marker)
    if remaining <= 0:
        return text[:limit]
    head_chars = remaining * 7 // 10
    tail_chars = remaining - head_chars
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        path.chmod(mode)
    except OSError:
        pass
