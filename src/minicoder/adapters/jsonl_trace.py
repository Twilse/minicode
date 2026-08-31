"""Append-only JSON Lines observer for sanitized agent events."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path

from minicoder.domain.events import AgentEvent

TRACE_SCHEMA_VERSION = 1


class JsonlTraceSink:
    """Persist each event as one independently parseable UTF-8 JSON line."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        if not self._path.parent.is_dir():
            raise ValueError("trace parent directory must already exist")
        if self._path.exists() and not self._path.is_file():
            raise ValueError("trace path must be a regular file")

    @property
    def path(self) -> Path:
        return self._path

    def handle(self, event: AgentEvent) -> None:
        timestamp = (
            event.occurred_at.astimezone(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": event.run_id,
            "sequence": event.sequence,
            "timestamp": timestamp,
            "type": event.kind.value,
            "model_step": event.model_step,
            "details": {
                name: value
                for name, value in event.details.items()
                if not name.startswith("display_")
            },
        }
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._path.open("a", encoding="utf-8", newline="") as trace_file:
            trace_file.write(f"{line}\n")
