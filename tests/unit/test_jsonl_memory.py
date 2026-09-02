import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from minicoder.adapters.jsonl_memory import JsonlProjectMemoryStore
from minicoder.domain.errors import DomainValidationError, MemoryPersistenceError
from minicoder.domain.memory import ProjectMemoryRecord

FIXED_TIME = datetime(2026, 8, 31, 9, 30, tzinfo=timezone.utc)


def test_jsonl_memory_appends_versioned_sanitized_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    storage = tmp_path / "private-memory"
    workspace.mkdir()
    store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=storage,
        sensitive_values=("secret-api-key",),
    )

    store.append(
        ProjectMemoryRecord(
            recorded_at=FIXED_TIME,
            summary="Changed app.py without storing secret-api-key.",
        )
    )

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload == {
        "recorded_at": "2026-08-31T09:30:00.000Z",
        "schema_version": 1,
        "summary": "Changed app.py without storing <redacted>.",
    }
    assert store.path.parent == storage
    assert workspace not in store.path.parents


def test_jsonl_memory_uses_workspace_identity_and_isolates_projects(
    tmp_path: Path,
) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    storage = tmp_path / "memory"
    first_workspace.mkdir()
    second_workspace.mkdir()

    first = JsonlProjectMemoryStore(
        workspace=first_workspace,
        storage_root=storage,
    )
    same_first = JsonlProjectMemoryStore(
        workspace=first_workspace / ".",
        storage_root=storage,
    )
    second = JsonlProjectMemoryStore(
        workspace=second_workspace,
        storage_root=storage,
    )

    assert first.path == same_first.path
    assert first.path != second.path


def test_jsonl_memory_loads_all_valid_records_and_skips_bad_lines(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=tmp_path / "memory",
    )
    lines = [
        "not-json",
        json.dumps({"schema_version": 99, "summary": "unknown"}),
        json.dumps(
            {
                "schema_version": 1,
                "recorded_at": "2026-08-29T09:30:00.000Z",
                "summary": "old valid record",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "recorded_at": "2026-08-30T09:30:00.000Z",
                "summary": "middle valid record",
            }
        ),
        json.dumps(
            {
                "schema_version": 1,
                "recorded_at": "2026-08-31T09:30:00.000Z",
                "summary": "latest valid record",
            }
        ),
    ]
    store.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = store.load_all()

    assert [record.summary for record in records] == [
        "old valid record",
        "middle valid record",
        "latest valid record",
    ]
    assert all(record.recorded_at.tzinfo is not None for record in records)


def test_jsonl_memory_skips_one_invalid_utf8_line_without_losing_neighbors(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=tmp_path / "memory",
    )
    first = (
        b'{"schema_version":1,"recorded_at":'
        b'"2026-08-30T09:30:00Z","summary":"first"}\n'
    )
    invalid = b'{"summary":"\xff"}\n'
    second = (
        b'{"schema_version":1,"recorded_at":'
        b'"2026-08-31T09:30:00Z","summary":"second"}\n'
    )
    store.path.write_bytes(first + invalid + second)

    records = store.load_all()

    assert [record.summary for record in records] == ["first", "second"]


def test_jsonl_memory_bounds_loaded_and_appended_summaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = JsonlProjectMemoryStore(
        workspace=workspace,
        storage_root=tmp_path / "memory",
        max_summary_chars=80,
    )

    store.append(
        ProjectMemoryRecord(
            recorded_at=FIXED_TIME,
            summary="begin-" + "x" * 200 + "-end",
        )
    )

    record = store.load_all()[0]
    assert len(record.summary) == 80
    assert record.summary.startswith("begin-")
    assert record.summary.endswith("-end")
    assert "memory truncated" in record.summary


def test_jsonl_memory_rejects_non_directory_storage_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    storage_file = tmp_path / "memory-file"
    storage_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(MemoryPersistenceError, match="directory"):
        JsonlProjectMemoryStore(
            workspace=workspace,
            storage_root=storage_file,
        )


def test_project_memory_record_requires_aware_time_and_text() -> None:
    with pytest.raises(DomainValidationError, match="timezone-aware"):
        ProjectMemoryRecord(
            recorded_at=datetime(2026, 8, 31),
            summary="valid",
        )
    with pytest.raises(DomainValidationError, match="non-blank"):
        ProjectMemoryRecord(recorded_at=FIXED_TIME, summary="   ")
