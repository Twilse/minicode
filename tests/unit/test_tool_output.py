from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from minicoder.tools.output import (
    ARTIFACT_NOT_FOUND,
    ARTIFACT_STORE_CLOSED,
    INVALID_ARTIFACT_RANGE,
    ArtifactStoreError,
    DiagnosticOutputCompactor,
    ToolOutputArtifactStore,
)


def test_artifact_store_reads_unicode_ranges_by_character(tmp_path: Path) -> None:
    store = ToolOutputArtifactStore(
        max_read_chars=4,
        temporary_parent=tmp_path,
    )
    output_id = store.save("ab你好吗xyz")

    first = store.read(output_id, offset=0, limit=4)
    second = store.read(output_id, offset=first.end, limit=4)

    assert output_id.startswith("out_")
    assert first.content == "ab你好"
    assert first.has_more is True
    assert second.content == "吗xyz"
    assert second.has_more is False
    assert second.total_chars == 8
    store.close()


def test_artifact_id_is_not_the_backing_filename(tmp_path: Path) -> None:
    store = ToolOutputArtifactStore(
        max_read_chars=100,
        temporary_parent=tmp_path,
    )
    output_id = store.save("complete output")
    files = list(store.root.iterdir())

    assert len(files) == 1
    assert output_id not in files[0].name
    if os.name != "nt":
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    store.close()


@pytest.mark.parametrize(
    ("output_id", "offset", "limit", "error_code"),
    [
        ("out_forged", 0, 4, ARTIFACT_NOT_FOUND),
        ("stored", -1, 4, INVALID_ARTIFACT_RANGE),
        ("stored", 0, 0, INVALID_ARTIFACT_RANGE),
        ("stored", 0, 5, INVALID_ARTIFACT_RANGE),
        ("stored", 9, 1, INVALID_ARTIFACT_RANGE),
    ],
)
def test_artifact_store_rejects_unknown_or_invalid_ranges(
    tmp_path: Path,
    output_id: str,
    offset: int,
    limit: int,
    error_code: str,
) -> None:
    store = ToolOutputArtifactStore(
        max_read_chars=4,
        temporary_parent=tmp_path,
    )
    stored_id = store.save("content")
    selected_id = stored_id if output_id == "stored" else output_id

    with pytest.raises(ArtifactStoreError) as captured:
        store.read(selected_id, offset=offset, limit=limit)

    assert captured.value.error_code == error_code
    store.close()


def test_closing_store_removes_files_and_invalidates_ids(tmp_path: Path) -> None:
    store = ToolOutputArtifactStore(
        max_read_chars=10,
        temporary_parent=tmp_path,
    )
    output_id = store.save("content")
    root = store.root

    store.close()

    assert not root.exists()
    with pytest.raises(ArtifactStoreError) as captured:
        store.read(output_id, offset=0, limit=1)
    assert captured.value.error_code == ARTIFACT_STORE_CLOSED


def test_compactor_leaves_short_output_unchanged() -> None:
    compacted = DiagnosticOutputCompactor().compact(
        "all tests passed\n",
        max_chars=100,
    )

    assert compacted.content == "all tests passed\n"
    assert compacted.truncated is False
    assert compacted.returned_chars == 17
    assert [(item.start, item.end) for item in compacted.included_ranges] == [
        (0, 17)
    ]


def test_compactor_keeps_head_diagnostic_window_and_tail() -> None:
    content = (
        "command header\n"
        + "setup line\n" * 40
        + "Traceback: important middle failure\n"
        + "context line\n" * 40
        + "final failure summary\n"
    )

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=500)

    assert compacted.truncated is True
    assert len(compacted.content) <= 500
    assert "command header" in compacted.content
    assert "Traceback: important middle failure" in compacted.content
    assert "final failure summary" in compacted.content
    assert "omitted" in compacted.content


def test_compactor_without_diagnostics_prefers_head_and_tail() -> None:
    content = "HEAD:" + "x" * 2_000 + ":TAIL"

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=300)

    assert len(compacted.content) <= 300
    assert compacted.content.startswith("HEAD:")
    assert compacted.content.endswith(":TAIL")
    assert "omitted" in compacted.content


def test_compactor_without_diagnostics_uses_thirty_seventy_payload_split() -> None:
    content = "H" * 2_000 + "T" * 2_000

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=1_000)

    head, tail = compacted.included_ranges
    head_chars = head.end - head.start
    tail_chars = tail.end - tail.start
    kept_chars = head_chars + tail_chars
    assert head_chars * 10 == kept_chars * 3
    assert tail_chars * 10 == kept_chars * 7


@pytest.mark.parametrize(
    "diagnostic",
    [
        "E   assert value == expected",
        "FAILED tests/test_app.py::test_case",
        "ModuleNotFoundError: missing_package",
        "TypeError: invalid value",
        "warning: deprecated option",
        "compiler fatal error",
        "npm ERR! command failed",
    ],
)
def test_compactor_recognizes_common_diagnostic_styles(diagnostic: str) -> None:
    content = "start\n" + "noise\n" * 100 + diagnostic + "\n" + "tail\n" * 100

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=420)

    assert diagnostic in compacted.content
    assert len(compacted.content) <= 420


def test_compactor_merges_overlapping_diagnostic_windows() -> None:
    content = (
        "head\n"
        + "noise\n" * 30
        + "ERROR first\n"
        + "between\n" * 2
        + "ERROR second\n"
        + "tail\n" * 50
    )

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=500)

    ranges = compacted.included_ranges
    assert all(left.end < right.start for left, right in zip(ranges, ranges[1:]))
    assert compacted.content.count("ERROR first") == 1
    assert compacted.content.count("ERROR second") == 1


def test_compactor_handles_one_million_characters_with_a_hard_budget() -> None:
    content = "begin\n" + "x" * 499_980 + "\nERROR middle\n" + "y" * 499_980 + "\nend"

    compacted = DiagnosticOutputCompactor().compact(content, max_chars=12_000)

    assert len(content) > 999_000
    assert len(compacted.content) <= 12_000
    assert compacted.original_chars == len(content)
    assert compacted.truncated is True
    assert "ERROR middle" in compacted.content
