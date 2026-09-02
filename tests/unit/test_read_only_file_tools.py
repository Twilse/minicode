from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

import minicoder.tools.files as file_tools
from minicoder.domain.models import ToolCall, ToolResult
from minicoder.tools.files import (
    BINARY_FILE,
    FILE_ALREADY_EXISTS,
    FILE_CONTENT_MISMATCH,
    FILE_IO_ERROR,
    INVALID_OFFSET,
    NO_CHANGES,
    NOT_A_DIRECTORY,
    PARENT_DIRECTORY_NOT_FOUND,
    REPLACEMENT_BATCH_TOO_LARGE,
    TEXT_NOT_FOUND,
    TEXT_NOT_UNIQUE,
    CreateFileTool,
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    WriteFileTool,
)
from minicoder.tools.registry import INVALID_ARGUMENTS, ToolRegistry
from minicoder.tools.safety import PATH_OUTSIDE_WORKSPACE, WorkspacePathPolicy


def _registry(workspace: Path, *, read_limit: int = 12_000) -> ToolRegistry:
    paths = WorkspacePathPolicy(workspace)
    return ToolRegistry(
        (
            ListFilesTool(paths),
            ReadFileTool(paths, max_chars=read_limit),
            SearchTextTool(paths),
            CreateFileTool(paths),
            WriteFileTool(paths),
            ReplaceTextTool(paths),
        )
    )


def _execute(
    registry: ToolRegistry,
    tool_name: str,
    arguments: dict[str, object],
) -> ToolResult:
    return registry.execute(
        ToolCall(
            id=f"call-{tool_name}",
            name=tool_name,
            arguments_json=json.dumps(arguments),
        )
    )


def test_list_files_is_sorted_recursive_and_ignores_generated_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "z.txt").write_text("z", encoding="utf-8")
    (workspace / "a.py").write_text("a", encoding="utf-8")
    source = workspace / "src"
    source.mkdir()
    (source / "main.py").write_text("main", encoding="utf-8")
    ignored = workspace / ".git"
    ignored.mkdir()
    (ignored / "config").write_text("secret", encoding="utf-8")

    result = _execute(_registry(workspace), "list_files", {})

    assert result.ok is True
    assert result.content.splitlines() == ["a.py", "src/", "src/main.py", "z.txt"]
    assert result.metadata == {
        "path": ".",
        "entries": 4,
        "max_depth": 3,
        "truncated": False,
    }


def test_list_files_honors_depth_and_entry_limits(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "src" / "package"
    nested.mkdir(parents=True)
    (nested / "main.py").write_text("content", encoding="utf-8")
    (workspace / "top.txt").write_text("content", encoding="utf-8")
    registry = _registry(workspace)

    shallow = _execute(
        registry,
        "list_files",
        {"max_depth": 1, "max_entries": 20},
    )
    limited = _execute(
        registry,
        "list_files",
        {"max_depth": 8, "max_entries": 2},
    )

    assert shallow.content.splitlines() == ["src/", "top.txt"]
    assert limited.metadata["truncated"] is True
    assert "entry limit reached" in limited.content
    assert limited.content.splitlines()[:2] == ["src/", "src/package/"]


def test_list_files_reports_a_non_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "main.py").write_text("content", encoding="utf-8")

    result = _execute(_registry(workspace), "list_files", {"path": "main.py"})

    assert result.ok is False
    assert result.error_code == NOT_A_DIRECTORY


def test_list_and_search_do_not_follow_a_symlink_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("needle", encoding="utf-8")
    link = workspace / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")
    registry = _registry(workspace)

    listed = _execute(registry, "list_files", {})
    searched = _execute(registry, "search_text", {"query": "needle"})

    assert "linked" not in listed.content
    assert "secret.txt" not in searched.content
    assert searched.metadata["files_scanned"] == 0


def test_read_file_paginates_by_unicode_characters(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "message.txt").write_text("ab你好吗xyz", encoding="utf-8")
    registry = _registry(workspace, read_limit=4)

    first = _execute(
        registry,
        "read_file",
        {"path": "message.txt", "offset": 0, "limit": 4},
    )
    second = _execute(
        registry,
        "read_file",
        {"path": "message.txt", "offset": first.metadata["next_offset"]},
    )

    assert first.ok is True
    assert first.content == (
        "[path='message.txt' chars=0:4 has_more=true next_offset=4]\nab你好"
    )
    assert first.metadata["returned_chars"] == 4
    assert second.content == (
        "[path='message.txt' chars=4:8 has_more=false]\n吗xyz"
    )
    assert second.metadata["next_offset"] is None


def test_read_file_rejects_an_offset_beyond_end(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "short.txt").write_text("abc", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "read_file",
        {"path": "short.txt", "offset": 4},
    )

    assert result.ok is False
    assert result.error_code == INVALID_OFFSET


def test_read_file_rejects_binary_and_invalid_utf8_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "null.bin").write_bytes(b"hello\x00world")
    (workspace / "invalid.bin").write_bytes(b"\xff\xfe")
    registry = _registry(workspace)

    null_result = _execute(registry, "read_file", {"path": "null.bin"})
    invalid_result = _execute(registry, "read_file", {"path": "invalid.bin"})

    assert null_result.error_code == BINARY_FILE
    assert invalid_result.error_code == BINARY_FILE


def test_read_file_cannot_escape_workspace_and_schema_caps_limit(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = _registry(workspace, read_limit=20)

    escaped = _execute(registry, "read_file", {"path": "../secret.txt"})
    too_large = _execute(
        registry,
        "read_file",
        {"path": "main.py", "limit": 21},
    )

    assert escaped.error_code == PATH_OUTSIDE_WORKSPACE
    assert too_large.error_code == INVALID_ARGUMENTS


def test_search_text_is_literal_sorted_and_optionally_case_insensitive(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "z.txt").write_text("needle\nNEEDLE\na.b\n", encoding="utf-8")
    (workspace / "a.txt").write_text("needle first\naxb\n", encoding="utf-8")
    registry = _registry(workspace)

    sensitive = _execute(registry, "search_text", {"query": "needle"})
    insensitive = _execute(
        registry,
        "search_text",
        {"query": "needle", "case_sensitive": False},
    )
    literal = _execute(registry, "search_text", {"query": "a.b"})

    assert sensitive.content.splitlines()[:2] == [
        "a.txt:1: needle first",
        "z.txt:1: needle",
    ]
    assert "z.txt:2: NEEDLE" in insensitive.content
    assert "z.txt:3: a.b" in literal.content
    assert "a.txt:2: axb" not in literal.content


def test_search_text_reports_skipped_files_in_model_visible_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "text.txt").write_text("nothing here", encoding="utf-8")
    (workspace / "binary.bin").write_bytes(b"needle\x00data")

    result = _execute(_registry(workspace), "search_text", {"query": "needle"})

    assert result.ok is True
    assert "No matches" in result.content
    assert "files_scanned=1 files_skipped=1" in result.content
    assert result.metadata["files_skipped"] == 1


def test_search_text_only_reports_truncation_when_more_matches_exist(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "matches.txt"
    registry = _registry(workspace)

    target.write_text("hit\nhit\n", encoding="utf-8")
    exact = _execute(
        registry,
        "search_text",
        {"query": "hit", "max_matches": 2},
    )
    target.write_text("hit\nhit\nhit\n", encoding="utf-8")
    extra = _execute(
        registry,
        "search_text",
        {"query": "hit", "max_matches": 2},
    )

    assert exact.metadata["truncated"] is False
    assert extra.metadata["truncated"] is True
    assert extra.metadata["matches"] == 2
    assert "match limit reached" in extra.content


def test_create_file_makes_parents_without_overwriting_existing_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = _registry(workspace)

    created = _execute(
        registry,
        "create_file",
        {"path": "src/package/main.py", "content": "print('first')\n"},
    )
    repeated = _execute(
        registry,
        "create_file",
        {"path": "src/package/main.py", "content": "print('second')\n"},
    )

    target = workspace / "src" / "package" / "main.py"
    assert created.ok is True
    assert created.metadata["parent_directories_created"] is True
    assert repeated.error_code == FILE_ALREADY_EXISTS
    assert target.read_text(encoding="utf-8") == "print('first')\n"


def test_create_file_can_require_the_parent_to_exist(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = _execute(
        _registry(workspace),
        "create_file",
        {
            "path": "missing/main.py",
            "content": "content",
            "create_parents": False,
        },
    )

    assert result.error_code == PARENT_DIRECTORY_NOT_FOUND
    assert not (workspace / "missing").exists()


def test_write_file_populates_an_existing_empty_file_and_preserves_mode(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "empty.py"
    target.write_text("", encoding="utf-8")
    os.chmod(target, 0o744)

    result = _execute(
        _registry(workspace),
        "write_file",
        {
            "path": "empty.py",
            "expected_content": "",
            "content": "print('ready')\n",
        },
    )

    assert result.ok is True
    assert result.metadata == {
        "path": "empty.py",
        "old_characters": 0,
        "new_characters": 15,
    }
    assert target.read_text(encoding="utf-8") == "print('ready')\n"
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o744


def test_write_file_rejects_stale_expected_content_without_modifying_file(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "config.py"
    target.write_text("timeout = 30\n", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "write_file",
        {
            "path": "config.py",
            "expected_content": "timeout = 10\n",
            "content": "timeout = 60\n",
        },
    )

    assert result.error_code == FILE_CONTENT_MISMATCH
    assert "Read the file again" in result.content
    assert target.read_text(encoding="utf-8") == "timeout = 30\n"


def test_write_file_requires_an_existing_file_and_exact_schema(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = _registry(workspace)

    missing = _execute(
        registry,
        "write_file",
        {
            "path": "missing.py",
            "expected_content": "",
            "content": "new\n",
        },
    )
    missing_guard = _execute(
        registry,
        "write_file",
        {"path": "missing.py", "content": "new\n"},
    )

    assert missing.error_code == file_tools.FILE_NOT_FOUND
    assert missing_guard.error_code == INVALID_ARGUMENTS
    assert not (workspace / "missing.py").exists()


def test_replace_text_changes_one_unique_match_and_preserves_file_mode(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "script.py"
    target.write_text("value = 'old'\r\nprint(value)\r\n", encoding="utf-8", newline="")
    os.chmod(target, 0o744)

    result = _execute(
        _registry(workspace),
        "replace_text",
        {"path": "script.py", "old_text": "'old'", "new_text": "'new'"},
    )

    assert result.ok is True
    assert target.read_bytes() == b"value = 'new'\r\nprint(value)\r\n"
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o744


def test_replace_text_applies_multiple_related_edits_in_one_atomic_batch(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "cli.py"
    target.write_text(
        "command = 'add'\npriority = 'medium'\ncolor = False\n",
        encoding="utf-8",
    )

    result = _execute(
        _registry(workspace),
        "replace_text",
        {
            "path": "cli.py",
            "replacements": [
                {"old_text": "'add'", "new_text": "'edit'"},
                {"old_text": "'medium'", "new_text": "'high'"},
                {"old_text": "False", "new_text": "True"},
            ],
        },
    )

    assert result.ok is True
    assert result.metadata["replacement_count"] == 3
    assert target.read_text(encoding="utf-8") == (
        "command = 'edit'\npriority = 'high'\ncolor = True\n"
    )


def test_replace_text_batch_failure_leaves_every_edit_unwritten(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "cli.py"
    original = "first = 1\nsecond = 2\n"
    target.write_text(original, encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {
            "path": "cli.py",
            "replacements": [
                {"old_text": "first = 1", "new_text": "first = 10"},
                {"old_text": "missing = 3", "new_text": "missing = 30"},
            ],
        },
    )

    assert result.error_code == TEXT_NOT_FOUND
    assert "Replacement 2" in result.content
    assert target.read_text(encoding="utf-8") == original


def test_replace_text_rejects_mixed_single_and_batch_arguments(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "content.txt").write_text("old", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {
            "path": "content.txt",
            "old_text": "old",
            "new_text": "new",
            "replacements": [{"old_text": "old", "new_text": "new"}],
        },
    )

    assert result.error_code == INVALID_ARGUMENTS
    assert (workspace / "content.txt").read_text(encoding="utf-8") == "old"


def test_replace_text_rejects_a_batch_over_the_combined_character_budget(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "content.txt"
    target.write_text("x", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {
            "path": "content.txt",
            "replacements": [
                {"old_text": "x", "new_text": "y" * 100_001},
                {"old_text": "y", "new_text": "z" * 100_001},
            ],
        },
    )

    assert result.error_code == REPLACEMENT_BATCH_TOO_LARGE
    assert target.read_text(encoding="utf-8") == "x"


@pytest.mark.parametrize(
    ("old_text", "error_code"),
    [
        ("missing", TEXT_NOT_FOUND),
        ("repeat", TEXT_NOT_UNIQUE),
    ],
)
def test_replace_text_rejects_zero_or_multiple_matches_without_modifying_file(
    tmp_path: Path,
    old_text: str,
    error_code: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "content.txt"
    original = "repeat once, repeat twice"
    target.write_text(original, encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {"path": "content.txt", "old_text": old_text, "new_text": "changed"},
    )

    assert result.error_code == error_code
    assert target.read_text(encoding="utf-8") == original


def test_replace_text_treats_overlapping_matches_as_ambiguous(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "content.txt"
    target.write_text("aaa", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {"path": "content.txt", "old_text": "aa", "new_text": "b"},
    )

    assert result.error_code == TEXT_NOT_UNIQUE
    assert target.read_text(encoding="utf-8") == "aaa"


def test_replace_text_rejects_an_identical_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "content.txt").write_text("same", encoding="utf-8")

    result = _execute(
        _registry(workspace),
        "replace_text",
        {"path": "content.txt", "old_text": "same", "new_text": "same"},
    )

    assert result.error_code == NO_CHANGES


def test_replace_text_keeps_original_when_atomic_swap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "content.txt"
    target.write_text("before", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated swap failure")

    monkeypatch.setattr(file_tools.os, "replace", fail_replace)
    result = _execute(
        _registry(workspace),
        "replace_text",
        {"path": "content.txt", "old_text": "before", "new_text": "after"},
    )

    assert result.error_code == FILE_IO_ERROR
    assert target.read_text(encoding="utf-8") == "before"
    assert list(workspace.glob(".content.txt.*.tmp")) == []
