"""Session-scoped output artifacts and deterministic diagnostic compaction."""

from __future__ import annotations

import os
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from minicoder.domain.models import ProcessResult

ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
ARTIFACT_STORE_CLOSED = "ARTIFACT_STORE_CLOSED"
INVALID_ARTIFACT_RANGE = "INVALID_ARTIFACT_RANGE"

_DIAGNOSTIC_PATTERN = re.compile(
    r"traceback|error|exception|failed|failure|fatal|panic|assertion|warning|"
    r"modulenotfounderror|typeerror|valueerror",
    re.IGNORECASE,
)
_DIAGNOSTIC_LINES_BEFORE = 3
_DIAGNOSTIC_LINES_AFTER = 8
_MAX_DIAGNOSTIC_PREFIX_CHARS = 1_000
_MAX_DIAGNOSTIC_WINDOW_CHARS = 6_000
_MAX_DIAGNOSTIC_RANGES = 5
_COMPACTION_OVERHEAD_RESERVE = 240
_DIAGNOSTIC_HEAD_PERCENT = 20
_DIAGNOSTIC_CONTENT_PERCENT = 50
_NO_DIAGNOSTIC_HEAD_PERCENT = 30
_SUCCESS_STDOUT_PERCENT = 70  # stdout share when both long streams succeeded.
_FAILURE_STDOUT_PERCENT = 40  # stdout share when the command failed or timed out.
_STDOUT_LABEL = "[stdout]\n"
_STDERR_LABEL = "[stderr]\n"
_STREAM_SEPARATOR_RESERVE = 1
_OMISSION_MARKER_PATTERN = re.compile(
    r"(\.\.\.\[output chars )(\d+):(\d+)( omitted\]\.\.\.)"
)


class ArtifactStoreError(ValueError):
    """A model-visible artifact lookup or range failure."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class ArtifactChunk:
    """One character range read from a session-owned output artifact."""

    output_id: str  # Opaque session identifier, never a filesystem path.
    content: str  # Exact text in the requested available character range.
    offset: int  # Inclusive character offset where this chunk starts.
    end: int  # Exclusive character offset where this chunk ends.
    total_chars: int  # Complete artifact length in Unicode characters.
    has_more: bool  # Whether another character exists at end.


@dataclass(frozen=True, slots=True, order=True)
class CharacterRange:
    """A half-open character interval selected from complete output."""

    start: int  # Inclusive position in the original output.
    end: int  # Exclusive position in the original output.

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("character range must satisfy 0 <= start <= end")


@dataclass(frozen=True, slots=True)
class CompactedOutput:
    """A bounded preview and the original ranges represented by it."""

    content: str  # Model-visible complete output or compacted preview.
    original_chars: int  # Character count before compaction.
    truncated: bool  # Whether any original output was omitted.
    included_ranges: tuple[CharacterRange, ...]  # Original text present in content.

    @property
    def returned_chars(self) -> int:
        """Count preview characters, including omission markers."""

        return len(self.content)


class TextCompactionStrategy(Protocol):
    """Select a bounded preview from one text stream."""

    def compact(self, content: str, *, max_chars: int) -> CompactedOutput:
        ...


class OutputCompactionStrategy(Protocol):
    """Select a bounded preview from structured child-process output."""

    def compact(self, result: ProcessResult, *, max_chars: int) -> CompactedOutput:
        ...


class ToolOutputArtifactStore:
    """Keep full tool output in a private temporary directory for one session."""

    def __init__(
        self,
        *,
        max_read_chars: int,
        temporary_parent: str | Path | None = None,
    ) -> None:
        if max_read_chars <= 0:
            raise ValueError("max_read_chars must be greater than zero")
        parent = None if temporary_parent is None else str(temporary_parent)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="minicoder-output-",
            dir=parent,
            ignore_cleanup_errors=True,
        )
        self._root = Path(self._temporary_directory.name)
        self._max_read_chars = max_read_chars
        self._artifacts: dict[str, Path] = {}
        self._closed = False
        _best_effort_chmod(self._root, 0o700)

    @property
    def root(self) -> Path:
        """Return the host-only artifact directory for lifecycle tests and cleanup."""

        return self._root

    @property
    def max_read_chars(self) -> int:
        return self._max_read_chars

    def save(self, content: str) -> str:
        """Persist complete text and return an opaque unguessable session ID."""

        self._require_open()
        if not isinstance(content, str):
            raise TypeError("artifact content must be text")

        output_id = self._new_output_id()
        artifact_path = self._root / f"{secrets.token_hex(16)}.txt"
        artifact_path.write_text(content, encoding="utf-8", newline="")
        _best_effort_chmod(artifact_path, 0o600)
        self._artifacts[output_id] = artifact_path
        return output_id

    def read(self, output_id: str, *, offset: int, limit: int) -> ArtifactChunk:
        """Read a bounded character range without treating output_id as a path."""

        self._require_open()
        artifact_path = self._artifacts.get(output_id)
        if artifact_path is None or not artifact_path.is_file():
            raise ArtifactStoreError(
                ARTIFACT_NOT_FOUND,
                f"Output artifact {output_id!r} is unavailable in this session.",
            )
        if offset < 0 or limit <= 0 or limit > self._max_read_chars:
            raise ArtifactStoreError(
                INVALID_ARTIFACT_RANGE,
                (
                    "offset must be non-negative and limit must be between 1 and "
                    f"{self._max_read_chars} characters"
                ),
            )

        content = artifact_path.read_text(encoding="utf-8")
        total_chars = len(content)
        if offset > total_chars:
            raise ArtifactStoreError(
                INVALID_ARTIFACT_RANGE,
                f"Offset {offset} is beyond artifact length {total_chars}.",
            )
        end = min(total_chars, offset + limit)
        return ArtifactChunk(
            output_id=output_id,
            content=content[offset:end],
            offset=offset,
            end=end,
            total_chars=total_chars,
            has_more=end < total_chars,
        )

    def close(self) -> None:
        """Invalidate IDs and remove the session directory on a best-effort basis."""

        if self._closed:
            return
        self._closed = True
        self._artifacts.clear()
        self._temporary_directory.cleanup()

    def _new_output_id(self) -> str:
        while True:
            candidate = f"out_{secrets.token_urlsafe(18)}"
            if candidate not in self._artifacts:
                return candidate

    def _require_open(self) -> None:
        if self._closed:
            raise ArtifactStoreError(
                ARTIFACT_STORE_CLOSED,
                "Output artifacts are unavailable because the session is closed.",
            )


class DiagnosticOutputCompactor:
    """Preserve one stream's head, diagnostic windows, and tail within a budget."""

    def compact(self, content: str, *, max_chars: int) -> CompactedOutput:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        original_chars = len(content)
        if original_chars <= max_chars:
            ranges = () if not content else (CharacterRange(0, original_chars),)
            return CompactedOutput(
                content=content,
                original_chars=original_chars,
                truncated=False,
                included_ranges=ranges,
            )

        if max_chars <= _COMPACTION_OVERHEAD_RESERVE:
            return _fallback_head_tail(content, max_chars)

        payload_budget = max_chars - _COMPACTION_OVERHEAD_RESERVE
        diagnostics = _diagnostic_ranges(content)
        if diagnostics:
            head_budget = max(
                1,
                payload_budget * _DIAGNOSTIC_HEAD_PERCENT // 100,
            )
            diagnostic_budget = max(
                1,
                payload_budget * _DIAGNOSTIC_CONTENT_PERCENT // 100,
            )
            tail_budget = max(1, payload_budget - head_budget - diagnostic_budget)
            diagnostic_selection = _take_range_prefixes(
                diagnostics,
                diagnostic_budget,
            )
        else:
            head_budget = max(
                1,
                payload_budget * _NO_DIAGNOSTIC_HEAD_PERCENT // 100,
            )
            tail_budget = max(1, payload_budget - head_budget)
            diagnostic_selection = ()

        selected = _merge_ranges(
            (
                CharacterRange(0, min(original_chars, head_budget)),
                *diagnostic_selection,
                CharacterRange(
                    max(0, original_chars - tail_budget),
                    original_chars,
                ),
            )
        )
        preview = _render_ranges(content, selected)
        if len(preview) > max_chars:
            return _fallback_head_tail(content, max_chars)
        return CompactedOutput(
            content=preview,
            original_chars=original_chars,
            truncated=True,
            included_ranges=selected,
        )


class CombinedOutputCompactor:
    """Legacy strategy that compacts stdout and stderr as one combined string."""

    def __init__(
        self,
        text_compactor: TextCompactionStrategy | None = None,
    ) -> None:
        self._text_compactor = (
            DiagnosticOutputCompactor()
            if text_compactor is None
            else text_compactor
        )

    def compact(self, result: ProcessResult, *, max_chars: int) -> CompactedOutput:
        return self._text_compactor.compact(
            result.combined_output(),
            max_chars=max_chars,
        )


class StreamAwareOutputCompactor:
    """Compact stdout and stderr independently before combining labeled previews."""

    def __init__(
        self,
        text_compactor: TextCompactionStrategy | None = None,
    ) -> None:
        self._text_compactor = (
            DiagnosticOutputCompactor()
            if text_compactor is None
            else text_compactor
        )

    def compact(self, result: ProcessResult, *, max_chars: int) -> CompactedOutput:
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")

        complete_output = result.combined_output()
        original_chars = len(complete_output)
        if original_chars <= max_chars:
            ranges = () if not complete_output else (CharacterRange(0, original_chars),)
            return CompactedOutput(
                content=complete_output,
                original_chars=original_chars,
                truncated=False,
                included_ranges=ranges,
            )

        if not result.stdout or not result.stderr:
            return self._text_compactor.compact(
                complete_output,
                max_chars=max_chars,
            )

        minimum_labeled_chars = (
            len(_STDOUT_LABEL)
            + len(_STDERR_LABEL)
            + _STREAM_SEPARATOR_RESERVE
            + 2
        )
        if max_chars < minimum_labeled_chars:
            return self._text_compactor.compact(
                complete_output,
                max_chars=max_chars,
            )

        stdout_offset = len(_STDOUT_LABEL)
        original_separator_chars = 0 if result.stdout.endswith("\n") else 1
        stderr_label_start = (
            stdout_offset + len(result.stdout) + original_separator_chars
        )
        stderr_offset = stderr_label_start + len(_STDERR_LABEL)
        stream_budget = (
            max_chars
            - len(_STDOUT_LABEL)
            - len(_STDERR_LABEL)
            - _STREAM_SEPARATOR_RESERVE
        )
        stdout_percent = (
            _FAILURE_STDOUT_PERCENT
            if result.timed_out or result.exit_code != 0
            else _SUCCESS_STDOUT_PERCENT
        )

        seen_budgets: set[int] = set()
        while stream_budget >= 2 and stream_budget not in seen_budgets:
            seen_budgets.add(stream_budget)
            stdout_budget, stderr_budget = _allocate_stream_budgets(
                total_budget=stream_budget,
                stdout_chars=len(result.stdout),
                stderr_chars=len(result.stderr),
                stdout_percent=stdout_percent,
            )
            stdout_preview = self._text_compactor.compact(
                result.stdout,
                max_chars=stdout_budget,
            )
            stderr_preview = self._text_compactor.compact(
                result.stderr,
                max_chars=stderr_budget,
            )
            rendered_stdout = _shift_omission_markers(
                stdout_preview.content,
                stdout_offset,
            )
            rendered_stderr = _shift_omission_markers(
                stderr_preview.content,
                stderr_offset,
            )
            preview_separator = "" if rendered_stdout.endswith("\n") else "\n"
            preview = (
                f"{_STDOUT_LABEL}{rendered_stdout}{preview_separator}"
                f"{_STDERR_LABEL}{rendered_stderr}"
            )
            if len(preview) <= max_chars:
                included_ranges = _stream_included_ranges(
                    stdout_preview=stdout_preview,
                    stderr_preview=stderr_preview,
                    stdout_offset=stdout_offset,
                    stderr_label_start=stderr_label_start,
                    stderr_offset=stderr_offset,
                    include_original_separator=bool(
                        original_separator_chars and preview_separator
                    ),
                )
                return CompactedOutput(
                    content=preview,
                    original_chars=original_chars,
                    truncated=True,
                    included_ranges=included_ranges,
                )
            stream_budget -= max(1, len(preview) - max_chars)

        return _minimal_labeled_stream_preview(
            result,
            stdout_offset=stdout_offset,
            stderr_label_start=stderr_label_start,
            stderr_offset=stderr_offset,
            include_original_separator=bool(original_separator_chars),
        )


def _allocate_stream_budgets(
    *,
    total_budget: int,
    stdout_chars: int,
    stderr_chars: int,
    stdout_percent: int,
) -> tuple[int, int]:
    if total_budget < 2 or stdout_chars <= 0 or stderr_chars <= 0:
        raise ValueError("two non-empty streams require at least two budget characters")

    stdout_target = max(1, total_budget * stdout_percent // 100)
    stdout_target = min(stdout_target, total_budget - 1)
    stderr_target = total_budget - stdout_target
    stdout_budget = min(stdout_chars, stdout_target)
    stderr_budget = min(stderr_chars, stderr_target)
    remaining = total_budget - stdout_budget - stderr_budget

    stdout_room = stdout_chars - stdout_budget
    stdout_extra = min(stdout_room, remaining)
    stdout_budget += stdout_extra
    remaining -= stdout_extra

    stderr_room = stderr_chars - stderr_budget
    stderr_extra = min(stderr_room, remaining)
    stderr_budget += stderr_extra
    return stdout_budget, stderr_budget


def _shift_omission_markers(content: str, offset: int) -> str:
    def replace(match: re.Match[str]) -> str:
        start = int(match.group(2)) + offset
        end = int(match.group(3)) + offset
        return f"{match.group(1)}{start}:{end}{match.group(4)}"

    return _OMISSION_MARKER_PATTERN.sub(replace, content)


def _stream_included_ranges(
    *,
    stdout_preview: CompactedOutput,
    stderr_preview: CompactedOutput,
    stdout_offset: int,
    stderr_label_start: int,
    stderr_offset: int,
    include_original_separator: bool,
) -> tuple[CharacterRange, ...]:
    ranges: list[CharacterRange] = [CharacterRange(0, len(_STDOUT_LABEL))]
    ranges.extend(
        CharacterRange(
            stdout_offset + current.start,
            stdout_offset + current.end,
        )
        for current in stdout_preview.included_ranges
    )
    if include_original_separator:
        ranges.append(CharacterRange(stderr_label_start - 1, stderr_label_start))
    ranges.append(
        CharacterRange(
            stderr_label_start,
            stderr_label_start + len(_STDERR_LABEL),
        )
    )
    ranges.extend(
        CharacterRange(
            stderr_offset + current.start,
            stderr_offset + current.end,
        )
        for current in stderr_preview.included_ranges
    )
    return _merge_ranges(ranges)


def _minimal_labeled_stream_preview(
    result: ProcessResult,
    *,
    stdout_offset: int,
    stderr_label_start: int,
    stderr_offset: int,
    include_original_separator: bool,
) -> CompactedOutput:
    stdout_content = result.stdout[:1]
    stderr_content = result.stderr[:1]
    preview_separator = "" if stdout_content.endswith("\n") else "\n"
    preview = (
        f"{_STDOUT_LABEL}{stdout_content}{preview_separator}"
        f"{_STDERR_LABEL}{stderr_content}"
    )

    stdout_preview = CompactedOutput(
        stdout_content,
        len(result.stdout),
        True,
        (CharacterRange(0, 1),),
    )
    stderr_preview = CompactedOutput(
        stderr_content,
        len(result.stderr),
        True,
        (CharacterRange(0, 1),),
    )
    included_ranges = _stream_included_ranges(
        stdout_preview=stdout_preview,
        stderr_preview=stderr_preview,
        stdout_offset=stdout_offset,
        stderr_label_start=stderr_label_start,
        stderr_offset=stderr_offset,
        include_original_separator=bool(
            include_original_separator and preview_separator
        ),
    )
    return CompactedOutput(
        content=preview,
        original_chars=len(result.combined_output()),
        truncated=True,
        included_ranges=included_ranges,
    )


def _diagnostic_ranges(content: str) -> tuple[CharacterRange, ...]:
    lines = content.splitlines(keepends=True)
    if not lines:
        return ()

    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    windows: list[CharacterRange] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        match = _DIAGNOSTIC_PATTERN.search(line)
        if not (match or stripped.startswith("E ")):
            continue
        start_line = max(0, index - _DIAGNOSTIC_LINES_BEFORE)
        end_line = min(len(lines), index + _DIAGNOSTIC_LINES_AFTER + 1)
        focus = offsets[index] + (match.start() if match else len(line) - len(stripped))
        window_start = offsets[start_line]
        window_end = offsets[end_line]
        if window_end - window_start > _MAX_DIAGNOSTIC_WINDOW_CHARS:
            window_start = max(
                window_start,
                focus - _MAX_DIAGNOSTIC_PREFIX_CHARS,
            )
            window_end = min(
                window_end,
                window_start + _MAX_DIAGNOSTIC_WINDOW_CHARS,
            )
        windows.append(CharacterRange(window_start, window_end))

    return _merge_ranges(windows)


def _take_range_prefixes(
    ranges: tuple[CharacterRange, ...],
    character_budget: int,
) -> tuple[CharacterRange, ...]:
    selected: list[CharacterRange] = []
    remaining = character_budget
    for current in ranges[:_MAX_DIAGNOSTIC_RANGES]:
        if remaining <= 0:
            break
        length = current.end - current.start
        selected_length = min(length, remaining)
        selected.append(
            CharacterRange(current.start, current.start + selected_length)
        )
        remaining -= selected_length
    return tuple(selected)


def _merge_ranges(
    ranges: list[CharacterRange] | tuple[CharacterRange, ...],
) -> tuple[CharacterRange, ...]:
    non_empty = sorted(current for current in ranges if current.end > current.start)
    if not non_empty:
        return ()

    merged = [non_empty[0]]
    for current in non_empty[1:]:
        previous = merged[-1]
        if current.start <= previous.end:
            merged[-1] = CharacterRange(
                previous.start,
                max(previous.end, current.end),
            )
        else:
            merged.append(current)
    return tuple(merged)


def _render_ranges(content: str, ranges: tuple[CharacterRange, ...]) -> str:
    rendered: list[str] = []
    previous_end = 0
    for current in ranges:
        if current.start > previous_end:
            rendered.append(_omission_marker(previous_end, current.start))
        rendered.append(content[current.start : current.end])
        previous_end = current.end
    if previous_end < len(content):
        rendered.append(_omission_marker(previous_end, len(content)))
    return "".join(rendered)


def _fallback_head_tail(content: str, max_chars: int) -> CompactedOutput:
    original_chars = len(content)
    if max_chars < 32:
        preview = content[:max_chars]
        ranges = () if not preview else (CharacterRange(0, len(preview)),)
        return CompactedOutput(preview, original_chars, True, ranges)

    head_chars = max(1, max_chars * _NO_DIAGNOSTIC_HEAD_PERCENT // 100)
    while True:
        marker = _omission_marker(head_chars, original_chars)
        tail_chars = max_chars - head_chars - len(marker)
        if tail_chars > 0:
            tail_start = original_chars - tail_chars
            marker = _omission_marker(head_chars, tail_start)
            tail_chars = max_chars - head_chars - len(marker)
            if tail_chars > 0:
                break
        head_chars -= 1
        if head_chars <= 0:
            preview = content[:max_chars]
            return CompactedOutput(
                preview,
                original_chars,
                True,
                (CharacterRange(0, len(preview)),),
            )

    tail_start = original_chars - tail_chars
    marker = _omission_marker(head_chars, tail_start)
    preview = content[:head_chars] + marker + content[tail_start:]
    if len(preview) > max_chars:
        overflow = len(preview) - max_chars
        tail_start += overflow
        marker = _omission_marker(head_chars, tail_start)
        preview = content[:head_chars] + marker + content[tail_start:]
    return CompactedOutput(
        preview,
        original_chars,
        True,
        (
            CharacterRange(0, head_chars),
            CharacterRange(tail_start, original_chars),
        ),
    )


def _omission_marker(start: int, end: int) -> str:
    return f"\n...[output chars {start}:{end} omitted]...\n"


def _best_effort_chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, stat.S_IMODE(mode))
    except OSError:
        pass
