"""Terminal-specific line editing and Markdown presentation."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Callable
from typing import TextIO

from rich.console import Console
from rich.markdown import Markdown

try:
    import readline as _readline  # noqa: F401 - installs input() editing hooks.
except ImportError:  # pragma: no cover - readline is unavailable on Windows.
    _readline = None


class InteractiveLineReader:
    """Read editable terminal lines while retaining injectable test streams."""

    def __init__(
        self,
        input_stream: TextIO,
        output: TextIO,
        *,
        use_line_editor: bool,
        input_function: Callable[[str], str] | None = None,
    ) -> None:
        self._input_stream = input_stream
        self._output = output
        self._use_line_editor = use_line_editor
        self._input_function = (
            builtins.input if input_function is None else input_function
        )

    def read(self, prompt: str) -> str | None:
        """Return one line without its newline, or None when input reaches EOF."""

        if self._use_line_editor:
            try:
                return self._input_function(prompt)
            except EOFError:
                return None

        print(prompt, end="", file=self._output, flush=True)
        line = self._input_stream.readline()
        if line == "":
            return None
        return line.removesuffix("\n").removesuffix("\r")


class MarkdownTerminalRenderer:
    """Render model Markdown through the capabilities of the target terminal."""

    def __init__(
        self,
        output: TextIO | None = None,
        *,
        force_terminal: bool | None = None,
        width: int | None = None,
    ) -> None:
        self._console = Console(
            file=sys.stdout if output is None else output,
            force_terminal=force_terminal,
            width=width,
        )

    def render(self, content: str) -> None:
        """Render headings, lists, fences, and inline Markdown without raw markers."""

        self._console.print(Markdown(content), soft_wrap=True)
