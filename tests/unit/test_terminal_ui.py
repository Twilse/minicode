from __future__ import annotations

from io import StringIO

from minicoder.adapters.terminal_ui import (
    InteractiveLineReader,
    MarkdownTerminalRenderer,
)


def test_interactive_reader_uses_the_editable_input_path() -> None:
    prompts: list[str] = []

    def editable_input(prompt: str) -> str:
        prompts.append(prompt)
        return "如果我想修改代码"

    reader = InteractiveLineReader(
        StringIO("fallback must not be read"),
        StringIO(),
        use_line_editor=True,
        input_function=editable_input,
    )

    assert reader.read("minicoder> ") == "如果我想修改代码"
    assert prompts == ["minicoder> "]


def test_interactive_reader_maps_editable_input_eof_to_none() -> None:
    def end_of_input(_: str) -> str:
        raise EOFError

    reader = InteractiveLineReader(
        StringIO(),
        StringIO(),
        use_line_editor=True,
        input_function=end_of_input,
    )

    assert reader.read("minicoder> ") is None


def test_injected_stream_reader_remains_deterministic_for_tests_and_pipes() -> None:
    output = StringIO()
    reader = InteractiveLineReader(
        StringIO("第一条中文消息\n"),
        output,
        use_line_editor=False,
    )

    assert reader.read("minicoder> ") == "第一条中文消息"
    assert output.getvalue() == "minicoder> "


def test_markdown_renderer_removes_fences_and_keeps_code_content() -> None:
    output = StringIO()
    renderer = MarkdownTerminalRenderer(
        output,
        force_terminal=False,
        width=80,
    )

    renderer.render(
        "输入格式：\n\n```text\nn m\nu v w   # 共 m 行\nsource\n```"
    )

    rendered = output.getvalue()
    assert "```" not in rendered
    assert "n m" in rendered
    assert "u v w   # 共 m 行" in rendered
    assert "source" in rendered
