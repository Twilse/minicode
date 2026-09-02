"""Command-line driving adapter for MiniCoder."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, TextIO

from minicoder.adapters.console import (
    ConsoleEventSink,
    format_agent_failure,
    format_minicoder_error,
)
from minicoder.adapters.jsonl_trace import JsonlTraceSink
from minicoder.adapters.terminal_ui import (
    InteractiveLineReader,
    MarkdownTerminalRenderer,
)
from minicoder.application.ports import EventSinkPort
from minicoder.bootstrap import AgentSession, ApplicationFactory, BootstrapContext
from minicoder.domain.errors import MiniCoderError
from minicoder.domain.session import ArchivedDialogueTurn, ArchivedTurnStatus
from minicoder.domain.state import AgentPhase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minicoder",
        description="A locally executing coding agent built from first principles.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="project directory the agent is allowed to access",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate startup configuration without calling a model",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="append sanitized agent events to this JSONL file",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="one coding task; omit it to start an interactive session",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code."""

    input_stream = sys.stdin if stdin is None else stdin
    output = sys.stdout if stdout is None else stdout
    error_output = sys.stderr if stderr is None else stderr
    line_reader = InteractiveLineReader(
        input_stream,
        output,
        use_line_editor=stdin is None and stdout is None,
    )
    renderer = MarkdownTerminalRenderer(output)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        context = ApplicationFactory.create_bootstrap_context(
            environ=environ,
            workspace=args.workspace,
        )
    except MiniCoderError as exc:
        print(format_minicoder_error(exc), file=error_output)
        return 2

    if args.check_config:
        print(_configuration_summary(context), file=output)
        return 0

    console_sink = ConsoleEventSink(
        output,
        defer_recovery_messages=True,
    )
    event_sinks: list[EventSinkPort] = [console_sink]
    if args.trace is not None:
        try:
            event_sinks.append(JsonlTraceSink(args.trace))
        except ValueError:
            print(
                "跟踪文件配置错误：请确认父目录已经存在，并且目标路径是普通文件。",
                file=error_output,
            )
            return 2

    try:
        session = ApplicationFactory.create_agent_session(
            context,
            event_sinks=event_sinks,
        )
        _render_dialogue_history(
            session.dialogue_history,
            renderer=renderer,
            output=output,
        )
        console_sink.flush_recovery_messages()
        if args.task is None:
            return _run_interactive(
                session,
                line_reader=line_reader,
                renderer=renderer,
                output=output,
                error_output=error_output,
            )
        result = session.run(args.task)
    except KeyboardInterrupt:
        print("MiniCoder 已停止：任务被用户中断。", file=error_output)
        return 130
    except MiniCoderError as exc:
        print(format_minicoder_error(exc), file=error_output)
        return 1

    _print_event_failures(session, error_output=error_output)

    if result.phase is AgentPhase.COMPLETE:
        renderer.render(result.final_response or "")
        return 0
    print(format_agent_failure(result), file=error_output)
    return 1


def _run_interactive(
    session: AgentSession,
    *,
    line_reader: InteractiveLineReader,
    renderer: MarkdownTerminalRenderer,
    output: TextIO,
    error_output: TextIO,
) -> int:
    """Read user turns until EOF or an explicit exit command."""

    print("MiniCoder 交互模式。输入 /exit 或 /quit 退出。", file=output)
    reported_failure_count = 0
    last_exit_code = 0
    with session:
        while True:
            line = line_reader.read("minicoder> ")
            if line is None:
                print(file=output)
                return last_exit_code

            user_message = line.strip()
            if not user_message:
                continue
            if user_message.casefold() in {"/exit", "/quit"}:
                return last_exit_code

            result = session.submit(user_message)
            reported_failure_count = _print_event_failures(
                session,
                error_output=error_output,
                start=reported_failure_count,
            )
            if result.phase is AgentPhase.COMPLETE:
                renderer.render(result.final_response or "")
                last_exit_code = 0
            else:
                print(format_agent_failure(result), file=error_output)
                last_exit_code = 1


def _print_event_failures(
    session: AgentSession,
    *,
    error_output: TextIO,
    start: int = 0,
) -> int:
    failures = session.event_failures
    for failure in failures[start:]:
        print(
            "事件输出警告：某个进度记录器未能处理事件；"
            f"类型={failure.sink_type}。",
            file=error_output,
        )
    return len(failures)


def _render_dialogue_history(
    turns: Sequence[ArchivedDialogueTurn],
    *,
    renderer: MarkdownTerminalRenderer,
    output: TextIO,
) -> None:
    """Replay exact external dialogue while hiding host/model protocol records."""

    if not turns:
        return
    for position, turn in enumerate(turns):
        if position > 0:
            print(file=output)
        print("你：", file=output)
        print(turn.task, file=output)
        print("MiniCoder：", file=output)
        if turn.final_response is not None:
            renderer.render(turn.final_response)
        elif turn.status is ArchivedTurnStatus.FAILED:
            print(
                "本轮执行失败，没有生成最终回复。"
                + (
                    f" 原因：{turn.failure_message}"
                    if turn.failure_message is not None
                    else ""
                ),
                file=output,
            )
        else:
            print("本轮在完成前中断，没有生成最终回复。", file=output)
    print(file=output)


def _configuration_summary(context: BootstrapContext) -> str:
    config = context.config
    return "\n".join(
        (
            "MiniCoder configuration is valid.",
            "api_key=<configured>",
            f"base_url={config.base_url}",
            f"model={config.model}",
            f"workspace={config.workspace}",
            f"operating_system={context.operating_system.value}",
            f"verification_commands={len(context.verification_commands)}",
            f"context_budget_chars={config.context_budget_chars}",
            "context_response_reserve_chars="
            f"{config.context_response_reserve_chars}",
            f"planning_enabled={config.planning_enabled}",
            f"memory_enabled={config.memory_enabled}",
            f"session_archive_enabled={config.session_archive_enabled}",
        )
    )
