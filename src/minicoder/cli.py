"""Command-line driving adapter for MiniCoder."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, TextIO

from minicoder.adapters.console import ConsoleEventSink
from minicoder.adapters.jsonl_trace import JsonlTraceSink
from minicoder.application.ports import EventSinkPort
from minicoder.bootstrap import ApplicationFactory, BootstrapContext
from minicoder.domain.errors import MiniCoderError
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
        help="one coding task for the agent to complete",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the CLI and return a process exit code."""

    output = sys.stdout if stdout is None else stdout
    error_output = sys.stderr if stderr is None else stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.check_config and args.task is None:
        parser.print_help(file=output)
        return 0

    try:
        context = ApplicationFactory.create_bootstrap_context(
            environ=environ,
            workspace=args.workspace,
        )
    except MiniCoderError as exc:
        print(f"configuration error: {exc}", file=error_output)
        return 2

    if args.check_config:
        print(_configuration_summary(context), file=output)
        return 0

    event_sinks: list[EventSinkPort] = [ConsoleEventSink(output)]
    if args.trace is not None:
        try:
            event_sinks.append(JsonlTraceSink(args.trace))
        except ValueError as exc:
            print(f"trace error: {exc}", file=error_output)
            return 2

    try:
        assert args.task is not None
        session = ApplicationFactory.create_agent_session(
            context,
            event_sinks=event_sinks,
        )
        result = session.run(args.task)
    except KeyboardInterrupt:
        print("agent interrupted by user", file=error_output)
        return 130
    except MiniCoderError as exc:
        print(f"agent error: {exc}", file=error_output)
        return 1

    for failure in session.event_failures:
        print(
            f"event sink warning: {failure.sink_type}: {failure.message}",
            file=error_output,
        )

    if result.phase is AgentPhase.COMPLETE:
        print(result.final_response, file=output)
        return 0
    print(f"agent failed: {result.failure_message}", file=error_output)
    return 1


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
        )
    )
