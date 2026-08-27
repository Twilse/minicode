"""Command-line driving adapter for MiniCoder."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping, TextIO

from minicoder.bootstrap import ApplicationFactory, BootstrapContext
from minicoder.domain.errors import MiniCoderError


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

    if not args.check_config:
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

    print(_configuration_summary(context), file=output)
    return 0


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
