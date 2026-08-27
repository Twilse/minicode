"""Allow MiniCoder to run with ``python -m minicoder``."""

from minicoder.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
