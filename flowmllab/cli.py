"""Command-line interface for FlowMLLab."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .core import (
    ValidationError,
    format_report,
    generate_validation_figures,
    run_repository_qa,
    validate_core_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowmllab",
        description="Validate and reproduce FlowMLLab scientific-ML experiments.",
    )
    parser.add_argument("--version", action="version", version=f"FlowMLLab {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="validate the fixed core dataset and boundary contract")
    smoke.add_argument("--root", type=Path, help="FlowMLLab checkout root")

    qa = subparsers.add_parser("qa", help="run the complete repository release gate")
    qa.add_argument("--root", type=Path, help="FlowMLLab checkout root")

    figures = subparsers.add_parser("figures", help="generate manuscript validation figures")
    figures.add_argument("target", choices=("ghia", "pressure", "dsmc", "all"), nargs="?", default="all")
    figures.add_argument("--root", type=Path, help="FlowMLLab checkout root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "smoke":
            print(format_report(validate_core_assets(args.root)))
            return 0
        if args.command == "qa":
            return run_repository_qa(args.root)
        if args.command == "figures":
            return generate_validation_figures(args.target, args.root)
    except ValidationError as error:
        print(f"FlowMLLab validation failed: {error}")
        return 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

