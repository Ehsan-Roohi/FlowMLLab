"""Command-line interface for FlowMLLab."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .core import (
    ValidationError,
    discover_repository_root,
    format_report,
    generate_cavity_rom_validation,
    generate_validation_figures,
    run_repository_qa,
    validate_core_assets,
    verify_cylinder_lbm_validation,
    verify_gas_dynamics_week8,
    verify_mahdavi_deeponet_week9,
)
from .probabilistic_uq import validate_probabilistic_uq_evidence


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

    rom = subparsers.add_parser(
        "rom", help="regenerate the validated Week-4.1 cavity ROM evidence"
    )
    rom.add_argument("--root", type=Path, help="FlowMLLab checkout root")

    cylinder = subparsers.add_parser(
        "cylinder",
        help="verify retained Week-7 LBM grid study and learned-model evidence",
    )
    cylinder.add_argument("--root", type=Path, help="FlowMLLab checkout root")

    gasdynamics = subparsers.add_parser(
        "gasdynamics",
        help="verify retained Week-8 exact and learned gas-dynamics evidence",
    )
    gasdynamics.add_argument("--root", type=Path, help="FlowMLLab checkout root")
    mahdavi = subparsers.add_parser(
        "mahdavi",
        help="verify Week-9 micro-step and micro-nozzle DSMC evidence",
    )
    mahdavi.add_argument("--root", type=Path, help="FlowMLLab checkout root")
    mahdavi.add_argument(
        "--step-source",
        type=Path,
        help="checkout/cache of Ehsan-Roohi/roohi-step-dnn-mahdavi",
    )
    mahdavi.add_argument(
        "--require-step-data",
        action="store_true",
        help="require a hash-checked upstream checkout in addition to included archives",
    )
    uq = subparsers.add_parser(
        "uq", help="verify retained probabilistic-UQ cavity evidence"
    )
    uq.add_argument("--root", type=Path, help="FlowMLLab checkout root")
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
        if args.command == "rom":
            return generate_cavity_rom_validation(args.root)
        if args.command == "cylinder":
            return verify_cylinder_lbm_validation(args.root)
        if args.command == "gasdynamics":
            return verify_gas_dynamics_week8(args.root)
        if args.command == "mahdavi":
            return verify_mahdavi_deeponet_week9(
                args.root,
                args.step_source,
                args.require_step_data,
            )
        if args.command == "uq":
            root = discover_repository_root(args.root)
            try:
                report = validate_probabilistic_uq_evidence(root)
            except (OSError, ValueError, KeyError) as error:
                raise ValidationError(
                    f"Probabilistic-UQ evidence failed: {error}"
                ) from error
            print(format_report(report))
            return 0
    except ValidationError as error:
        print(f"FlowMLLab validation failed: {error}")
        return 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
