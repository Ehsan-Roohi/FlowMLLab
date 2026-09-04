from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest
from unittest import mock

import flowmllab
from flowmllab.cli import build_parser, main
from flowmllab.core import EXPECTED_DATA_SHA256, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class FlowMLLabCoreTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(flowmllab.__version__, "1.2.0")

    def test_package_metadata_supports_colab_python_313(self) -> None:
        metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('requires-python = ">=3.10,<3.14"', metadata)
        self.assertIn('"Programming Language :: Python :: 3.13"', metadata)

    def test_core_asset_contract(self) -> None:
        report = flowmllab.validate_core_assets(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["cases"], 11)
        self.assertEqual(report["grid"], [65, 65])
        self.assertEqual(report["reynolds_numbers"][0], 100.0)
        self.assertEqual(report["reynolds_numbers"][-1], 400.0)

    def test_packaged_asset_fallback_matches_release_hash(self) -> None:
        with mock.patch(
            "flowmllab.core.discover_repository_root",
            side_effect=ValidationError("no checkout"),
        ):
            report = flowmllab.validate_core_assets()
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["dataset_sha256"], EXPECTED_DATA_SHA256)
        asset_root = Path(report["root"])
        self.assertEqual(asset_root.name, "assets")
        self.assertEqual(asset_root.parent.name, "flowmllab")

    def test_cli_smoke(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main(["smoke", "--root", str(ROOT)])
        self.assertEqual(status, 0)
        self.assertIn('"status": "pass"', captured.getvalue())

    def test_cavity_rom_cli_is_exposed(self) -> None:
        args = build_parser().parse_args(["rom", "--root", str(ROOT)])
        self.assertEqual(args.command, "rom")
        self.assertEqual(args.root, ROOT)

    def test_cylinder_cli_is_exposed(self) -> None:
        args = build_parser().parse_args(["cylinder", "--root", str(ROOT)])
        self.assertEqual(args.command, "cylinder")
        self.assertEqual(args.root, ROOT)

    def test_gas_dynamics_cli_is_exposed(self) -> None:
        args = build_parser().parse_args(["gasdynamics", "--root", str(ROOT)])
        self.assertEqual(args.command, "gasdynamics")
        self.assertEqual(args.root, ROOT)

    def test_mahdavi_cli_is_exposed(self) -> None:
        args = build_parser().parse_args([
            "mahdavi", "--root", str(ROOT), "--step-source", str(ROOT),
            "--require-step-data",
        ])
        self.assertEqual(args.command, "mahdavi")
        self.assertEqual(args.root, ROOT)
        self.assertEqual(args.step_source, ROOT)
        self.assertTrue(args.require_step_data)

    def test_probabilistic_uq_cli_is_exposed(self) -> None:
        args = build_parser().parse_args(["uq", "--root", str(ROOT)])
        self.assertEqual(args.command, "uq")
        self.assertEqual(args.root, ROOT)


if __name__ == "__main__":
    unittest.main()
