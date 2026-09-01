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
        self.assertEqual(flowmllab.__version__, "1.1.0")

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
        self.assertTrue(report["root"].endswith("flowmllab/assets"))

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


if __name__ == "__main__":
    unittest.main()
