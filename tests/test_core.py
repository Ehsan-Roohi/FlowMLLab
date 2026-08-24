from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import unittest

import flowmllab
from flowmllab.cli import main


ROOT = Path(__file__).resolve().parents[1]


class FlowMLLabCoreTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(flowmllab.__version__, "1.0.2")

    def test_core_asset_contract(self) -> None:
        report = flowmllab.validate_core_assets(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["cases"], 11)
        self.assertEqual(report["grid"], [65, 65])
        self.assertEqual(report["reynolds_numbers"][0], 100.0)
        self.assertEqual(report["reynolds_numbers"][-1], 400.0)

    def test_cli_smoke(self) -> None:
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main(["smoke", "--root", str(ROOT)])
        self.assertEqual(status, 0)
        self.assertIn('"status": "pass"', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
