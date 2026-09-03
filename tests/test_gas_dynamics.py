from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from flowmllab.gas_dynamics import (
    fanno_inverse_friction_length,
    fanno_ratios,
    nozzle_back_pressure,
    nozzle_shock_area,
    oblique_beta,
    oblique_detachment,
    oblique_theta,
    rayleigh_inverse_t0,
    rayleigh_ratios,
    shock_tube_pressure_ratio,
    shock_tube_residual_general,
    validate_week8_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gas_dynamics_week8"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    result.update(path.read_bytes())
    return result.hexdigest()


class GasDynamicsWeek8Tests(unittest.TestCase):
    def test_rayleigh_inverse_closes_on_both_branches(self) -> None:
        target = 0.82
        subsonic = rayleigh_inverse_t0(target, "subsonic")
        supersonic = rayleigh_inverse_t0(target, "supersonic")
        self.assertLess(subsonic, 1.0)
        self.assertGreater(supersonic, 1.0)
        self.assertAlmostEqual(float(rayleigh_ratios(subsonic)[4]), target, places=10)
        self.assertAlmostEqual(float(rayleigh_ratios(supersonic)[4]), target, places=10)

    def test_fanno_inverse_closes_on_both_branches(self) -> None:
        target = 0.18
        subsonic = fanno_inverse_friction_length(target, "subsonic")
        supersonic = fanno_inverse_friction_length(target, "supersonic")
        self.assertLess(subsonic, 1.0)
        self.assertGreater(supersonic, 1.0)
        self.assertAlmostEqual(float(fanno_ratios(subsonic)[4]), target, places=10)
        self.assertAlmostEqual(float(fanno_ratios(supersonic)[4]), target, places=10)

    def test_oblique_shock_keeps_weak_and_strong_roots(self) -> None:
        mach = 2.0
        theta = np.radians(12.0)
        peak, maximum = oblique_detachment(mach)
        weak = oblique_beta(mach, theta, "weak")
        strong = oblique_beta(mach, theta, "strong")
        self.assertLess(theta, maximum)
        self.assertLess(weak, peak)
        self.assertGreater(strong, peak)
        self.assertAlmostEqual(float(oblique_theta(mach, weak)), theta, places=10)
        self.assertAlmostEqual(float(oblique_theta(mach, strong)), theta, places=10)

    def test_nozzle_inverse_is_bounded_and_closes(self) -> None:
        exit_area = 2.5
        expected_shock_area = 1.8
        pressure = nozzle_back_pressure(exit_area, expected_shock_area)
        recovered = nozzle_shock_area(exit_area, pressure)
        self.assertGreater(recovered, 1.0)
        self.assertLess(recovered, exit_area)
        self.assertAlmostEqual(recovered, expected_shock_area, places=9)

    def test_shock_tube_root_satisfies_compatibility(self) -> None:
        driver = np.array([1.1, 2.0, 10.0, 50.0])
        star = np.array([shock_tube_pressure_ratio(value) for value in driver])
        residual = shock_tube_residual_general(star, driver, np.ones_like(driver))
        self.assertLess(float(np.max(np.abs(residual))), 1.0e-9)
        self.assertTrue(np.all(star > 1.0))

    def test_retained_evidence_and_source_hashes(self) -> None:
        report = validate_week8_evidence(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["problems"], 5)
        self.assertLess(report["max_primary_relative_l2"], 3.5e-3)
        self.assertEqual(report["shock_tube_queries"], 100_000)
        provenance = json.loads((RESULTS / "provenance.json").read_text())
        for filename, expected in provenance["copied_files"].items():
            self.assertEqual(digest(RESULTS / filename), expected)


if __name__ == "__main__":
    unittest.main()
