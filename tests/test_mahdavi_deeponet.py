from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from flowmllab.mahdavi_deeponet import (
    NOZZLE_HELD_OUT_KPA,
    density_snapshot_matrix,
    load_nozzle_centerlines,
    manufactured_step_velocity,
    pod_spectrum,
    validate_week9_evidence,
    zonal_velocity_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


class MahdaviDeepONetWeek9Tests(unittest.TestCase):
    def test_manufactured_step_has_solid_and_recirculation_regions(self) -> None:
        x = np.linspace(0.0, 5.0, 80)
        y = np.linspace(0.0, 1.0, 36)
        xx, yy = np.meshgrid(x, y)
        u, v, solid = manufactured_step_velocity(xx, yy, 0.44)
        self.assertTrue(np.any(solid))
        self.assertTrue(np.any((u < 0.0) & ~solid))
        self.assertTrue(np.all(u[solid] == 0.0))
        self.assertTrue(np.all(v[solid] == 0.0))

    def test_zonal_metric_exposes_local_error_hidden_by_global_average(self) -> None:
        x = np.linspace(0.0, 5.0, 100)
        y = np.linspace(0.0, 1.0, 40)
        xx, yy = np.meshgrid(x, y)
        u, v, solid = manufactured_step_velocity(xx, yy, 0.55)
        prediction_u = u.copy()
        prediction_u[u < 0.0] = 0.0
        metrics = zonal_velocity_metrics(
            u, v, prediction_u, v, alpha=0.7, valid_mask=~solid
        )
        self.assertGreater(metrics["vortex_relative_l2"], metrics["full_relative_l2"])
        self.assertGreater(metrics["zonal_loss"], metrics["full_mse"])

    def test_public_nozzle_derivative_reproduces_density_pod_audit(self) -> None:
        data = load_nozzle_centerlines(ROOT)
        self.assertEqual(data["density"].shape, (15, 101))
        self.assertTrue(np.isin(NOZZLE_HELD_OUT_KPA, data["pressure_kpa"]).all())
        _, physical = density_snapshot_matrix(
            data["x_m"], data["density"], data["shock_x_m"], data["delta_jump_m"],
            coordinate="physical",
        )
        _, aligned = density_snapshot_matrix(
            data["x_m"], data["density"], data["shock_x_m"], data["delta_jump_m"],
            coordinate="shock_centered",
        )
        physical_pod = pod_spectrum(physical)
        aligned_pod = pod_spectrum(aligned)
        self.assertEqual(physical_pod["n99"], 8)
        self.assertEqual(aligned_pod["n99"], 2)
        self.assertLess(abs(physical_pod["first_mode_percent"] - 78.9221), 0.03)
        self.assertLess(abs(aligned_pod["first_mode_percent"] - 97.9855), 0.03)

    def test_week9_evidence_contract(self) -> None:
        report = validate_week9_evidence(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["nozzle_cases"], 15)
        self.assertEqual(report["held_out_pressures_kpa"], [16, 25, 30])


if __name__ == "__main__":
    unittest.main()
