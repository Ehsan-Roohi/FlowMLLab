from __future__ import annotations

import unittest

import numpy as np

from flowmllab import cylinder_lbm


class CylinderLBMTests(unittest.TestCase):
    def test_equilibrium_recovers_prescribed_macroscopic_state(self) -> None:
        y, x = np.mgrid[:5, :7]
        rho = 1.0 + 1.0e-3 * np.sin(x)
        u = 0.03 + 2.0e-3 * np.cos(y)
        v = 1.0e-3 * np.sin(x + y)
        recovered = cylinder_lbm.macroscopic(cylinder_lbm.equilibrium(rho, u, v))
        self.assertTrue(np.allclose(recovered[0], rho, atol=1.0e-14))
        self.assertTrue(np.allclose(recovered[1], u, atol=1.0e-14))
        self.assertTrue(np.allclose(recovered[2], v, atol=1.0e-14))

    def test_synthetic_lift_gives_correct_strouhal(self) -> None:
        time = np.arange(0.0, 4000.0, 4.0)
        expected = 0.16
        diameter = 10.0
        speed = 0.05
        frequency = expected * speed / diameter
        lift = 0.7 * np.sin(2.0 * np.pi * frequency * time + 0.3)
        estimated = cylinder_lbm.estimate_strouhal(
            time, lift, diameter, speed, transient_fraction=0.0
        )
        self.assertAlmostEqual(estimated, expected, delta=0.006)

    def test_recirculation_length_uses_first_centerline_zero(self) -> None:
        u = np.ones((21, 60))
        solid = cylinder_lbm.cylinder_mask(60, 21, 8.0, (15.0, 10.0))
        u[10, 20:30] = np.linspace(-0.2, -0.01, 10)
        u[10, 30] = 0.01
        length, normalized = cylinder_lbm.recirculation_length(
            u, solid, (15.0, 10.0), 8.0
        )
        self.assertAlmostEqual(length, 10.5)
        self.assertAlmostEqual(normalized, 10.5 / 8.0)

    def test_smoke_run_is_deterministic_finite_and_no_slip(self) -> None:
        kwargs = dict(
            reynolds=40,
            nx=64,
            ny=32,
            diameter=8.0,
            center=(18.0, 15.5),
            inflow_velocity=0.04,
            steps=40,
            history_stride=5,
            snapshot_stride=20,
            seed=17,
        )
        first = cylinder_lbm.simulate_cylinder(**kwargs)
        second = cylinder_lbm.simulate_cylinder(**kwargs)
        for key in ("rho", "u", "v", "p", "vorticity", "drag_coefficient"):
            self.assertTrue(np.array_equal(first[key], second[key]))
            self.assertTrue(np.isfinite(first[key]).all())
        solid = first["solid"]
        self.assertTrue(np.all(first["u"][solid] == 0.0))
        self.assertTrue(np.all(first["v"][solid] == 0.0))
        self.assertEqual(first["snapshots"]["u"].shape, (2, 32, 64))
        self.assertEqual(first["metadata"]["transverse_boundary"], "periodic")
        self.assertIn("analytical circle", first["metadata"]["cylinder_boundary"])
        self.assertLess(np.max(np.abs(first["mean_density_ratio"] - 1.0)), 0.02)

    def test_bouzidi_links_follow_the_analytical_circle(self) -> None:
        center = (18.0, 15.5)
        diameter = 8.0
        solid = cylinder_lbm.cylinder_mask(64, 32, diameter, center)
        fractions = cylinder_lbm.curved_link_fractions(solid, center, diameter)
        finite = np.concatenate(
            [values[np.isfinite(values)] for values in fractions[1:]]
        )
        self.assertGreater(finite.size, 20)
        self.assertTrue(np.all((finite > 0.0) & (finite <= 1.0)))
        self.assertTrue(np.any(np.abs(finite - 0.5) > 0.1))

        result = cylinder_lbm.simulate_cylinder(
            40,
            nx=64,
            ny=32,
            diameter=diameter,
            center=center,
            inflow_velocity=0.04,
            steps=80,
            history_stride=5,
            seed=17,
            cylinder_boundary="bouzidi",
        )
        self.assertTrue(np.isfinite(result["rho"]).all())
        self.assertIn("analytical circle", result["metadata"]["cylinder_boundary"])

    def test_low_mach_and_relaxation_stability_gates(self) -> None:
        with self.assertRaisesRegex(ValueError, "low-Mach"):
            cylinder_lbm.simulate_cylinder(
                20, nx=64, ny=32, diameter=8, inflow_velocity=0.13, steps=1
            )
        with self.assertRaisesRegex(ValueError, "too close"):
            cylinder_lbm.simulate_cylinder(
                1000, nx=64, ny=32, diameter=8, inflow_velocity=0.02, steps=1
            )

    def test_presets_separate_quick_from_validation_settings(self) -> None:
        quick = cylinder_lbm.recommended_parameters(100, "quick")
        validated = cylinder_lbm.recommended_parameters(100, "validation")
        self.assertLess(quick["diameter"], validated["diameter"])
        self.assertGreater(validated["steps"], quick["steps"])
        tau = 0.5 + 3.0 * validated["inflow_velocity"] * validated["diameter"] / 100
        self.assertGreaterEqual(tau, 0.53)
        self.assertLessEqual(validated["diameter"] / validated["ny"], 0.05)
        self.assertGreaterEqual(validated["center"][0] / validated["diameter"], 8.0)
        self.assertGreaterEqual(
            (validated["nx"] - validated["center"][0]) / validated["diameter"], 22.0
        )


if __name__ == "__main__":
    unittest.main()
