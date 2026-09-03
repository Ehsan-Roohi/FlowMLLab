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
        time = np.arange(0.0, 14000.0, 4.0)
        expected = 0.16
        diameter = 10.0
        speed = 0.05
        frequency = expected * speed / diameter
        lift = 0.7 * np.sin(2.0 * np.pi * frequency * time + 0.3)
        estimated = cylinder_lbm.estimate_strouhal(
            time, lift, diameter, speed, transient_fraction=0.0
        )
        self.assertAlmostEqual(estimated, expected, delta=0.006)

    def test_strouhal_gate_rejects_acoustic_mode_and_short_record(self) -> None:
        time = np.arange(0.0, 5000.0, 4.0)
        acoustic = np.sin(2.0 * np.pi * 1.42 * 0.05 / 10.0 * time)
        diagnostic = cylinder_lbm.strouhal_diagnostics(
            time, acoustic, 10.0, 0.05, transient_fraction=0.0
        )
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(
            diagnostic["reason"], "dominant_energy_outside_physical_band"
        )
        short_physical = np.sin(2.0 * np.pi * 0.16 * 0.05 / 10.0 * time)
        diagnostic = cylinder_lbm.strouhal_diagnostics(
            time, short_physical, 10.0, 0.05, transient_fraction=0.0
        )
        self.assertFalse(diagnostic["valid"])
        self.assertEqual(diagnostic["reason"], "fewer_than_minimum_cycles")

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

    def test_restart_matches_an_uninterrupted_run(self) -> None:
        common = dict(
            reynolds=40,
            nx=64,
            ny=32,
            diameter=8.0,
            center=(18.0, 15.5),
            inflow_velocity=0.04,
            history_stride=10,
            perturbation=0.0,
            seed=17,
        )
        full = cylinder_lbm.simulate_cylinder(steps=60, **common)
        first = cylinder_lbm.simulate_cylinder(
            steps=30, return_restart_state=True, **common
        )
        continued = cylinder_lbm.simulate_cylinder(
            steps=30,
            restart_state=first["restart_state"],
            **common,
        )
        np.testing.assert_allclose(continued["rho"], full["rho"], rtol=0, atol=0)
        np.testing.assert_allclose(continued["u"], full["u"], rtol=0, atol=0)
        self.assertEqual(continued["time"][-1], 60.0)

    def test_smooth_startup_preserves_mass_with_convective_outlet(self) -> None:
        result = cylinder_lbm.simulate_cylinder(
            40,
            nx=64,
            ny=32,
            diameter=8.0,
            center=(18.0, 15.5),
            inflow_velocity=0.04,
            steps=160,
            history_stride=8,
            startup_ramp_steps=40,
            perturbation=0.0,
        )
        self.assertLess(
            np.max(np.abs(result["mean_density_ratio"] - 1.0)), 0.01
        )

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
        self.assertGreater(quick["startup_ramp_steps"], 0)
        self.assertGreater(validated["startup_ramp_steps"], 0)
        self.assertLessEqual(quick["diameter"] / quick["ny"], 0.15)
        tau = 0.5 + 3.0 * validated["inflow_velocity"] * validated["diameter"] / 100
        self.assertGreaterEqual(tau, 0.53)
        self.assertLessEqual(validated["diameter"] / validated["ny"], 0.05)
        self.assertGreaterEqual(validated["center"][0] / validated["diameter"], 8.0)
        self.assertGreaterEqual(
            (validated["nx"] - validated["center"][0]) / validated["diameter"], 22.0
        )

    def test_taylor_green_decay_recovers_viscosity_for_bgk_and_trt(self) -> None:
        n = 32
        wave_number = 2.0 * np.pi / n
        tau = 0.8
        viscosity = cylinder_lbm.CS2 * (tau - 0.5)
        steps = 160
        y, x = np.mgrid[:n, :n]
        basis_u = np.sin(wave_number * x) * np.cos(wave_number * y)
        basis_v = -np.cos(wave_number * x) * np.sin(wave_number * y)
        amplitude = 1.0e-4

        for model in ("bgk", "trt"):
            f = cylinder_lbm.equilibrium(
                np.ones((n, n)), amplitude * basis_u, amplitude * basis_v
            )
            for _ in range(steps):
                rho, u, v = cylinder_lbm.macroscopic(f)
                post = cylinder_lbm._collide(
                    f,
                    cylinder_lbm.equilibrium(rho, u, v),
                    tau,
                    model,
                    3.0 / 16.0,
                )
                f = np.empty_like(post)
                for q, (cx, cy) in enumerate(cylinder_lbm.LATTICE_VELOCITIES):
                    f[..., q] = np.roll(
                        np.roll(post[..., q], int(cy), axis=0), int(cx), axis=1
                    )
            _, u, v = cylinder_lbm.macroscopic(f)
            recovered = (
                np.sum(u * basis_u + v * basis_v)
                / np.sum(basis_u**2 + basis_v**2)
            )
            expected = amplitude * np.exp(-2.0 * viscosity * wave_number**2 * steps)
            self.assertAlmostEqual(recovered / expected, 1.0, delta=0.035)


if __name__ == "__main__":
    unittest.main()
