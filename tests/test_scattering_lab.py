import unittest
import numpy as np
from flowmllab.scattering_lab import (deflection_angle, rutherford_angle, hard_sphere_angle,
    coulomb_repulsive, build_table, transport_cross_sections, fit_surrogate, predict_surrogate,
    collision_weighted_audit, omega22, omega22_star)


class ScatteringLabTests(unittest.TestCase):
    def test_rutherford_matches_analytic_deflection(self):
        k = 0.8
        for energy in (0.7, 2.0, 9.0):
            for b in (0.2, 1.0, 3.5):
                numeric = deflection_angle(b, energy, coulomb_repulsive(k))
                self.assertAlmostEqual(numeric, rutherford_angle(b, energy, k), places=6)

    def test_steep_repulsion_approaches_hard_sphere(self):
        steep = lambda r: (1.0 / np.asarray(r, float)) ** 200
        for b in (0.2, 0.6, 0.95):
            self.assertAlmostEqual(deflection_angle(b, 5.0, steep), float(hard_sphere_angle(b)), delta=0.02)
        self.assertLess(deflection_angle(1.4, 5.0, steep), 1e-3)

    def test_head_on_and_far_limits(self):
        self.assertAlmostEqual(deflection_angle(0.0, 3.0), np.pi)
        self.assertLess(abs(deflection_angle(6.0, 3.0)), 5e-3)

    def test_lennard_jones_glory_angle_is_negative(self):
        # Attractive well: at moderate energy the angle crosses zero and becomes negative.
        chi = [deflection_angle(b, 1.5) for b in (0.9, 1.1, 1.3, 1.6)]
        self.assertTrue(chi[0] > 0.0 and min(chi) < 0.0)

    def test_table_cross_sections_and_surrogate(self):
        table = build_table(np.geomspace(1.0, 20.0, 6), np.linspace(0.0, 3.0, 25))
        q1, q2 = transport_cross_sections(table.impact, table.cos_chi)
        self.assertTrue(np.all(q1 > 0.0) and np.all(q2 > 0.0))
        self.assertTrue(np.all(np.diff(q2) < 0.0))  # softer collisions at higher energy
        fitted = fit_surrogate(table, hidden=(24, 24), seed=0, max_iter=800)
        pred = predict_surrogate(fitted, table.energies, table.impact)
        self.assertLess(np.sqrt(np.mean((pred - table.cos_chi) ** 2)), 0.08)
        audit = collision_weighted_audit(fitted, 2.0, count=200, seed=3)
        self.assertEqual(set(audit) >= {"rms_cos_error", "max_abs_cos_error", "states"}, True)
        self.assertGreater(omega22(table.energies, q2, 2.0), 0.0)


    def test_lennard_jones_collision_integral_matches_hirschfelder_table(self):
        # Independent end-to-end check: deflection integral -> Q2 -> Omega(2,2)* against the
        # classical Lennard-Jones table (Hirschfelder, Curtiss & Bird), 1% tolerance.
        energies = np.geomspace(0.6, 120.0, 40); impact = np.linspace(0.0, 3.0, 61)
        _, q2 = transport_cross_sections(impact, build_table(energies, impact).cos_chi)
        for t_star, reference in ((1.0, 1.587), (2.0, 1.175), (4.0, 0.9700), (10.0, 0.8242)):
            self.assertAlmostEqual(omega22_star(energies, q2, t_star) / reference, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
