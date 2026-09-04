from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from flowmllab.mahdavi_deeponet import load_step_height_archive
from flowmllab.step_geom_deeponet import (
    GEOM_DEEPONET_DOI,
    StepDomain,
    build_step_geom_deeponet,
    fit_step_velocity_scale,
    infer_step_domain,
    sample_step_geom_training_batch,
    step_geom_deeponet_inputs,
    step_signed_distance,
)

ROOT = Path(__file__).resolve().parents[1]


class StepGeomDeepONetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.learning = load_step_height_archive(ROOT, split="learning")
        first = cls.learning[16]
        cls.domain = infer_step_domain(first["x"], first["y"])

    def test_reference_and_inferred_physical_domain(self) -> None:
        self.assertEqual(GEOM_DEEPONET_DOI, "10.1016/j.cma.2024.117130")
        self.assertAlmostEqual(self.domain.x_min_m, 0.0, delta=2.0e-12)
        self.assertAlmostEqual(self.domain.y_min_m, 0.0, delta=2.0e-12)
        self.assertAlmostEqual(self.domain.length_over_height, 5.0, delta=0.02)
        self.assertAlmostEqual(self.domain.step_x_m, 25.0e-9)

    def test_signed_distance_has_exact_step_sign_convention(self) -> None:
        domain = StepDomain(0.0, 5.0, 0.0, 1.0, step_x_m=1.0)
        x = np.array([0.5, 2.0, 0.5, 0.5, 1.0, 2.0])
        y = np.array([0.7, 0.1, 0.2, 0.4, 0.2, 0.0])
        sdf = step_signed_distance(0.4, x, y, domain=domain)
        self.assertGreater(sdf[0], 0.0)
        self.assertAlmostEqual(sdf[1], 0.1)
        self.assertAlmostEqual(sdf[2], -0.2)
        np.testing.assert_allclose(sdf[3:], 0.0, atol=1.0e-14)

    def test_geom_inputs_are_only_parameter_coordinates_and_sdf(self) -> None:
        x = np.array([5.0e-9, 35.0e-9, 70.0e-9])
        y = np.array([12.0e-9, 4.0e-9, 8.0e-9])
        branch, trunk = step_geom_deeponet_inputs(
            0.44, x, y, domain=self.domain
        )
        self.assertEqual(branch.shape, (1, 1))
        self.assertEqual(trunk.shape, (3, 3))
        self.assertEqual(float(branch[0, 0]), np.float32(0.44))
        np.testing.assert_allclose(
            trunk[:, 2],
            step_signed_distance(0.44, x, y, domain=self.domain),
            rtol=2.0e-7,
        )

    def test_target_changes_cannot_change_inputs_or_sample_locations(self) -> None:
        cases = {height: dict(case) for height, case in self.learning.items()}
        scale = fit_step_velocity_scale(cases, [16, 21])
        original = sample_step_geom_training_batch(
            cases,
            [16, 21],
            domain=self.domain,
            velocity_scale=scale,
            points_per_case=128,
            seed=17,
        )
        corrupted = {height: dict(case) for height, case in cases.items()}
        for height in (16, 21):
            corrupted[height]["u"] = -3.0 * np.asarray(corrupted[height]["u"])
            corrupted[height]["v"] = 7.0 + np.asarray(corrupted[height]["v"])
        changed = sample_step_geom_training_batch(
            corrupted,
            [16, 21],
            domain=self.domain,
            velocity_scale=scale,
            points_per_case=128,
            seed=17,
        )
        np.testing.assert_array_equal(original.point_indices, changed.point_indices)
        np.testing.assert_array_equal(original.parameters, changed.parameters)
        np.testing.assert_array_equal(original.trunk, changed.trunk)
        self.assertFalse(np.array_equal(original.targets, changed.targets))

    def test_training_store_cannot_supply_a_sealed_geometry(self) -> None:
        scale = fit_step_velocity_scale(self.learning, [16])
        with self.assertRaises(KeyError):
            sample_step_geom_training_batch(
                self.learning,
                [44],
                domain=self.domain,
                velocity_scale=scale,
                points_per_case=16,
            )

    def test_height_changes_sdf_at_fixed_query_point(self) -> None:
        x = np.array([10.0e-9])
        y = np.array([10.0e-9])
        low = step_signed_distance(0.25, x, y, domain=self.domain)
        high = step_signed_distance(0.67, x, y, domain=self.domain)
        self.assertGreater(float(low[0]), 0.0)
        self.assertLess(float(high[0]), 0.0)

    def test_model_accepts_arbitrary_point_counts_when_ml_extra_exists(self) -> None:
        try:
            model = build_step_geom_deeponet(width=8, seed=4)
        except ImportError:
            self.skipTest("optional TensorFlow dependency is not installed")
        branch = np.array([[0.44]], dtype=np.float32)
        for count in (7, 19):
            trunk = np.zeros((1, count, 3), dtype=np.float32)
            prediction = np.asarray(model((branch, trunk), training=False))
            self.assertEqual(prediction.shape, (1, count, 2))
            self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
