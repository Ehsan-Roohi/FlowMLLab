from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from flowmllab.hypersonic_cylinder import (
    TARGET_NAMES,
    case_interpolation_baseline,
    casewise_split_masks,
    ensemble_predict,
    fit_separable_ridge_ensemble,
    load_cylinder_teaching_data,
    relative_l2,
    validate_hypersonic_cylinder_evidence,
    weighted_standardized_mse,
)


ROOT = Path(__file__).resolve().parents[1]


class HypersonicCylinderWeek71Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_cylinder_teaching_data(ROOT)

    def test_dataset_and_provenance_contract(self) -> None:
        report = validate_hypersonic_cylinder_evidence(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["cases"], 20)
        self.assertGreater(report["points"], 40_000)
        self.assertEqual(tuple(report["targets"]), TARGET_NAMES)
        self.assertEqual(report["paper_doi"], "10.1063/5.0334590")

    def test_casewise_split_has_no_case_leakage(self) -> None:
        masks = casewise_split_masks(self.data.mach_inf)
        for left, left_mask in masks.items():
            for right, right_mask in masks.items():
                if left >= right:
                    continue
                self.assertFalse(np.any(left_mask & right_mask))
                left_cases = set(np.unique(self.data.case_id[left_mask]))
                right_cases = set(np.unique(self.data.case_id[right_mask]))
                self.assertTrue(left_cases.isdisjoint(right_cases))

    def test_fast_operator_ensemble_predicts_finite_fields(self) -> None:
        masks = casewise_split_masks(self.data.mach_inf)
        # A small deterministic slice keeps the unit test below a few seconds.
        train_indices = np.flatnonzero(masks["train"])[::25]
        train_mask = np.zeros(len(self.data.mach_inf), dtype=bool)
        train_mask[train_indices] = True
        ensemble = fit_separable_ridge_ensemble(
            self.data, train_mask, members=2, latent_dim=16, seed=17
        )
        test = np.flatnonzero(masks["interpolation"])[::100]
        mean, spread = ensemble_predict(
            ensemble,
            self.data.mach_inf[test],
            self.data.x[test],
            self.data.y[test],
        )
        self.assertEqual(mean.shape, (len(test), 3))
        self.assertEqual(spread.shape, mean.shape)
        self.assertTrue(np.isfinite(mean).all())
        self.assertTrue(np.isfinite(spread).all())
        self.assertTrue(np.all(spread >= 0.0))

    def test_structured_interpolation_is_a_finite_strong_baseline(self) -> None:
        masks = casewise_split_masks(self.data.mach_inf)
        prediction = case_interpolation_baseline(
            self.data, masks["train"], masks["interpolation"]
        )
        truth = self.data.targets[masks["interpolation"]]
        errors = relative_l2(truth, prediction)
        self.assertTrue(np.isfinite(errors).all())
        self.assertTrue(np.all(errors < 0.20))

    def test_metrics_validate_shapes_and_weights(self) -> None:
        truth = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.5, 4.0]])
        estimate = truth + 0.1
        self.assertEqual(relative_l2(truth, estimate).shape, (3,))
        pressure_weighted = weighted_standardized_mse(truth, estimate)
        equal_weighted = weighted_standardized_mse(
            truth, estimate, weights=(1.0, 1.0, 1.0)
        )
        self.assertGreater(pressure_weighted, equal_weighted)


if __name__ == "__main__":
    unittest.main()
