from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from flowmllab.mahdavi_deeponet import (
    NOZZLE_HELD_OUT_KPA,
    STEP_HEIGHT_ARCHIVE_SHA256,
    STEP_HEIGHT_FILE_SHA256,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_PERCENT,
    STEP_SOURCE_COMMIT,
    density_snapshot_matrix,
    discover_step_source,
    fit_nozzle_pod_neural_operator,
    fit_step_coordinate_surrogate,
    load_step_height_archive,
    load_nozzle_centerlines,
    load_nozzle_fields,
    interpolate_nozzle_fields_locally,
    manufactured_step_velocity,
    pod_spectrum,
    predict_nozzle_pod_neural_operator,
    predict_nozzle_gap_aware_operator,
    select_nozzle_pod_rank,
    step_coordinate_features,
    validate_step_height_archives,
    validate_step_archives_against_source,
    validate_step_height_dataset,
    validate_step_contour_evidence,
    validate_step_teaching_results,
    validate_nozzle_flowmllab_results,
    validate_week9_evidence,
    zonal_velocity_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


class MahdaviDeepONetWeek9Tests(unittest.TestCase):
    def test_real_step_source_contract(self) -> None:
        self.assertEqual(STEP_SOURCE_COMMIT, "c3f211376b42b8dc30daad380eaef5e0ab800b5c")
        self.assertEqual(STEP_HEIGHT_PERCENT.tolist(), [16, 21, 25, 33, 44, 50, 58, 67, 75])
        self.assertEqual(STEP_HEIGHT_HELD_OUT_PERCENT.tolist(), [44, 67])
        self.assertEqual(len(STEP_HEIGHT_FILE_SHA256), 9)

    def test_compact_step_archives_are_file_separated_and_complete(self) -> None:
        report = validate_step_height_archives(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["archive_sha256"], STEP_HEIGHT_ARCHIVE_SHA256)
        self.assertTrue(report["file_level_test_isolation"])
        learning = load_step_height_archive(ROOT, split="learning")
        test = load_step_height_archive(ROOT, split="test")
        self.assertFalse(set(learning).intersection(test))
        self.assertEqual(sorted(test), [44, 67])
        self.assertEqual(len(test[44]["u"]), 21180)

    def test_training_store_cannot_supply_a_sealed_case(self) -> None:
        learning = load_step_height_archive(ROOT, split="learning")
        first = learning[16]
        bounds = (
            float(first["x"].min()), float(first["x"].max()),
            float(first["y"].min()), float(first["y"].max()),
        )
        with self.assertRaises(KeyError):
            fit_step_coordinate_surrogate(
                learning, [44], None, bounds_m=bounds, sample_size=10, max_iter=1
            )

    def test_step_coordinate_features_have_no_flow_input(self) -> None:
        x = np.array([0.0, 25.0e-9, 80.0e-9])
        y = np.array([12.0e-9, 4.0e-9, 8.0e-9])
        features = step_coordinate_features(
            0.44,
            x,
            y,
            bounds_m=(0.0, 85.0e-9, 0.0, 17.0e-9),
        )
        self.assertEqual(features.shape, (3, 8))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.all(features[:, 0] == 0.44))

    def test_real_step_dataset_when_checkout_is_available(self) -> None:
        source = discover_step_source(ROOT)
        if source is None:
            self.skipTest("pinned upstream step checkout/cache not available")
        report = validate_step_height_dataset(source)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["held_out_percent"], [44, 67])
        self.assertEqual(report["row_counts"]["44"], 21180)
        match = validate_step_archives_against_source(ROOT, source)
        self.assertEqual(match["status"], "pass")

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

    def test_full_field_nozzle_operator_uses_complete_development_cases(self) -> None:
        data = load_nozzle_fields(ROOT)
        self.assertEqual(data["density"].shape, (15, 31, 101))
        development = np.flatnonzero(~np.isin(data["pressure_kpa"], NOZZLE_HELD_OUT_KPA))
        selected_rank, rows = select_nozzle_pod_rank(
            data["pressure_kpa"], data["density"], development,
            candidate_ranks=(1,),
        )
        self.assertEqual(selected_rank, 1)
        self.assertEqual(len(rows), 1)
        fitted = fit_nozzle_pod_neural_operator(
            data["pressure_kpa"], data["density"], development, rank=1,
        )
        self.assertFalse(
            np.isin(fitted["train_pressures_kpa"], NOZZLE_HELD_OUT_KPA).any()
        )
        prediction = predict_nozzle_pod_neural_operator(fitted, np.array([16.0]))
        self.assertEqual(prediction.shape, (1, 31, 101))
        self.assertTrue(np.isfinite(prediction).all())

    def test_gap_aware_nozzle_operator_records_fixed_route(self) -> None:
        data = load_nozzle_fields(ROOT)
        development = np.flatnonzero(
            ~np.isin(data["pressure_kpa"], NOZZLE_HELD_OUT_KPA)
        )
        local, brackets = interpolate_nozzle_fields_locally(
            data["pressure_kpa"], data["density"], development,
            np.array([16.0, 25.0, 30.0]),
        )
        self.assertEqual(local.shape, (3, 31, 101))
        self.assertEqual([row["bracket_gap_kpa"] for row in brackets], [3.0, 2.0, 4.0])
        fitted = fit_nozzle_pod_neural_operator(
            data["pressure_kpa"], data["density"], development, rank=4,
        )
        prediction, routes = predict_nozzle_gap_aware_operator(
            fitted, data["pressure_kpa"], data["density"], development,
            np.array([16.0, 25.0, 30.0]), local_gap_limit_kpa=3.0,
        )
        self.assertEqual(prediction.shape, local.shape)
        self.assertEqual(
            [row["method"] for row in routes],
            ["local_field_interpolation", "local_field_interpolation", "pod_neural"],
        )

    def test_code_generated_nozzle_figures_and_metrics(self) -> None:
        report = validate_nozzle_flowmllab_results(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["held_out_pressures_kpa"], [16, 25, 30])
        self.assertEqual(len(report["held_out_metrics"]), 18)
        self.assertLess(
            max(float(row["full_field_relative_l2_percent"])
                for row in report["held_out_metrics"]),
            15.0,
        )

    def test_week9_evidence_contract(self) -> None:
        report = validate_week9_evidence(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["nozzle_cases"], 15)
        self.assertEqual(report["held_out_pressures_kpa"], [16, 25, 30])
        self.assertEqual(report["step_dataset"]["status"], "pass")
        self.assertFalse(report["step_teaching_validation"]["test_used_for_selection"])
        self.assertEqual(report["step_contour_evidence"]["article_cases"], [
            "Kn0p004", "Kn0p02", "H44", "H67"
        ])
        self.assertEqual(report["step_contour_evidence"]["final_paper_case_coverage"], [
            "Kn0p004", "Kn0p02", "Kn1", "H44", "H67"
        ])
        self.assertEqual(validate_step_teaching_results(ROOT)["selected_alpha"], 0.6)

    def test_step_contours_separate_article_and_independent_evidence(self) -> None:
        report = validate_step_contour_evidence(ROOT)
        self.assertEqual(report["status"], "pass")
        self.assertTrue(report["article_results_are_privileged_input"])
        self.assertFalse(report["independent_test_used_for_selection"])
        self.assertEqual(report["independent_test_cases"], [44, 67])
        self.assertEqual(report["final_paper_case_coverage"], [
            "Kn0p004", "Kn0p02", "Kn1", "H44", "H67"
        ])


if __name__ == "__main__":
    unittest.main()
