from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from flowmllab.probabilistic_uq import (
    GaussianPrediction,
    calibration_curve,
    fit_bayesian_linear_regression,
    fit_pod_gaussian_process,
    gaussian_crps,
    gaussian_negative_log_likelihood,
    interpolate_complete_cases,
    predict_bayesian_linear_regression,
    predict_pod_gaussian_process,
    relative_l2_per_case,
    rescale_prediction,
    validate_probabilistic_uq_evidence,
    validation_scale_factor,
)


ROOT = Path(__file__).resolve().parents[1]


class ProbabilisticUQTests(unittest.TestCase):
    def test_conjugate_scalar_posterior_matches_closed_form(self) -> None:
        design = np.ones((4, 1))
        observations = np.array([0.8, 1.0, 1.2, 1.0])
        noise_std = 0.5
        posterior = fit_bayesian_linear_regression(
            design,
            observations,
            prior_mean=np.array([0.0]),
            prior_covariance=np.array([[1.0]]),
            noise_std=noise_std,
        )
        expected_variance = 1.0 / (1.0 + len(observations) / noise_std**2)
        expected_mean = expected_variance * observations.sum() / noise_std**2
        self.assertAlmostEqual(float(posterior.covariance[0, 0]), expected_variance)
        self.assertAlmostEqual(float(posterior.mean[0]), expected_mean)

    def test_posterior_predictive_separates_latent_and_observation_noise(self) -> None:
        posterior = fit_bayesian_linear_regression(
            np.array([[1.0], [1.0]]),
            np.array([1.0, 1.2]),
            prior_mean=np.array([0.0]),
            prior_covariance=np.array([[2.0]]),
            noise_std=0.2,
        )
        latent = predict_bayesian_linear_regression(
            posterior, np.array([[1.0]]), include_observation_noise=False
        )
        observed = predict_bayesian_linear_regression(
            posterior, np.array([[1.0]]), include_observation_noise=True
        )
        self.assertAlmostEqual(
            float(observed.std[0] ** 2 - latent.std[0] ** 2), 0.2**2
        )

    def test_bayesian_fit_rejects_non_positive_definite_prior(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive definite"):
            fit_bayesian_linear_regression(
                np.eye(2),
                np.ones(2),
                prior_mean=np.zeros(2),
                prior_covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
                noise_std=0.1,
            )

    def test_gaussian_scores_have_known_value_at_the_mean(self) -> None:
        nll = gaussian_negative_log_likelihood(0.0, 0.0, 1.0)
        crps = gaussian_crps(0.0, 0.0, 1.0)
        self.assertAlmostEqual(nll, 0.5 * np.log(2.0 * np.pi))
        self.assertAlmostEqual(crps, (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi))

    def test_calibration_curve_recovers_gaussian_nominal_coverage(self) -> None:
        generator = np.random.default_rng(690)
        observations = generator.normal(size=50_000)
        curve = calibration_curve(
            observations,
            GaussianPrediction(
                mean=np.zeros_like(observations), std=np.ones_like(observations)
            ),
            levels=(0.5, 0.8, 0.95),
        )
        np.testing.assert_allclose(curve.observed, curve.nominal, atol=0.01)
        self.assertTrue(np.all(np.diff(curve.mean_width) > 0.0))

    def test_validation_scale_reaches_declared_empirical_coverage(self) -> None:
        observations = np.linspace(-2.0, 2.0, 101)
        raw = GaussianPrediction(
            mean=np.zeros_like(observations),
            std=np.full_like(observations, 0.25),
        )
        factor = validation_scale_factor(observations, raw, target_level=0.9)
        calibrated = rescale_prediction(raw, factor)
        coverage = calibration_curve(
            observations, calibrated, levels=(0.9,)
        ).observed[0]
        self.assertGreaterEqual(float(coverage), 0.9)
        self.assertGreater(factor, 1.0)

    def test_complete_case_interpolation_is_exact_for_linear_fields(self) -> None:
        reynolds = np.array([100.0, 200.0, 400.0])
        fields = reynolds[:, None, None] * np.array([[[1.0, -0.5]]])
        prediction = interpolate_complete_cases(reynolds, fields, [150.0, 300.0])
        expected = np.array([150.0, 300.0])[:, None, None] * np.array(
            [[[1.0, -0.5]]]
        )
        np.testing.assert_allclose(prediction, expected)
        with self.assertRaises(ValueError):
            interpolate_complete_cases(reynolds, fields, 450.0)

    def test_pod_gp_reconstructs_training_cases_and_retains_split(self) -> None:
        reynolds = np.array([100.0, 180.0, 260.0, 400.0])
        spatial_mode = np.array([[1.0, 0.5], [-0.25, 0.75]])
        fields = np.stack(
            [(1.0 + 0.002 * value) * spatial_mode for value in reynolds]
        )
        model = fit_pod_gaussian_process(
            reynolds,
            fields,
            rank=1,
            length_scale=1.0,
            noise_level=1.0e-10,
        )
        training_prediction = predict_pod_gaussian_process(model, reynolds)
        error = relative_l2_per_case(fields, training_prediction.mean)
        self.assertLess(float(error.max()), 1.0e-6)
        np.testing.assert_array_equal(model.train_reynolds, reynolds)
        self.assertEqual(training_prediction.mean.shape, fields.shape)
        self.assertEqual(training_prediction.std.shape, fields.shape)

    def test_pod_gp_reports_more_spread_away_from_training_support(self) -> None:
        reynolds = np.array([100.0, 200.0, 300.0, 400.0])
        mode = np.array([[1.0, -1.0], [0.5, -0.5]])
        fields = np.stack([np.sin(value / 150.0) * mode for value in reynolds])
        model = fit_pod_gaussian_process(
            reynolds,
            fields,
            rank=1,
            length_scale=0.8,
            noise_level=1.0e-8,
        )
        at_training = predict_pod_gaussian_process(model, 200.0)
        outside = predict_pod_gaussian_process(model, 600.0)
        self.assertGreater(float(outside.std.mean()), float(at_training.std.mean()))

    def test_retained_probabilistic_uq_evidence(self) -> None:
        summary = validate_probabilistic_uq_evidence(ROOT)
        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["blind_cases"], [175.0, 275.0, 375.0])
        self.assertLess(summary["mean_pod_gp_relative_L2_uv"], 0.01)
        self.assertLess(summary["aggregate_calibrated_90_coverage"], 0.9)


if __name__ == "__main__":
    unittest.main()
