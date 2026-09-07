import unittest
import numpy as np
from pathlib import Path
import hashlib, json
from flowmllab.state_estimation import (fit_reduced_kalman, kalman_filter,
    reconstruct_states, pointwise_interval_coverage)


class StateEstimationTests(unittest.TestCase):
    def test_filter_tracks_known_linear_low_rank_system(self):
        rng = np.random.default_rng(4)
        modes, _ = np.linalg.qr(rng.normal(size=(24, 2)))
        states = np.array([[np.cos(.12*k), np.sin(.12*k)] for k in range(100)])
        fields = states @ modes.T
        model = fit_reduced_kalman(fields[:60], rank=2, sensor_count=5, noise_sigma=.01)
        observations = fields[60:, model.sensor_indices] + rng.normal(0, .01, (40, 5))
        filtered, covariances, innovations = kalman_filter(model, observations)
        prediction = reconstruct_states(model, filtered)
        self.assertLess(np.linalg.norm(prediction-fields[60:]) / np.linalg.norm(fields[60:]), .15)
        self.assertEqual(covariances.shape, (40, 2, 2))
        self.assertEqual(innovations.shape, (40, 5))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariances) >= -1e-12))

    def test_filter_is_causal(self):
        rng = np.random.default_rng(5); fields = rng.normal(size=(30, 12))
        model = fit_reduced_kalman(fields[:20], rank=3, sensor_count=4, noise_sigma=.1)
        y = fields[20:, model.sensor_indices]
        full = kalman_filter(model, y)[0]
        prefix = kalman_filter(model, y[:5])[0]
        np.testing.assert_allclose(full[:5], prefix)

    def test_interval_summary_has_valid_range(self):
        rng = np.random.default_rng(6); fields = rng.normal(size=(30, 10))
        model = fit_reduced_kalman(fields[:20], rank=3, sensor_count=5, noise_sigma=.2)
        states, covs, _ = kalman_filter(model, fields[20:, model.sensor_indices])
        coverage, width = pointwise_interval_coverage(model, states, covs, fields[20:])
        self.assertTrue(0 <= coverage <= 1); self.assertGreater(width, 0)

    def test_retained_evidence_and_source_hashes(self):
        root = Path(__file__).resolve().parents[1]
        retained = json.loads((root/'results/week07_2_state_estimation/metrics.json').read_text())
        self.assertEqual(retained['selected']['rank'], 8)
        self.assertEqual(retained['selected']['sensor_count'], 32)
        for relative, digest in retained['source_hashes'].items():
            self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(), digest)
        kalman = np.mean([row['test']['relative_l2'] for row in retained['methods']['kalman']])
        sensor = np.mean([row['test']['relative_l2'] for row in retained['methods']['sensor_only']])
        open_loop = np.mean([row['test']['relative_l2'] for row in retained['methods']['open_loop']])
        coverage = np.mean([row['test_pointwise_95_coverage'] for row in retained['coverage']])
        self.assertLess(kalman, sensor)
        self.assertLess(kalman, open_loop)
        self.assertLess(coverage, .75)  # Retained under-coverage, not a success gate.


if __name__ == '__main__': unittest.main()
