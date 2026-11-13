from __future__ import annotations

import unittest

import numpy as np

from flowmllab import cylinder_cnn


class CylinderCNNTests(unittest.TestCase):
    def test_temporal_windows_use_four_prior_frames(self) -> None:
        windows = cylinder_cnn.temporal_windows(8, history=4)
        self.assertEqual(len(windows), 4)
        np.testing.assert_array_equal(windows[0].history, [0, 1, 2, 3])
        self.assertEqual(windows[0].target, 4)
        np.testing.assert_array_equal(windows[-1].history, [3, 4, 5, 6])
        self.assertEqual(windows[-1].target, 7)

    def test_stack_history_is_time_major(self) -> None:
        base = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
        fields = {"u": base, "v": base + 100, "p": base + 200}
        stacked = cylinder_cnn.stack_history(fields, [0, 2])
        self.assertEqual(stacked.shape, (2, 4, 6))
        np.testing.assert_array_equal(stacked[..., 0], fields["u"][0])
        np.testing.assert_array_equal(stacked[..., 4], fields["v"][2])

    def test_polynomial_forecast_uses_the_same_four_history_frames(self) -> None:
        time = np.arange(4.0)[:, None, None]
        quadratic = 2.0 + 0.5 * time + 3.0 * time**2
        predicted = cylinder_cnn.polynomial_forecast(
            quadratic, horizon=2, degree=2
        )
        np.testing.assert_allclose(predicted, 2.0 + 0.5 * 5.0 + 3.0 * 5.0**2)

    def test_vorticity_and_divergence_for_linear_field(self) -> None:
        y, x = np.mgrid[:7, :9]
        u = -2.0 * y + 0.5 * x
        v = 3.0 * x - 0.5 * y
        omega = cylinder_cnn.vorticity(u, v, diameter=2.0)
        div = cylinder_cnn.divergence(u, v, diameter=2.0)
        np.testing.assert_allclose(omega, 10.0)
        np.testing.assert_allclose(div, 0.0, atol=1.0e-12)

    def test_stationwise_metrics_detect_amplitude_loss(self) -> None:
        time = np.linspace(0, 2 * np.pi, 6, endpoint=False)[:, None, None]
        y = np.linspace(-1, 1, 12)[None, :, None]
        x = np.arange(20)[None, None, :]
        truth = np.sin(time + 2 * y + 0.1 * x)
        prediction = 0.8 * truth
        rows = cylinder_cnn.stationwise_wake_metrics(
            truth, prediction, center_x=2.0, diameter=2.0, stations=(2.0, 4.0)
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["enstrophy_ratio"], 0.64, places=12)
            self.assertLess(row["normalized_psd_relative_l2"], 1.0e-12)
            self.assertAlmostEqual(row["complex_spectral_coherence"], 1.0, places=12)

    def test_multiscale_model_preserves_shape_and_initial_persistence(self) -> None:
        try:
            model = cylinder_cnn.build_multiscale_predictor(filters=4)
        except ImportError:
            self.skipTest("optional TensorFlow dependency is not installed")
        values = np.zeros((2, 16, 20, 14), dtype=np.float32)
        values[..., 9:12] = [1.0, -0.25, 0.4]
        values[..., -1] = 1.0
        prediction = np.asarray(model(values, training=False))
        self.assertEqual(prediction.shape, (2, 16, 20, 3))
        np.testing.assert_allclose(prediction, values[..., 9:12], atol=1.0e-7)

    def test_cubic_extrapolation_weights_match_polynomial_forecast(self) -> None:
        weights = cylinder_cnn.extrapolation_weights(4, base="cubic")
        np.testing.assert_allclose(weights, [-1.0, 4.0, -6.0, 4.0], atol=1.0e-12)
        history = np.random.default_rng(3).normal(size=(4, 5, 6))
        np.testing.assert_allclose(
            np.tensordot(weights, history, axes=(0, 0)),
            cylinder_cnn.polynomial_forecast(history, degree=3),
            atol=1.0e-10,
        )
        np.testing.assert_array_equal(
            cylinder_cnn.extrapolation_weights(4), [0.0, 0.0, 0.0, 1.0]
        )
        with self.assertRaises(ValueError):
            cylinder_cnn.extrapolation_weights(4, base="quintic")

    def test_periodic_cubic_model_starts_from_cubic_extrapolation(self) -> None:
        try:
            model = cylinder_cnn.build_multiscale_predictor(
                filters=4, base="cubic", periodic_y=True
            )
            reference = cylinder_cnn.build_multiscale_predictor(filters=4)
        except ImportError:
            self.skipTest("optional TensorFlow dependency is not installed")
        self.assertEqual(model.count_params(), reference.count_params())
        rng = np.random.default_rng(5)
        values = rng.normal(size=(2, 16, 20, 14)).astype(np.float32)
        values[..., -1] = 1.0
        values[:, 6:9, 4:7, -1] = 0.0
        prediction = np.asarray(model(values, training=False))
        frames = [values[..., 3 * k : 3 * k + 3] for k in range(4)]
        expected = -frames[0] + 4 * frames[1] - 6 * frames[2] + 4 * frames[3]
        expected[..., :2] *= values[..., -1:]
        self.assertEqual(prediction.shape, (2, 16, 20, 3))
        np.testing.assert_allclose(prediction, expected, atol=1.0e-5)

    def test_periodic_padding_is_equivariant_to_transverse_rolls(self) -> None:
        try:
            import tensorflow as tf
            model = cylinder_cnn.build_multiscale_predictor(
                filters=4, base="cubic", periodic_y=True
            )
        except ImportError:
            self.skipTest("optional TensorFlow dependency is not installed")
        tf.keras.utils.set_random_seed(11)
        for weight in model.weights:
            weight.assign(0.1 * tf.random.normal(weight.shape))
        values = np.random.default_rng(7).normal(size=(1, 16, 24, 14)).astype(np.float32)
        values[..., -1] = 1.0
        rolled = np.roll(values, 4, axis=1)
        prediction = np.asarray(model(values, training=False))
        rolled_prediction = np.asarray(model(rolled, training=False))
        np.testing.assert_allclose(
            np.roll(prediction, 4, axis=1), rolled_prediction, atol=1.0e-4
        )


if __name__ == "__main__":
    unittest.main()
