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


if __name__ == "__main__":
    unittest.main()
