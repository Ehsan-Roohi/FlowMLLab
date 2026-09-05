"""Regression tests for compression sensing, anchored maps, and deployment."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from flowmllab.mahdavi_deeponet import load_nozzle_fields
from flowmllab.nozzle_transport import (
    FIELDS, compression_surface, fit_transport_pod, load_transport_model,
    predict_transport_pod, predict_with_symmetry, save_transport_model, warp_fields,
)

ROOT = Path(__file__).resolve().parents[1]


class NozzleTransportTests(unittest.TestCase):
    def test_sensor_rejects_expansion_and_outlet_gradient(self):
        x = np.linspace(0, 1, 101)
        # A larger negative expansion and a much larger positive outlet jump
        # must not displace the known interior compression.
        rho = 4 - 2*np.tanh((x-.25)/.02) + .6*np.tanh((x-.66)/.025)
        rho[-2:] += 10
        detected = compression_surface(x, rho[None, :], .25)
        self.assertLess(abs(detected[0]-.66), .01)

    def test_anchor_map_preserves_inlet_throat_and_outlet(self):
        x = np.linspace(0, 1, 101)
        values = np.stack((np.sin(x), x*x), axis=-1)[None, :, :]
        mapped = warp_fields(values, x, .25, np.array([.8]), np.array([.6]))
        np.testing.assert_array_equal(mapped[:, [0, 25, 100]], values[:, [0, 25, 100]])
        np.testing.assert_allclose(mapped[0, 60], values[0, 80])
        np.testing.assert_array_equal(warp_fields(values, x, .25, np.array([.8]), np.array([.8])), values)

    def test_crossing_map_is_rejected(self):
        with self.assertRaises(ValueError):
            warp_fields(np.ones((1, 101, 2)), np.linspace(0, 1, 101), .25,
                        np.array([.8]), np.array([.2]))

    def test_real_training_only_model_checkpoint_and_extrapolation(self):
        data = load_nozzle_fields(ROOT)
        train = ~np.isin(data["pressure_kpa"], [16, 25, 30])
        fields = np.stack([data[name][train] for name in FIELDS], axis=-1)
        for branch in ("polynomial", "neural"):
            with self.subTest(branch=branch):
                model = fit_transport_pod(data["pressure_kpa"][train], fields,
                                           data["x_m"], data["y_m"], rank=2, branch=branch)
                self.assertFalse(np.isin(model["training_pressures_kpa"], [16, 25, 30]).any())
                before = predict_transport_pod(model, [16, 25, 30])
                self.assertEqual(before.shape, (3, 31, 101, 6))
                self.assertTrue(np.isfinite(before).all())
                constrained = predict_with_symmetry(model, [16, 25, 30], symmetry_y_m=92e-6)
                np.testing.assert_array_equal(constrained[:, -1, :, 2], 0)
                np.testing.assert_array_equal(constrained[:, :-1], before[:, :-1])
                np.testing.assert_array_equal(constrained[..., [0, 1, 3, 4, 5]], before[..., [0, 1, 3, 4, 5]])
                with self.assertRaises(ValueError):
                    predict_with_symmetry(model, [16], symmetry_y_m=90e-6)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary)/"model.npz"
                    save_transport_model(model, path)
                    after = predict_transport_pod(load_transport_model(path), [16, 25, 30])
                np.testing.assert_array_equal(before, after)
                with self.assertRaises(ValueError):
                    predict_transport_pod(model, [34])


if __name__ == "__main__":
    unittest.main()
