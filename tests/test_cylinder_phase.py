import unittest

import numpy as np

from flowmllab.cylinder_phase import PhaseDecoder, align_and_predict, design, interpolate


class CylinderPhaseTests(unittest.TestCase):
    def test_design_has_expected_columns(self):
        self.assertEqual(design(np.arange(4), 3).shape, (4, 7))

    def test_interpolation_is_bracketed(self):
        a = PhaseDecoder(90, .17, 1, np.zeros((3, 2, 2, 3), np.float32))
        b = PhaseDecoder(110, .19, 1, np.ones((3, 2, 2, 3), np.float32))
        m = interpolate([a, b], 100)
        self.assertAlmostEqual(m.strouhal, .18)
        np.testing.assert_allclose(m.coefficients, .5)

    def test_autonomous_prediction_preserves_periodic_amplitude(self):
        coefficients = np.zeros((3, 1, 1, 3), np.float32)
        coefficients[1, 0, 0, 0] = 1
        model = PhaseDecoder(100, .2, 1, coefficients)
        phase = .4 + 2*np.pi*.2*.1*np.arange(4)
        initial = np.tensordot(design(phase, 1), coefficients, axes=(1, 0))
        prediction, _ = align_and_predict(model, initial, 100, delta_t_star=.1)
        self.assertGreater(np.ptp(prediction[:, 0, 0, 0]), 1.9)


if __name__ == "__main__":
    unittest.main()
