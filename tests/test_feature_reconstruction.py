import unittest
import numpy as np
from flowmllab.feature_reconstruction import (
    diagnostics, manufactured_case, dice, additive_moments,
    central_heat_numerator, reconstruct, nrmse, sampling_case,
    fit_spectral_prior, support_score,
)


class FeatureReconstructionTests(unittest.TestCase):
    def test_rotation_shear_compression(self):
        x = np.linspace(-1, 1, 21)
        X, Y = np.meshgrid(x, x)
        h = x[1]-x[0]
        rotation = diagnostics(-Y, X, h, h)
        np.testing.assert_allclose(rotation['swirl'], 1, atol=1e-12)
        shear = diagnostics(Y, 0*X, h, h)
        np.testing.assert_allclose(shear['omega'], -1, atol=1e-12)
        np.testing.assert_allclose(shear['swirl'], 0, atol=1e-12)
        compression = diagnostics(-X, -Y, h, h)
        np.testing.assert_allclose(compression['compression'], 2, atol=1e-12)
        np.testing.assert_allclose(compression['swirl'], 0, atol=1e-12)

    def test_galilean_shift(self):
        c = manufactured_case(1)
        h = c['x'][1]-c['x'][0]
        shifted = diagnostics(c['u']+7, c['v']-4, h, h)
        np.testing.assert_allclose(shifted['qd'], c['qd'], atol=1e-10)

    def test_multilabel_and_dice(self):
        self.assertEqual(dice([0, 0], [0, 0]), 1.)
        self.assertEqual(dice([1, 0], [0, 1]), 0.)
        self.assertTrue(any(np.any(c['labels'].all(axis=-1))
                            for c in [manufactured_case(i) for i in range(10)]))

    def test_additive_first_equals_direct(self):
        rng = np.random.default_rng(41)
        a = rng.normal(size=(100, 3)) + [2, 0, 0]
        b = rng.normal(size=(70, 3)) + [-1, 1, 0]
        ra, rb = additive_moments(a), additive_moments(b)
        pooled = {k: ra[k]+rb[k] for k in ra}
        all_v = np.concatenate([a, b])
        peculiar = all_v-all_v.mean(axis=0)
        direct = (np.sum(peculiar**2, axis=1)[:, None]*peculiar).sum(axis=0)
        np.testing.assert_allclose(central_heat_numerator(pooled), direct, atol=1e-10)
        self.assertGreater(np.linalg.norm(direct-central_heat_numerator(ra)
                                          -central_heat_numerator(rb)), 1)

    def test_gain_endpoints_and_mean(self):
        rng = np.random.default_rng(1)
        obs, prior = rng.normal(size=(2, 12, 12))
        np.testing.assert_allclose(reconstruct(obs, prior, np.ones_like(obs)), obs, atol=1e-12)
        out = reconstruct(obs, prior, np.zeros_like(obs))
        np.testing.assert_allclose(out, prior+obs.mean()-prior.mean(), atol=1e-12)
        self.assertAlmostEqual(out.mean(), obs.mean())
        with self.assertRaises(ValueError):
            reconstruct(obs, prior, np.ones_like(obs)*1.1)

    def test_training_and_support(self):
        cases = [sampling_case(i, size=12) for i in range(4)]
        prior, gain = fit_spectral_prior(cases)
        self.assertTrue(np.all((gain >= 0) & (gain <= 1)))
        self.assertEqual(gain[0, 0], 1.)
        self.assertGreater(support_score(prior+4, prior, .1),
                           support_score(prior+.1, prior, .1))

    def test_weighted_error(self):
        self.assertAlmostEqual(nrmse(np.ones(3)*1.1, np.ones(3)), .1)
        with self.assertRaises(ValueError):
            nrmse(np.ones(3), np.zeros(3))


if __name__ == '__main__':
    unittest.main()
