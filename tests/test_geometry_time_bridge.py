"""Behavioral checks for the optional manufactured geometry/time bridge."""
import unittest
import numpy as np
from flowmllab.geometry_time_bridge import (
    manufactured_cases, split_before_windows, masked_mse, autonomous,
    teacher_forced, harmonic_summary, phase_metrics,
)


class GeometryTimeContracts(unittest.TestCase):
    def test_split_keeps_parameter_variants_and_windows_in_one_geometry_group(self):
        cases = manufactured_cases()
        parts = dict(train=['g0', 'g1', 'g3', 'g4'], validation=['g2'], test=['g5'])
        rows = split_before_windows(cases, parts, 4)
        self.assertEqual([len(rows[k]) for k in parts], [168, 42, 42])
        for name, items in rows.items():
            for key, start, target in items:
                self.assertIn(cases[key]['geometry_id'], parts[name])
                self.assertEqual(target-start, 4)
                self.assertLess(target, len(cases[key]['time']))
        # Two parameter variants of a geometry must never masquerade as groups.
        with self.assertRaises(ValueError):
            split_before_windows(cases, dict(train=['g0', 'g1', 'g2'], test=['g0', 'g3', 'g4', 'g5']))
        with self.assertRaises(ValueError):
            split_before_windows(cases, dict(train=['g0'], test=['g5']))

    def test_sdf_sign_boundary_and_geometry_are_not_flow_validation(self):
        cases = manufactured_cases()
        circle = cases['g0_rate0.2']
        box = cases['g3_rate0.2']
        self.assertLess(circle['sdf'][8, 12], 0)
        self.assertLess(box['sdf'][8, 12], 0)
        self.assertGreater(circle['sdf'][0, 0], 0)
        self.assertEqual(circle['values'].shape, (25, 17, 25, 2))
        np.testing.assert_array_equal(circle['sdf'], cases['g0_rate0.3']['sdf'])

    def test_masked_loss_has_hand_checked_weighting_and_batch_invariance(self):
        p = np.array([[[[2.], [100.]]], [[[3.], [3.]]]])
        q = np.zeros_like(p)
        m = np.array([[[True, False]], [[True, True]]])
        result = masked_mse(p, q, m)
        np.testing.assert_allclose(result['per_case'], [4, 9])
        self.assertEqual(result['equal_case'], 6.5)
        self.assertAlmostEqual(result['pooled'], 22/3)
        duplicate = masked_mse(np.concatenate([p,p]), np.concatenate([q,q]), np.concatenate([m,m]))
        self.assertEqual(result['equal_case'], duplicate['equal_case'])
        self.assertEqual(result['pooled'], duplicate['pooled'])
        p[0,0,1,0] = -1e5
        self.assertEqual(masked_mse(p,q,m)['equal_case'], 6.5)
        with self.assertRaises(ValueError):
            masked_mse(p[:,None],q[:,None],m)  # extra singleton cannot silently broadcast
        with self.assertRaises(ValueError):
            masked_mse(p,q,np.zeros_like(m))

    def test_future_poison_rejects_teacher_forced_control(self):
        observed = np.arange(1., 8.)[:,None]
        changed = observed.copy(); changed[2:] += 1000
        predictor = lambda history: history[-1] + .25
        a = autonomous(predictor, observed[:2], 5)
        b = autonomous(predictor, changed[:2], 5)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a[:,0], [2.25,2.5,2.75,3.,3.25])
        tf = teacher_forced(predictor,observed,2)
        tf_poison = teacher_forced(predictor,changed,2)
        self.assertEqual(tf[0,0], tf_poison[0,0])
        self.assertFalse(np.array_equal(tf[1:],tf_poison[1:]))
        self.assertNotEqual(float(np.mean((a-observed[2:])**2)),float(np.mean((b-changed[2:])**2)))

    def test_predictor_mutation_cannot_change_initialization(self):
        initial = np.array([[1.],[2.]])
        saved = initial.copy()
        def mutating(history):
            history[:] = 7
            return history[-1]
        autonomous(mutating,initial,2)
        np.testing.assert_array_equal(initial,saved)
        with self.assertRaises(ValueError):
            autonomous(lambda h: np.array([np.nan]),initial,2)
        with self.assertRaises(ValueError):
            autonomous(lambda h: np.ones(2),initial,2)

    def test_harmonic_phase_uses_common_origin_and_constant_is_undefined(self):
        t = 13.5 + np.arange(100)*.1
        y = .4 + 2*np.sin(2*np.pi*.2*(t-t[0])+.3)
        a = harmonic_summary(t,y,.2)
        b = harmonic_summary(t+70,y,.2)
        self.assertAlmostEqual(a['amplitude'],2)
        self.assertAlmostEqual(a['phase'],.3)
        self.assertAlmostEqual(a['phase'],b['phase'])
        self.assertLess(a['residual_mse'],1e-25)
        self.assertIsNone(harmonic_summary(t,np.full_like(t,.1),.2)['phase'])
        self.assertIsNone(harmonic_summary(t,np.ones_like(t),.2)['r_squared'])
        mixed = y + .5*np.sin(2*np.pi*.5*(t-t[0]))
        self.assertGreater(harmonic_summary(t,mixed,.2)['residual_mse'],.1)

    def test_circular_metrics_do_not_hide_cancellation_or_missingness(self):
        result = phase_metrics([np.pi/2,-np.pi/2,None],[0.,0.,0.])
        self.assertAlmostEqual(result['signed_mean'],0.)
        self.assertAlmostEqual(result['mean_absolute'],np.pi/2)
        self.assertEqual(result['undefined_pairs'],1)
        wrapped = phase_metrics([np.deg2rad(-179)],[np.deg2rad(179)])
        self.assertAlmostEqual(wrapped['mean_absolute'],np.deg2rad(2))
        self.assertIsNone(phase_metrics([None],[0.])['rms'])
        with self.assertRaises(ValueError):
            phase_metrics([float('nan')],[None])


if __name__ == '__main__':
    unittest.main()
