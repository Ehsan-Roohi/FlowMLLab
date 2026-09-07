"""The assignment checker must reject semantic faults, not only accept a solution."""
import unittest
import numpy as np
from qa.check_week01_1_candidate import assess


def reference(x, y, u, v):
    x, y, u, v = (np.asarray(a) for a in (x, y, u, v))
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2
        or u.shape != (y.size, x.size) or v.shape != u.shape
        or not all(np.isfinite(a).all() for a in (x, y, u, v))
        or not (np.diff(x) > 0).all() or not (np.diff(y) > 0).all()):
        raise ValueError('invalid grid or velocity')
    return np.trapezoid(u[:, -1]-u[:, 0], y)+np.trapezoid(v[-1, :]-v[0, :], x)


class CandidateCheckerTests(unittest.TestCase):
    def test_reference_and_all_four_levels(self):
        result = assess(reference)
        self.assertEqual(result['decision'], 'accept')
        self.assertEqual({r['level'] for r in result['checks']},
                         {'unit', 'physical invariant', 'numerical regression', 'baseline/reference'})

    def test_always_zero_is_rejected(self):
        result = assess(lambda *args: 0.)
        self.assertEqual(result['decision'], 'reject')
        self.assertTrue(any(not r['passed'] and 'leak' in r['name'] for r in result['checks']))

    def test_reversed_normal_sign_is_rejected(self):
        self.assertEqual(assess(lambda *args: -reference(*args))['decision'], 'reject')
