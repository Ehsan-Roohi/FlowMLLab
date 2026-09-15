import unittest
import numpy as np
from flowmllab.rans_closure import closure_features, relative_l2, cell_widths


class ClosureAuditTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([.01,.03,.08,.2,.7])
        self.a=np.c_[self.y,2*self.y,self.y*.2,np.ones(5)*2,np.zeros(5)]

    def test_exact_linear_velocity_features(self):
        f=closure_features(self.a,100)
        np.testing.assert_allclose(f[:,0],.1)
        np.testing.assert_allclose(f[:,1],2*(.01+.1*self.y))

    def test_cap(self):
        self.a[:,1]*=10000
        np.testing.assert_allclose(closure_features(self.a)[:,1],.995)

    def test_invalid_grid(self):
        with self.assertRaises(ValueError): closure_features(self.a[::-1])

    def test_metrics_and_quadrature(self):
        self.assertAlmostEqual(relative_l2([2,4],[1,2]),1)
        w=cell_widths(self.y)
        self.assertAlmostEqual(w.sum(),1)
        self.assertTrue((w>0).all())
        self.assertEqual(relative_l2(self.y,self.y,w),0)
        with self.assertRaises(ValueError): relative_l2([0],[0])

if __name__=='__main__': unittest.main()
