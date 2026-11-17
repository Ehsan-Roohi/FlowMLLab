import unittest
import numpy as np
from flowmllab.pressure_velocity import divergence
from flowmllab.pressure_velocity_studies import predictor_fields,pressure_work_study,startup_study


class TeachingStudyTests(unittest.TestCase):
    def test_compatible_closed_trial_fields(self):
        for u,v in predictor_fields(8):
            self.assertTrue(np.all(u[:,[0,-1]]==0)); self.assertTrue(np.all(v[[0,-1],:]==0))
            self.assertLess(abs(divergence(u,v,1/8).sum()),1e-12)

    def test_projection_and_reused_factor(self):
        rows,_=pressure_work_study(cells=(8,),repeats=1,rhs_count=2,tolerances=(1e-10,))
        self.assertEqual(len(rows),3)
        for r in rows:
            self.assertTrue(r['linear_solver_converged'])
            self.assertLess(r['max_divergence_linf'],1e-7)
            self.assertLess(r['max_pressure_relative_l2_vs_lu'],1e-7)

    def test_transient_same_time_and_reference_checks(self):
        rows,_,checks=startup_study(cells=4,end_time=.04,steps=(.02,),reference_dt=.0025,repeats=1)
        self.assertEqual(len(rows),5)
        self.assertTrue(all(abs(r['physical_time']-.04)<1e-12 for r in rows))
        self.assertTrue(all(r['continuity_linf']<1e-9 for r in rows))
        self.assertNotIn('ghia',str(rows).lower())
        self.assertGreaterEqual(checks['reference_dt_change'],0)


if __name__=='__main__':unittest.main()
