"""Higher-Re benchmark selection and preservation of the qualified branch."""
import unittest
import numpy as np
from flowmllab.pressure_velocity import CavityConfig,GHIA_X,GHIA_Y
from flowmllab.pressure_velocity_experiments import ExperimentalConfig,REFERENCE,benchmark_errors,benchmark_valid_mask
from common.w4utils import GHIA, ghia_errors, ghia_valid_mask

class ExperimentalPressureVelocityTests(unittest.TestCase):
    def test_reference_uses_requested_reynolds_number(self):
        for re in (400,1000):
            x=GHIA_X[::-1];y=GHIA_Y[::-1]
            result={'config':{'reynolds':re},'x':x,'y':y,
                    'u':np.tile(REFERENCE[re]['u'][::-1],(len(x),1)).T,
                    'v':np.tile(REFERENCE[re]['v'][::-1],(len(y),1))}
            self.assertLess(max(benchmark_errors(result).values()),1e-14)
            result['config']['reynolds']=1000 if re==400 else 400
            self.assertGreater(benchmark_errors(result)['ghia_u_relative_l2'],.05)

    def test_re400_anomalous_v_is_preserved_but_not_scored(self):
        x=GHIA_X[::-1]; y=GHIA_Y[::-1]
        ref=REFERENCE[400]
        result={'config':{'reynolds':400},'x':x,'y':y,
                'u':np.tile(ref['u'][::-1],(len(x),1)).T,
                'v':np.tile(ref['v'][::-1],(len(y),1))}
        anomaly=np.flatnonzero(np.isclose(x,0.9063,rtol=0,atol=1e-5))
        self.assertEqual(len(anomaly),1)
        self.assertAlmostEqual(ref['v'][5],-0.23827)
        self.assertAlmostEqual(GHIA[400]['v'][5],-0.23827)
        result['v'][:,anomaly[0]]+=0.2  # Only the known anomalous reference point differs.
        metrics=benchmark_errors(result)
        self.assertLess(metrics['ghia_v_relative_l2'],1e-12)
        self.assertGreater(metrics['ghia_v_relative_l2_including_x09063'],.05)
        self.assertAlmostEqual(metrics['ghia_v_relative_l2_without_x09063'],metrics['ghia_v_relative_l2'])
        self.assertEqual(benchmark_valid_mask(400,'v',GHIA_X).sum(),len(GHIA_X)-1)
        self.assertTrue(benchmark_valid_mask(1000,'v',GHIA_X).all())
        self.assertEqual(ghia_valid_mask(400,'v').sum(),len(GHIA_X)-1)
        self.assertTrue(ghia_valid_mask(100,'v').all())
        result['Re']=400; result['U']=1.
        self.assertLess(ghia_errors(result)[1],1e-12)

    def test_experiment_does_not_unlock_baseline_config(self):
        with self.assertRaises(ValueError):CavityConfig(reynolds=400)
        for re in [0,-1,float('nan'),float('inf'),5000]:
            with self.assertRaises(ValueError):ExperimentalConfig(reynolds=re)
        self.assertEqual(ExperimentalConfig(reynolds=1000).reynolds,1000)
        with self.assertRaises(ValueError):ExperimentalConfig(dt=-1)

if __name__=='__main__':unittest.main()
