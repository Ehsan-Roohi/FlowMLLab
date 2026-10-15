"""Higher-Re benchmark selection and preservation of the qualified branch."""
import unittest
import numpy as np
from flowmllab.pressure_velocity import CavityConfig,GHIA_X,GHIA_Y
from flowmllab.pressure_velocity_experiments import ExperimentalConfig,REFERENCE,benchmark_errors

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

    def test_experiment_does_not_unlock_baseline_config(self):
        with self.assertRaises(ValueError):CavityConfig(reynolds=400)
        for re in [0,-1,float('nan'),float('inf'),5000]:
            with self.assertRaises(ValueError):ExperimentalConfig(reynolds=re)
        self.assertEqual(ExperimentalConfig(reynolds=1000).reynolds,1000)
        with self.assertRaises(ValueError):ExperimentalConfig(dt=-1)

if __name__=='__main__':unittest.main()
