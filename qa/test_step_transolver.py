"""Scientific-contract tests; CPU only, no retraining required."""
import unittest
import importlib.util
import numpy as np
if not all(importlib.util.find_spec(name) for name in ('torch', 'einops')):
    raise unittest.SkipTest('Install the optional FlowMLLab [transolver] dependencies')
import torch
from train_step_transolver import Model, MODEL_CONFIG, load_data, make_features
from train_week15_vanilla_deeponet import case_metrics

class TransolverTests(unittest.TestCase):
    def test_split_and_inputs(self):
        d,_,groups,_=load_data()
        before=make_features(d)
        d['raw'][:]=np.nan
        np.testing.assert_array_equal(before,make_features(d))
        self.assertEqual([len(groups[k]) for k in ('train','validation','test')],[106,21,3])

    def test_permutation_and_backward(self):
        torch.set_num_threads(1);torch.manual_seed(4)
        model=Model(**MODEL_CONFIG).eval()
        x=torch.randn(1,63,2);f=torch.randn(1,63,2)
        out=model(x,f);perm=torch.randperm(63)
        torch.testing.assert_close(out[:,perm],model(x[:,perm],f[:,perm]),rtol=2e-5,atol=2e-6)
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_metrics_and_solid_exclusion(self):
        t=np.array([[1.,0.,1.],[-1.,0.,2.],[99.,99.,99.]])
        p=t.copy();p[:2,2]+=100;p[2]=1e8
        m=case_metrics(t,p,np.array([True,True,False]))
        self.assertEqual(m['velocity_percent'],0.)
        self.assertEqual(m['pressure_percent'],0.)
        self.assertEqual(m['reverse_iou'],1.)

if __name__=='__main__':unittest.main()
