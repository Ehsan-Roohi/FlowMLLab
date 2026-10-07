"""Behavioral contracts: causality, selection, leakage and artifact round trips."""
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name != 'torch':
        raise
    raise unittest.SkipTest('Transformer contracts require the optional .[transformer,test] environment') from exc
from flowmllab.transformer_course import (
    Protocol, Representation, CausalDecoder, SensorSet, attention, sequence_windows,
    seed_all, fit, safe_rollout_loss, error_components, transfer_indices,
    pack_prediction, prediction, spectral_fit, checkpoint_model, checkpoint_agreement,
)

class CourseContracts(unittest.TestCase):
    def setUp(self):
        seed_all(17)

    def test_causal_prefix_and_unmasked_counterexample(self):
        model=CausalDecoder();x=torch.randn(3,4,8);changed=x.clone();changed[:,2:]+=20
        torch.testing.assert_close(model(x)[:,:2],model(changed)[:,:2])
        for block in model.blocks: block.causal=False
        self.assertGreater(float((model(x)[:,:2]-model(changed)[:,:2]).abs().max().detach()),1e-3)

    def test_all_positions_have_gradients(self):
        model=CausalDecoder();x=torch.randn(2,4,8);y=torch.randn(2,4,8)
        out=model(x);out.retain_grad();((out-y)**2).mean().backward()
        self.assertTrue(torch.all(out.grad.abs().sum((0,2))>0))

    def test_attention_rows_and_mask(self):
        x=torch.randn(2,4,3);_,w=attention(x,x,x,True)
        torch.testing.assert_close(w.sum(-1),torch.ones(2,4))
        self.assertEqual(float(w.triu(1).sum()),0.)

    def test_sensor_permutation_and_cardinality(self):
        model=SensorSet();x=torch.randn(3,16,3)
        torch.testing.assert_close(model(x),model(x[:,torch.randperm(16)]),atol=1e-6,rtol=1e-5)
        self.assertEqual(tuple(model(x[:,:8]).shape),(3,8))

    def test_shifted_windows(self):
        z=np.arange(80).reshape(10,8);x,y=sequence_windows(z)
        np.testing.assert_array_equal(x[0],z[:4]);np.testing.assert_array_equal(y[0],z[1:5])

    def test_checkpoint_roundtrip_needs_no_new_basis(self):
        rng=np.random.default_rng(3);rep=Representation.fit(rng.normal(size=(24,18)),8);model=CausalDecoder()
        bundle={'state':model.state_dict(),'architecture':'Causal-Transformer','protocol':Protocol().__dict__,'representation':rep.tensors()}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';torch.save(bundle,path)
            restored,rr=checkpoint_model(torch.load(path,weights_only=True))
        x=torch.randn(2,4,8);torch.testing.assert_close(model(x),restored(x))
        np.testing.assert_array_equal(rr.modes,rep.modes)
        self.assertTrue(np.all(rep.modes[np.argmax(abs(rep.modes),axis=0),np.arange(8)]>0))

    def test_projection_error_decomposition(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(24,18));rep=Representation.fit(a,8)
        row=error_components(rep.decode(rep.encode(a)+.1),a,rep)
        self.assertLess(abs(row['pythagorean_residual']),1e-10)

    def test_label_accounting_and_fixed_initialization(self):
        for budget,total in [(4,44),(8,84),(15,154)]:
            tr,va,record=transfer_indices(budget)
            self.assertEqual(record['total_target_labels'],total)
            self.assertFalse(set(record['training_frames'])&set(record['validation_frames']))
            self.assertEqual(record['initialization_frames'],[156,157,158,159])
            self.assertLess(max(record['validation_frames']),160)

    def test_validation_changes_checkpoint_selection(self):
        # Identical training data; a reversed validation target must change
        # selection, proving the callback is not ignored or replaced by train loss.
        records=[]
        for sign in (1.,-1.):
            model=torch.nn.Linear(1,1,bias=False)
            model.weight.data.zero_()
            x=np.ones((8,1));y=np.ones((8,1))
            record=fit(model,x,y,lambda m:float(((m(torch.ones(8,1))-sign)**2).mean()),
                       replace(Protocol(),max_steps=12,eval_every=1))
            records.append(record)
        self.assertGreater(records[0]['selected_step'],0)
        self.assertEqual(records[1]['selected_step'],0)

    def test_nonfinite_rollout_rejects_checkpoint(self):
        class Bad(torch.nn.Module):
            def next(self,x):return x[:,-1]*float('nan')
        self.assertTrue(np.isinf(safe_rollout_loss(Bad(),np.zeros((4,8)),np.zeros((5,8)))))

    def test_storage_roundtrip(self):
        rng=np.random.default_rng(7);values=rng.normal(size=(18,5))@rng.normal(size=(5,32))+2
        archive={'example__'+k:v for k,v in pack_prediction(values).items()}
        np.testing.assert_allclose(prediction(archive,'example'),values,atol=1e-10)

    def test_frequency_fit_detects_sub_bin_change(self):
        t=np.arange(121)*.1041667
        actual=spectral_fit(t,1.7*np.sin(2*np.pi*.18537*t+.3)+.2)
        self.assertLess(abs(actual['frequency']-.18537),1e-5)
        self.assertLess(abs(actual['amplitude']-1.7),1e-4)

    def test_constant_probe_has_no_identifiable_frequency(self):
        self.assertIsNone(spectral_fit(np.arange(121),np.ones(121))['frequency'])

    def test_initial_phase_uses_common_nonzero_start_time(self):
        t = 137.5 + np.arange(121) * .1041667
        phase = .43
        fits = [spectral_fit(t, np.sin(2*np.pi*f*(t-t[0])+phase)) for f in (.185, .217)]
        for result in fits:
            self.assertLess(abs(result['phase']-phase), 1e-5)
        self.assertLess(abs(fits[0]['phase']-fits[1]['phase']), 1e-5)
        shifted = spectral_fit(t+71.3, np.sin(2*np.pi*.185*(t-t[0])+phase))
        self.assertLess(abs(shifted['phase']-fits[0]['phase']), 1e-5)

    def test_float32_constant_probe_has_no_spurious_spectrum(self):
        for value in (.1, 12345.67, -.037):
            result = spectral_fit(np.arange(121), np.full(121, value, dtype=np.float32))
            self.assertIsNone(result['frequency'])
            self.assertIsNone(result['phase'])
            self.assertIsNone(result['r_squared'])
            self.assertEqual(result['amplitude'], 0.)

    def test_error_reductions_promote_before_subtraction(self):
        rng = np.random.default_rng(42)
        truth = rng.normal(size=(121, 4096)).astype(np.float32)
        # Exact orthonormal basis keeps this a test of reduction precision.
        rep = Representation(np.zeros(4096), np.eye(4096, 8), np.ones(8), .1)
        pred = rng.normal(size=truth.shape).astype(np.float32)
        actual = error_components(pred, truth, rep)
        reference = error_components(pred.astype(np.float64), truth.astype(np.float64), rep)
        self.assertEqual(actual, reference)
        projected = rep.decode(rep.encode(truth))
        self.assertLess(abs(error_components(projected, truth, rep)['pythagorean_residual']), 1e-12)

    def test_sinusoid_r_squared_reports_model_mismatch(self):
        t=27.3+np.arange(301)*.1
        exact=np.sin(2*np.pi*.18537*(t-t[0])+.4)
        pure=spectral_fit(t,exact)
        mixed=spectral_fit(t,exact+.6*np.sin(2*np.pi*.431*(t-t[0])))
        self.assertGreater(pure['r_squared'],1-1e-10)
        self.assertLess(mixed['r_squared'],.9)
        self.assertGreater(mixed['r_squared'],0.)
        self.assertAlmostEqual(mixed['r_squared'],1-mixed['residual_mse']/np.var(exact+.6*np.sin(2*np.pi*.431*(t-t[0]))))

    def test_checkpoint_criteria_preserve_strict_failure(self):
        truth=np.ones((4,8));reference=np.zeros_like(truth)
        close=checkpoint_agreement(reference+2e-6,reference,truth)
        self.assertFalse(close['strict_passed'])
        self.assertFalse(close['passed'])
        self.assertTrue(close['scientific_passed'])
        bad=checkpoint_agreement(reference+2e-4,reference,truth)
        self.assertFalse(bad['scientific_passed'])
        exact=checkpoint_agreement(reference,reference,truth)
        self.assertTrue(exact['strict_passed'] and exact['scientific_passed'])

    def test_actual_experiments_reject_evaluation_leakage(self):
        from flowmllab.transformer_course import load_data,sensor_experiment,forecast_experiment,transfer_experiment
        root=Path(__file__).resolve().parents[1]
        cases,_=load_data(root)
        altered={r:{k:v.copy() for k,v in c.items()} for r,c in cases.items()}
        altered[105]['v']+=50
        for r in (100,105,110):altered[r]['omega'][210:]+=50
        p=replace(Protocol(),seeds=(17,),max_steps=2,eval_every=1,pretrain_steps=2,adaptation_steps=2,transfer_blocks=(4,))
        for experiment in (sensor_experiment,forecast_experiment,transfer_experiment):
            seed_all();a,pa,sa=experiment(cases,p)
            seed_all();b,pb,sb=experiment(altered,p)
            self.assertEqual(list(sa),list(sb))
            # The poisoned retained truth must actually affect scoring while
            # leaving the fitted states unchanged; this detects a vacuous test.
            self.assertNotEqual(a['rows'][0]['metrics']['field_relative_l2'],b['rows'][0]['metrics']['field_relative_l2'])
            for key in sa:
                for parameter in sa[key]['state']:
                    torch.testing.assert_close(sa[key]['state'][parameter],sb[key]['state'][parameter],rtol=0,atol=0)
            # Oracle controls use truth by definition and are not fitted predictors.
            if experiment is not sensor_experiment:
                for key in pa:
                    if key!='POD-oracle':np.testing.assert_array_equal(pa[key],pb[key])

def copy_state(model):return {k:v.detach().clone() for k,v in model.state_dict().items()}

if __name__=='__main__':unittest.main()
