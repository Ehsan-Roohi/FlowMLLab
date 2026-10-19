"""Independent bounded scientific regression checks; never rewrites retained data."""
import ast
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'common')]
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import matplotlib.pyplot as plt
from flowmllab import transformer_course as lab
from flowmllab import hypersonic_cylinder as hyp
from flowmllab.aescte_dsmc import linear_profile_prediction
import student_experiment_protocols as contracts
import run_pod_deeponet_validation as deeponet
from mini_dsmc import MiniDSMCConfig


def lesson(week):
    return json.loads((ROOT / f'course/lessons/week{week}.json').read_text())


class ScientificChecks(unittest.TestCase):
    def test_fresh_operator_real_fit_and_perturbation(self):
        data = deeponet.load_data()
        train = np.flatnonzero(data['split'] == 'train')
        chosen = {key: {'rank': 2, 'hidden': [2], 'input_transform': 'linear'}
                  for key in ('velocity', 'pressure')}
        trunks = {key: deeponet.make_trunk(data, train, 2, key) for key in chosen}
        bundle = deeponet.fit_operator(data, train, trunks, chosen, 177)
        with tempfile.TemporaryDirectory() as directory:
            a, predictions, protocol = deeponet.evaluate_fresh_operator(data, [bundle], chosen, Path(directory)/'first')
            damaged = copy.deepcopy(bundle)
            damaged['velocity']['model'].intercepts_[-1] += 5
            damaged['pressure']['model'].intercepts_[-1] += 7
            b, altered, _ = deeponet.evaluate_fresh_operator(data, [damaged], chosen, Path(directory)/'second')
            self.assertFalse(np.allclose(a.relative_L2_uv, b.relative_L2_uv))
            self.assertFalse(np.allclose(a.relative_L2_p, b.relative_L2_p))
            loaded = np.load(Path(directory)/'first/predictions.npz')
            for index, re in enumerate(loaded['Re']):
                np.testing.assert_array_equal(loaded['u'][index], predictions[re][0])
                expected = deeponet.field_report(data, re, predictions[re])
                self.assertAlmostEqual(a.iloc[index].relative_L2_uv, expected['relative_L2_uv'])
            self.assertEqual(protocol['dataset_sha256'], contracts.dataset_fingerprint(data))
            with self.assertRaises(FileExistsError):
                deeponet.evaluate_fresh_operator(data, [bundle], chosen, Path(directory)/'first')
        with self.assertRaises(ValueError):
            deeponet.evaluate_fresh_operator(data, [bundle], chosen, ROOT/'results/agent7-forbidden')
        print('Fresh real rank-2 fit UV baseline / perturbed:', a.relative_L2_uv.tolist(), b.relative_L2_uv.tolist())

    def test_cache_contract_changes_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); solver = folder/'solver.py'; solver.write_text('version=1\n')
            configs = [MiniDSMCConfig(Kn=.1, seed=2), MiniDSMCConfig(Kn=.2, seed=3)]
            request = contracts.kinetic_request(configs, solver, '1')
            cases = []
            for config in configs:
                case = dict(config=dataclasses.asdict(config), x=np.arange(3.), y=np.arange(3.))
                for name in ('u','v','T','rho'): case[name] = np.ones((3,3))*config.Kn
                for name in ('field_samples','sampling_start_time','sampling_end_time','sampling_window_time',
                             'mean_particle_observations_per_cell','runtime_s'): case[name] = 1
                case['temperature_estimator'] = request['temperature_estimator']; cases.append(case)
            path = folder/'cache.npz'; contracts.save_kinetic_cache(path, cases, request)
            result, _ = contracts.load_kinetic_cache(path, request)
            np.testing.assert_array_equal(result['u'][1], cases[1]['u'])
            changes = {'dt': .004, 'T_wall': 2., 'steps': 3300, 'sample_stride': 4, 'sample_start': 2300, 'ppc': 7}
            for key, value in changes.items():
                changed = [dataclasses.replace(configs[0], **{key:value}), configs[1]]
                with self.assertRaises(ValueError): contracts.load_kinetic_cache(path, contracts.kinetic_request(changed,solver,'1'))
            solver.write_text('version=2\n')
            with self.assertRaises(ValueError): contracts.load_kinetic_cache(path, contracts.kinetic_request(configs,solver,'1'))
            with self.assertRaises(ValueError): contracts.save_kinetic_cache(folder/'bad.npz', list(reversed(cases)), request)
            with self.assertRaises(FileExistsError): contracts.save_kinetic_cache(path, cases, request)
            np.savez(folder/'legacy.npz', u=result['u'])
            with self.assertRaises(ValueError): contracts.load_kinetic_cache(folder/'legacy.npz',request)
        original = {'a':np.array([1.,2.]), 'b':np.array(['train','test'])}
        changed = copy.deepcopy(original); changed['a'][1] += .1
        self.assertNotEqual(contracts.dataset_fingerprint(original),contracts.dataset_fingerprint(changed))

    def test_rank_selection_and_interpolation(self):
        frame = pd.DataFrame({'rank':[1,2,3], 'relative_L2_uv':[.01,.1,.095],
                              'relative_L2_p':[4.,.02,.01], 'wall':[0.,0.,0.]})
        self.assertEqual(contracts.select_physical_rank(frame, {'relative_L2_p': .1})['rank'], 2)
        with self.assertRaises(ValueError): contracts.select_physical_rank(frame, {'relative_L2_p': .001})
        np.testing.assert_allclose(contracts.interpolate_coefficients([3,1,2],np.array([[6,9],[2,3],[4,6]]),1.5),[3,4.5])
        with self.assertRaises(ValueError): contracts.interpolate_coefficients([1,2],np.ones((2,2)),3)

    def test_notebook_sample_variance_and_historical_bias_control(self):
        notebook=json.loads((ROOT/'notebooks/week05_06/P4_Uncertainty_Study.ipynb').read_text())
        assignments=[]
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                for node in ast.walk(ast.parse(''.join(cell['source']))):
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='spread' for t in node.targets):assignments.append(node)
        self.assertEqual(len(assignments),1)
        namespace=dict(np=np,stack_u=np.array([0.,2.,4.])[:,None,None],stack_v=np.array([1.,4.,7.])[:,None,None])
        exec(compile(ast.Module(body=assignments,type_ignores=[]),'p4spread','exec'),namespace)
        self.assertAlmostEqual(float(namespace['spread'][0,0]),np.sqrt(13))
        notebook=json.loads((ROOT/'notebooks/week03/AI_in_Fluids_Week3_Lab1_Maxwellian_Noise_ML_Student.ipynb').read_text())
        function=next(node for cell in notebook['cells'] if cell['cell_type']=='code' for node in ast.parse(''.join(cell['source'])).body
                      if isinstance(node,ast.FunctionDef) and node.name=='estimate_macroscopic_quantities')
        namespace={'np':np,'M_ARGON':1.,'K_B':1.}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'temperature','exec'),namespace)
        _,temperature,_,_=namespace['estimate_macroscopic_quantities'](np.array([[-1.,0,0],[1.,0,0]]),1.)
        self.assertAlmostEqual(temperature,1/3)
        self.assertAlmostEqual(temperature*2,2/3)

    def test_mask_exactness_and_duplicate_failure(self):
        data = hyp.load_cylinder_teaching_data(ROOT)
        pick = np.isclose(data.mach_inf,8.5)
        x,y,v=data.x[pick],data.y[pick],data.targets[pick,0]
        xs,ys,grid=hyp.masked_structured_field(x,y,v)
        self.assertEqual(int(grid.mask.sum()),275)
        np.testing.assert_array_equal(grid[np.searchsorted(ys,y),np.searchsorted(xs,x)],v)
        with self.assertRaises(ValueError): hyp.masked_structured_field([1,1],[2,2],[3,4])
        with self.assertRaises(ValueError): hyp.masked_structured_field([np.nan],[2],[3])
        print('Cylinder exact retained lattice:',grid.shape,'masked:',grid.mask.sum(),'fluid:',v.size)

    def test_fusion_actual_forward_rank(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(714)
        rng=np.random.default_rng(714)
        inputs=[rng.uniform(.5,2,(80,1)).astype('float32'),rng.normal(size=(80,2)).astype('float32')]
        vector=hyp.build_fusion_deeponet(latent_dim=8,hidden_layers=2,dropout_rate=0)
        with warnings.catch_warnings(record=True) as captured:
            legacy=hyp.build_fusion_deeponet(latent_dim=8,hidden_layers=2,dropout_rate=0,output_mode='legacy_scalar')
        self.assertTrue(any('affine line' in str(item.message) for item in captured))
        ranks=[]
        for model in (vector,legacy):
            values=model(inputs,training=False).numpy();self.assertEqual(values.shape,(80,3));self.assertTrue(np.isfinite(values).all())
            sv=np.linalg.svd(values-values.mean(0),compute_uv=False)
            ranks.append(int(np.sum(sv>sv[0]*1e-5)))
        self.assertEqual(ranks,[3,1])
        with self.assertRaises(ValueError): hyp.build_fusion_deeponet(output_mode='wrong')
        print('Actual TensorFlow vector / legacy centered output rank:',ranks)

    def test_linear_profile_numerical_oracle(self):
        p=np.array([4.,1.,2.]); profiles=np.stack([np.array([2*x+3,-x+4]) for x in p])
        for query in (-2.,1.,1.5,4.,7.):
            np.testing.assert_allclose(linear_profile_prediction(p,profiles,query),[2*query+3,-query+4])
        for invalid in ([1.,1.,2.],[1.,np.nan,2.]):
            with self.assertRaises(ValueError): linear_profile_prediction(invalid,profiles,1.5)

    def test_qkv_shapes_and_large_score_stability(self):
        namespace={'torch':torch,'np':np}
        for task in lesson(17)['tasks'][:3]: exec(task['solution'],namespace)
        rng=np.random.default_rng(17)
        for shape in ((7,3),(2,7,3)):
            q=torch.tensor(rng.normal(size=shape),dtype=torch.float64)*100
            k=torch.tensor(rng.normal(size=shape),dtype=torch.float64)*100
            v=torch.tensor(rng.normal(size=shape[:-1]+(5,)),dtype=torch.float64)
            scores=(q.numpy()@np.swapaxes(k.numpy(),-1,-2))/np.sqrt(3)
            weights=np.exp(scores-scores.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
            actual=namespace['student_attention'](q,k,v)
            np.testing.assert_allclose(actual.numpy(),weights@v.numpy(),atol=1e-12)
            self.assertTrue(torch.isfinite(actual).all())
        for temperature in (0,-1,np.nan,np.inf):
            with self.assertRaises(ValueError): namespace['temperature_weights'](q,k,temperature)

    def test_ce_all_tokens_batch_invariance_and_suffix(self):
        namespace={'torch':torch}
        tree=ast.parse(lesson(18)['demos'][0]['code'])
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='full_validation_ce')
        exec(compile(ast.Module(body=[function],type_ignores=[]),'lesson18','exec'),namespace)
        class Model(torch.nn.Module):
            def forward(self,x):return torch.nn.functional.one_hot(x,num_classes=3).double()*4
        x=torch.arange(83*5).reshape(83,5)%3;y=x.clone();y[64:]=(y[64:]+1)%3
        expected=float(torch.nn.functional.cross_entropy(Model()(x).reshape(-1,3),y.reshape(-1)))
        scores=[namespace['full_validation_ce'](Model(),x,y,n) for n in (1,17,64,200)]
        for score,tokens in scores:self.assertAlmostEqual(score,expected,places=12);self.assertEqual(tokens,415)
        clean=namespace['full_validation_ce'](Model(),x,x)[0]
        self.assertGreater(expected-clean,.9)
        print('All-window CE:',expected,'clean-prefix misleading CE:',clean,'tokens:',415)

    def test_rollout_no_future_truth_and_strict_checkpoint_status(self):
        namespace={'np':np}
        exec(lesson(21)['tasks'][3]['solution'],namespace)
        initial=np.ones((4,2));truth=np.ones((121,2));poison=truth.copy();poison[50:]*=1000
        predict=lambda history: history[-1]*1.01
        a=namespace['rollout_audit'](predict,initial,truth)
        b=namespace['rollout_audit'](predict,initial,poison)
        np.testing.assert_array_equal(a['predictions'],b['predictions'])
        np.testing.assert_allclose(a['predictions'][:,0],1.01**np.arange(1,122))
        np.testing.assert_array_equal(initial,np.ones((4,2)))
        strict=lab.checkpoint_agreement(np.full((2,2),2e-6),np.zeros((2,2)),np.ones((2,2)))
        self.assertFalse(strict['strict_passed']);self.assertFalse(strict['passed']);self.assertTrue(strict['scientific_passed'])
        print('Checkpoint deliberately strict-fail/scientific-pass:',strict)

    def test_retained_forecast_current_inference_and_demos(self):
        evidence=ROOT/'results/transformer_course_v3'
        cases,_=lab.load_data(ROOT)
        record=json.loads((evidence/'forecast.json').read_text())
        row=next(r for r in record['rows'] if r['method']=='Causal-Transformer' and r['seed']==17)
        bundles=torch.load(evidence/'checkpoints.pt',map_location='cpu',weights_only=True)
        model,rep=lab.checkpoint_model(bundles['forecast__'+row['key']])
        encoded=rep.encode(cases[110]['omega'][156:160])
        current=rep.decode(lab.rollout(model,encoded,121))
        retained=np.load(evidence/'predictions.npz',allow_pickle=False)
        expected=lab.prediction(retained,'forecast__'+row['key'])
        truth=cases[110]['omega'][160:281].reshape(121,-1)
        agreement=lab.checkpoint_agreement(current,expected,truth)
        self.assertTrue(agreement['scientific_passed'])
        np.testing.assert_allclose(rep.modes,bundles['forecast__'+row['key']]['representation']['modes'].numpy(),rtol=0,atol=0)
        print('Current saved checkpoint 121-step forecast agreement:',agreement)
        namespace=dict(ROOT=ROOT,EVIDENCE=evidence,cases=cases,lab=lab,np=np,pd=pd,torch=torch,plt=plt,json=json,display=lambda value:None)
        exec(lesson(17)['demos'][2]['code'],namespace)
        self.assertEqual(tuple(namespace['trained_weights'].shape),(3,4,16,16))
        self.assertEqual(namespace['frames'],[80,120,160])
        plt.close('all')
        with patch.object(lab.Representation,'fit',side_effect=AssertionError('forbidden basis refit')):
            for demo in lesson(21)['demos']:exec(demo['code'],namespace)
        axis=plt.figure(plt.get_fignums()[-1]).axes[0]
        trajectories=[line for line in axis.lines if len(line.get_xdata())==121]
        self.assertGreaterEqual(len(trajectories),4)
        self.assertTrue(all(line.get_xdata()[0]==1 and line.get_xdata()[-1]==121 for line in trajectories))
        plt.close('all')
        for demo in lesson(22)['demos']:exec(demo['code'],namespace)
        rows=namespace['rows']
        for target in (100,105,110):
            floors={n:np.mean([r['metrics']['representation_floor'] for r in rows if r['arm']=='target-POD-MLP' and r['target_re']==target and r['labels']['total_target_labels']==n]) for n in (44,84,154)}
            ax=namespace['axes'][0,list((100,105,110)).index(target)]
            curve=next(line for line in ax.lines if line.get_label()=='target-POD floor (target-basis arm)')
            np.testing.assert_allclose(curve.get_ydata(),list(floors.values()))
            self.assertGreater(np.ptp(list(floors.values())),0)
        resource=pd.DataFrame(namespace['resources'])
        neural=resource[resource.arm!='DMD']
        self.assertTrue(neural.seconds_adaptation.notna().all())
        self.assertTrue(neural.parameters.notna().all())
        plt.close('all')


if __name__=='__main__':
    unittest.main(verbosity=2)
