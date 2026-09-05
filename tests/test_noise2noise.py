import unittest
import warnings
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from flowmllab.noise2noise import (independent_targets,split_indices,patch_features,
    spectral_gain,spectral_predict,restore_mean,noisy_validation_score,
    fit_experiment,predict_experiment,audit_errors)


class Noise2NoiseTests(unittest.TestCase):
    def setUp(self):
        self.a=np.random.default_rng(4).normal(size=(4,2,12,12))

    def test_target_excludes_own_seed(self):
        target=independent_targets(self.a)
        changed=self.a.copy();changed[0]+=100
        np.testing.assert_array_equal(target[0],independent_targets(changed)[0])
        np.testing.assert_allclose(target[1],self.a[[0,2,3]].mean(axis=0))
        with self.assertRaises(ValueError):
            independent_targets(self.a[:1])

    def test_group_leakage_rejected(self):
        split={'train':[0,1],'validation':[2,3],'test':[4,5]}
        self.assertEqual(split_indices(range(6),split)['test'].tolist(),[4,5])
        with self.assertRaises(ValueError):
            split_indices(range(6),{**split,'test':[3,5]})

    def test_patch_order(self):
        x=np.arange(2*12*12).reshape(2,12,12)
        p=patch_features(x,3).reshape(12,12,2,3,3)
        np.testing.assert_array_equal(p[5,6],x[:,4:7,5:8])
        with self.assertRaises(ValueError):
            patch_features(x,4)

    def test_spectral_bounds_and_identity(self):
        gain=spectral_gain(self.a)
        self.assertTrue(np.all((gain>=0)&(gain<=1)))
        np.testing.assert_array_equal(gain[:,0,0],1)
        np.testing.assert_allclose(spectral_predict(self.a[0],np.ones_like(gain)),self.a[0],atol=1e-12)
        np.testing.assert_allclose(spectral_predict(self.a[0],gain).mean(axis=(-2,-1)),self.a[0].mean(axis=(-2,-1)),atol=1e-12)

    def test_mean_and_error_audit(self):
        result=restore_mean(self.a[0],self.a[1])
        np.testing.assert_allclose(result.mean(axis=(-2,-1)),self.a[1].mean(axis=(-2,-1)),atol=1e-12)
        for row in audit_errors(self.a[0],self.a[0],self.a[0]):
            self.assertTrue(all(v==0 for v in row.values()))
        with self.assertRaises(ValueError):
            audit_errors(self.a[0],np.zeros_like(self.a[0]),self.a[0])

    def test_noisy_validation_formula(self):
        score=noisy_validation_score(self.a[:2],self.a[2:],np.ones(2))
        expected=np.mean((self.a[:2]-self.a[[3,2]])**2)
        self.assertAlmostEqual(score,expected)

    def test_fit_predict_needs_no_reference(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            fitted=fit_experiment(self.a[:2],self.a[2:],max_iter=2,samples_per_seed=20)
        preds=predict_experiment(fitted,self.a[0])
        self.assertEqual(len(preds),6)
        self.assertTrue(all(np.isfinite(v).all() and v.shape==self.a[0].shape for v in preds.values()))
        self.assertIn(fitted['gaussian_width'],(.5,1.,1.5,2.))

    def test_retained_data_and_scores(self):
        root=Path(__file__).resolve().parents[1]
        data=root/'data/week12_noise2noise'
        result=root/'results/week12_noise2noise'
        for folder,name in [(data,'manifest.json'),(result,'run_manifest.json')]:
            meta=json.loads((folder/name).read_text())
            for file,sha in meta['files'].items():
                self.assertEqual(hashlib.sha256((folder/file).read_bytes()).hexdigest(),sha)
        split=json.loads((data/'manifest.json').read_text())['split']
        with np.load(data/'observations.npz') as obs, np.load(data/'evaluation_only.npz') as ref, np.load(result/'test_predictions.npz') as preds:
            split_indices(obs['seeds'],split)
            with (result/'metrics.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),28)
            self.assertEqual({int(r['seed']) for r in rows},set(split['test']))
            for r in rows:
                i=list(obs['seeds']).index(int(r['seed']))
                k=['qx','qy'].index(r['field'])
                calculated=audit_errors(preds[r['seed']+'_'+r['method']],ref['reference'],obs['raw3'][i])[k]
                for key,value in calculated.items():
                    self.assertAlmostEqual(value,float(r[key]),places=10)


if __name__=='__main__':
    unittest.main()
