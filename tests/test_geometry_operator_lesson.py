"""Scientific guards for the bounded manufactured operator lesson."""
import unittest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
from flowmllab import geometry_operator_lesson as lab

class GeometryOperatorTests(unittest.TestCase):
    def test_analytic_reference_is_streamfunction_derivative(self):
        a=.36;x=.31;y=.62;h=1e-6
        def psi(x,y):
            b=a*np.sin(np.pi*x)**2;e=(y-b)/(1-b)
            return e*e*(3-2*e)
        b=a*np.sin(np.pi*x)**2;e=(y-b)/(1-b)
        u=6*e*(1-e)/(1-b);v=u*a*np.pi*np.sin(2*np.pi*x)*(1-e)
        self.assertAlmostEqual(u,(psi(x,y+h)-psi(x,y-h))/(2*h),places=8)
        self.assertAlmostEqual(v,-(psi(x+h,y)-psi(x-h,y))/(2*h),places=8)
        field=lab.reference(a)['field']
        np.testing.assert_array_equal(field[[0,-1]],0)

    def test_full_field_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(2);shapes=[(1,3),(3,2)]
        theta=rng.normal(size=14)*.2;inputs=rng.normal(size=(4,1))
        truth=rng.normal(size=(4,7));mean=rng.normal(size=7)
        modes=rng.normal(size=(2,7));scale=np.array([.4,1.7]);weights=np.arange(1,8.)
        def objective(t):return lab.loss_gradient(t,shapes,inputs,truth,mean,modes,scale,weights)
        _,gradient=objective(theta);finite=np.zeros_like(theta)
        for i in range(len(theta)):
            delta=np.zeros_like(theta);delta[i]=1e-6
            finite[i]=(objective(theta+delta)[0]-objective(theta-delta)[0])/2e-6
        np.testing.assert_allclose(gradient,finite,rtol=1e-5,atol=1e-8)

    def test_partition_and_grid_guards(self):
        cfg=lab.Config(depth=1,width=3,rank=1,adam_steps=0,lbfgs_iterations=0,optimizer='adam',nx=5,ny=5)
        bad=[dict(train=[.2,.1,.3]),dict(train=[.1,.1,.3]),dict(train=[]),dict(validation=[]),dict(validation=[np.nan]),dict(validation=[.08]),dict(validation=[.7])]
        for kw in bad:
            with self.subTest(kw=kw),self.assertRaises(ValueError):lab.fit(cfg,**kw)
        with self.assertRaises(ValueError):lab.fit(lab.Config(nx=2))
        model=lab.fit(cfg)
        with self.assertRaises(ValueError):lab.evaluate(model,[.12])
        with self.assertRaises(ValueError):lab.predict(model,[np.nan])
        for result in lab.evaluate(model):
            self.assertTrue(all(np.isfinite(v) for v in result['scores'].values()))
            self.assertLessEqual(result['scores']['POD oracle'],result['scores']['POD-DeepONet']+1e-12)

    def test_colab_existing_sparse_checkout_adds_new_lesson_dependencies(self):
        from qa.late_course_support import bootstrap_source
        temporary_root=Path(__file__).resolve().parents[1]/'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root=Path(folder);(root/'flowmllab').mkdir();(root/'.git').mkdir()
            (root/'flowmllab/marker.py').write_text('')
            code=bootstrap_source('flowmllab/marker.py',['flowmllab','data/new-lesson'])
            def fake_git(args,**kwargs):
                if 'sparse-checkout' in args:(root/'data/new-lesson').mkdir(parents=True)
                return subprocess.CompletedProcess([],0,'true\n','')
            with patch('pathlib.Path.cwd',return_value=root),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run',side_effect=fake_git) as run:
                namespace={};exec(code,namespace)
            self.assertEqual(run.call_count,2)
            self.assertEqual(run.call_args.args[0],['git','-C',str(root),'sparse-checkout','add','flowmllab','data/new-lesson'])

    def test_colab_complete_portable_bundle_needs_no_git(self):
        from qa.late_course_support import bootstrap_source
        temporary_root=Path(__file__).resolve().parents[1]/'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root=Path(folder);(root/'flowmllab').mkdir();(root/'data').mkdir()
            (root/'flowmllab/marker.py').write_text('')
            code=bootstrap_source('flowmllab/marker.py',['flowmllab','data'])
            with patch('pathlib.Path.cwd',return_value=root),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
                namespace={};exec(code,namespace)
            self.assertEqual(namespace['ROOT'],root.resolve())
            run.assert_not_called()

            (root/'.git').mkdir()
            with patch('pathlib.Path.cwd',return_value=root),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
                namespace={};exec(code,namespace)
            self.assertEqual(namespace['ROOT'],root.resolve())
            run.assert_not_called()

    def test_colab_incomplete_non_git_folder_is_rejected_without_mutation(self):
        from qa.late_course_support import bootstrap_source
        temporary_root=Path(__file__).resolve().parents[1]/'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root=Path(folder);(root/'flowmllab').mkdir()
            (root/'flowmllab/marker.py').write_text('')
            code=bootstrap_source('flowmllab/marker.py',['flowmllab','data'])
            with patch('pathlib.Path.cwd',return_value=root),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):exec(code,{})
            run.assert_not_called()

    def test_older_full_checkout_requires_actual_plotting_module(self):
        from qa.late_course_support import bootstrap_source
        temporary_root=Path(__file__).resolve().parents[1]/'tmp'
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as folder:
            root=Path(folder);(root/'flowmllab').mkdir();(root/'.git').mkdir()
            (root/'flowmllab/transformer_course.py').write_text('')
            code=bootstrap_source('flowmllab/transformer_course.py',['flowmllab'])
            with patch('pathlib.Path.cwd',return_value=root),patch('importlib.util.find_spec',return_value=object()),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,'','')) as run:
                with self.assertRaisesRegex(RuntimeError,'missing required lesson files'):exec(code,{})
            self.assertEqual(run.call_count,1)
            self.assertIn('config',run.call_args.args[0])

if __name__=='__main__':unittest.main()
