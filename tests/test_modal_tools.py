import unittest
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np
from flowmllab.modal_tools import (fit_pod,project_pod,reconstruct_pod,fit_dmd,
    rollout_dmd,dmd_modes,select_sensors,reconstruct_sensors,fit_integral_sindy,rollout_sindy)
from flowmllab.field_metrics import field_metrics,relative_l2,temporal_spectrum


class ModalTests(unittest.TestCase):
    def test_dmd_orientation_and_frequency(self):
        dt=.02; t=np.arange(400)*dt
        z=np.column_stack((np.cos(2*t),np.sin(2*t)))
        a=fit_dmd(z[:200])
        np.testing.assert_allclose(rollout_dmd(a,z[199],200),z[200:],atol=1e-12)
        self.assertAlmostEqual(max(m['frequency'] for m in dmd_modes(a,dt)),1/np.pi,places=10)
        self.assertLess(max(abs(m['growth_rate']) for m in dmd_modes(a,dt)),1e-10)

    def test_sensors_exact_low_rank_and_oversampling(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(50,3))@rng.normal(size=(3,40))+rng.normal(size=40)
        pod=fit_pod(x,rank=3)
        for k in (3,6,12):
            ids=select_sensors(pod.modes,k)
            self.assertEqual(len(set(ids)),k)
            np.testing.assert_allclose(reconstruct_sensors(pod,ids,x[:,ids]),x,atol=1e-12)
        np.testing.assert_array_equal(select_sensors(pod.modes,6)[:3],select_sensors(pod.modes,3))
        with self.assertRaises(ValueError): select_sensors(pod.modes,2)
        with self.assertRaises(ValueError): reconstruct_sensors(pod,np.array([1,1,2]),x[:,:3])

    def test_integral_sindy_oscillator(self):
        dt=.01;t=np.arange(800)*dt;z=np.column_stack((np.cos(t),np.sin(t)))
        model=fit_integral_sindy(z[:600],dt,threshold=.1,degree=1)
        self.assertEqual(np.count_nonzero(model.coefficients),2)
        np.testing.assert_allclose(model.derivative(z),np.column_stack((-z[:,1],z[:,0])),atol=2e-5)
        np.testing.assert_allclose(rollout_sindy(model,z[599],dt,200),z[600:],atol=3e-5)

    def test_pod_training_only(self):
        rng=np.random.default_rng(4);train=rng.normal(size=(20,12))
        pod=fit_pod(train,rank=4);mean=pod.mean.copy();modes=pod.modes.copy()
        test=rng.normal(size=(10,12))+100
        reconstruct_pod(pod,project_pod(pod,test))
        np.testing.assert_array_equal(pod.mean,mean)
        np.testing.assert_array_equal(pod.modes,modes)

    def test_rollout_input_and_divergence(self):
        with self.assertRaises(ValueError): rollout_dmd(np.eye(2),1,2)
        with self.assertRaises(ValueError): rollout_dmd(np.eye(2)*1e13,[1,1],2)
        with self.assertRaises(ValueError): fit_dmd([[1],[np.nan],[2]])
        with self.assertRaises(ValueError): fit_integral_sindy(np.ones((20,2)),0)


class MetricTests(unittest.TestCase):
    def test_weighted_offset(self):
        truth=np.ones((3,4,5))*2;estimate=truth+1;w=np.ones((4,5))*.25
        m=field_metrics(estimate,truth,w,np.ones((4,5),bool))
        self.assertAlmostEqual(m['relative_l2'],.5)
        self.assertAlmostEqual(m['rmse'],1)
        self.assertAlmostEqual(m['mean_absolute_scalar_integral_error'],5)
        self.assertAlmostEqual(m['edge_relative_l2'],.5)
        self.assertEqual(field_metrics(truth,truth)['maximum_absolute_error'],0)
        self.assertAlmostEqual(field_metrics(estimate,truth,10*w)['relative_l2'],m['relative_l2'])

    def test_zero_reference_and_masked_points(self):
        self.assertIsNone(relative_l2(np.ones(3),np.zeros(3)))
        m=field_metrics(np.ones((2,3,4)),np.zeros((2,3,4)))
        self.assertIsNone(m['worst_frame_relative_l2']);self.assertEqual(m['zero_reference_frames'],2)
        truth=np.ones((2,3,4));a=truth.copy();a[:,0,0]=1e6;w=np.ones((3,4));w[0,0]=0
        self.assertEqual(field_metrics(a,truth,w)['maximum_absolute_error'],0)

    def test_invalid_metrics(self):
        z=np.ones((2,3,4))
        for weights in (np.ones((2,2)),-np.ones((3,4)),np.zeros((3,4))):
            with self.assertRaises(ValueError):field_metrics(z,z,weights)
        with self.assertRaises(ValueError):relative_l2([1,np.nan],[1,2])
        with self.assertRaises(ValueError):field_metrics(z,z,edge_mask=np.zeros((3,4),bool))

    def test_spectrum_resolution_phase(self):
        dt=.01;t=np.arange(1000)*dt;truth=np.sin(2*np.pi*3*t)
        m=temporal_spectrum(np.sin(2*np.pi*3*t+.3),truth,dt)
        self.assertAlmostEqual(m['reference_peak'],3)
        self.assertAlmostEqual(m['frequency_resolution'],.1)
        self.assertAlmostEqual(m['phase_error_at_reference_peak'],.3,places=5)
        self.assertIsNone(temporal_spectrum(np.ones(32),np.ones(32),dt)['reference_peak'])
        with self.assertRaises(ValueError):temporal_spectrum(truth,truth,dt,(0,60))


class ProtocolTests(unittest.TestCase):
    def test_sealed_test_fields_cannot_change_selection_or_forecast(self):
        from flowmllab.modal_experiments import load_cases,forecast_experiment,sensor_experiment
        cases=load_cases(Path(__file__).resolve().parents[1]/'data/modal_labs')
        with threadpool_limits(limits=1):
            f,p=forecast_experiment(cases);s,_=sensor_experiment(cases)
            cases[110]['omega'][210:]+=1000  # Only the sealed forecast truth changes.
            cases[105]['v']+=1000            # Only the sealed sensing truth changes.
            f2,p2=forecast_experiment(cases);s2,_=sensor_experiment(cases)
        self.assertEqual(f['selected_dmd'],f2['selected_dmd'])
        self.assertEqual(f['selected_sindy'],f2['selected_sindy'])
        self.assertEqual(s['selected_budget'],s2['selected_budget'])
        self.assertEqual(s['positions'],s2['positions'])
        self.assertEqual(s['validation_means'],s2['validation_means'])
        for name in p:
            if 'oracle' not in name:
                np.testing.assert_array_equal(p[name],p2[name])
        self.assertNotEqual(f['methods']['DMD-r8']['test']['relative_l2'],f2['methods']['DMD-r8']['test']['relative_l2'])


if __name__=='__main__':unittest.main()
