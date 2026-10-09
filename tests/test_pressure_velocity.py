"""Independent conservation and algorithm-limit checks for Week 1.2."""
import unittest
from dataclasses import replace
import numpy as np
from flowmllab.pressure_velocity import (
    CavityConfig, divergence, pressure_matrix, correct_velocity,
    coupling_step, _pressure_solver, _residual, run_cavity,
)


class PressureVelocityTests(unittest.TestCase):
    def test_face_flux_cancellation_and_pressure_nullspace(self):
        rng=np.random.default_rng(91); n=8; h=1/n
        u=rng.normal(size=(n,n+1)); v=rng.normal(size=(n+1,n))
        u[:,[0,-1]]=0; v[[0,-1],:]=0
        self.assertLess(abs(divergence(u,v,h).sum()),1e-12)
        du=rng.uniform(.1,1,(n,n-1)); dv=rng.uniform(.1,1,(n-1,n))
        matrix=pressure_matrix(du,dv,h)
        np.testing.assert_allclose(matrix@np.ones(n*n),0,atol=1e-12)
        p=_pressure_solver(du,dv,h)(-divergence(u,v,h))
        un,vn=correct_velocity(u,v,p,du,dv,h)
        shifted_u,shifted_v=correct_velocity(u,v,p+3.7,du,dv,h)
        np.testing.assert_allclose(shifted_u,un,atol=1e-12,rtol=0)
        np.testing.assert_allclose(shifted_v,vn,atol=1e-12,rtol=0)
        self.assertLess(np.abs(divergence(un,vn,h)).max(),1e-11)
        self.assertLess(abs(p.mean()),1e-14)

    def test_pimple_one_outer_is_piso(self):
        n=8; u=np.zeros((n,n+1)); v=np.zeros((n+1,n)); p=np.zeros((n,n))
        cfg=CavityConfig(method='piso',cells=n)
        a=coupling_step(u.copy(),v.copy(),p.copy(),cfg)
        b=coupling_step(u.copy(),v.copy(),p.copy(),replace(cfg,method='pimple',outer_correctors=1))
        for lhs,rhs in zip(a[:3],b[:3]): np.testing.assert_allclose(lhs,rhs,rtol=0,atol=0)

    def test_extra_piso_correction_reduces_momentum_defect(self):
        n=8; u=np.zeros((n,n+1)); v=np.zeros((n+1,n)); p=np.zeros((n,n))
        cfg=CavityConfig(method='piso',cells=n,dt=.02)
        errors=[]
        for correctors in [1,2,3]:
            un,vn,pn,_=coupling_step(u.copy(),v.copy(),p.copy(),replace(cfg,pressure_correctors=correctors))
            errors.append(_residual(un,vn,pn,cfg,u,v,cfg.dt))
        self.assertGreater(errors[0],errors[1]); self.assertGreater(errors[1],errors[2])

    def test_independent_algorithms_reach_same_discrete_steady_solution(self):
        outputs=[run_cavity(CavityConfig(method=m,cells=8,tolerance=2e-6,max_iterations=2000))
                 for m in ['simple','piso','pimple']]
        self.assertTrue(all(q['converged'] for q in outputs))
        for q in outputs[1:]:
            for key in ['u_faces','v_faces','p']:
                np.testing.assert_allclose(q[key],outputs[0][key],atol=1e-5,rtol=0)

    def test_transient_stopping_time_is_not_steady_convergence(self):
        q=run_cavity(CavityConfig(method='piso',cells=8,end_time=.12,dt=.05))
        self.assertTrue(q['reached_end_time']); self.assertFalse(q['converged'])
        self.assertAlmostEqual(q['physical_time'],.12)
        self.assertEqual(q['iterations'],3)


if __name__=='__main__': unittest.main()
