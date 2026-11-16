"""Independent checks of the reduced-BGK reference used by the fresh PINN lab."""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from flowmllab import kinetic_dvm as dvm


def test_maxwellian_moments_and_zero_nonequilibrium():
    z,w = leggauss(80); v,w = 10*z,10*w
    rho,u,T = np.array([.8,1.2]),np.array([.2,-.1]),np.array([.7,1.1])
    f = dvm.moments(dvm.maxwellian_reductions(v,rho,u,T),v,w)
    for name,target in [('rho',rho),('u',u),('T',T)]:
        np.testing.assert_allclose(f[name],target,rtol=1e-12,atol=1e-12)
    for name in ('q','sigma','m','R','Delta','S'):
        assert np.max(np.abs(f[name])) < 1e-11


def test_equilibrium_is_stationary_and_fluxes_balance():
    z,w=leggauss(64); v,w=8*z,8*w
    reservoir=dvm.maxwellian_reductions(v,1.,.15,.8)[0]
    state=np.broadcast_to(reservoir,(32,3,len(v))).copy()
    rhs,boundary,collision=dvm._rhs(state,v,w,1/32,1.,reservoir,reservoir)
    assert np.max(np.abs(rhs)) < 1e-12
    np.testing.assert_allclose(boundary,0,atol=1e-14)
    np.testing.assert_allclose(collision,0,atol=1e-12)


def test_smooth_free_transport_accuracy_improves_with_space_time_refinement():
    errors=[]
    for cells in (80,160):
        s=dvm.solve(dvm.DVMConfig(cells=cells,velocity_order=32,final_time=.003,tau=1e12))
        x,v,w=s['x'],s['v'],s['weights']
        back=x[:,None]-v[None,:]*.003
        rho,u,T=dvm.initial_macros(back)
        g=rho*np.exp(-v[None,:]**2/(2*T))/np.sqrt(2*np.pi*T)
        exact=np.sum(g*w,axis=-1)
        mask=np.abs(x)<.3
        errors.append(np.sqrt(np.mean((s['rho'][mask]-exact[mask])**2)))
    assert errors[1] < .6*errors[0]


def test_transient_balance_and_finite_positive_macros():
    s=dvm.solve(dvm.DVMConfig(cells=100,velocity_order=48,final_time=.005))
    d=s['diagnostics']
    assert d['minimum_rho']>0 and d['minimum_T']>0
    assert min(d['minimum_g_h_k_over_stages'])>=0
    assert np.max(np.abs(d['mass_momentum_energy_balance_defect']))<1e-10
    assert np.max(np.abs(d['algebraic_balance_defect_after_collision_quadrature']))<1e-12
    assert np.isfinite(s['fields']).all()
    assert d['clipping_applied'] is False


@pytest.mark.parametrize('config',[dvm.DVMConfig(cells=0),dvm.DVMConfig(cfl=.8),
                                 dvm.DVMConfig(tau=0),dvm.DVMConfig(vmax=float('nan'))])
def test_invalid_controls_fail_before_solving(config):
    with pytest.raises(ValueError): dvm.solve(config)
