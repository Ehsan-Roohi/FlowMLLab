"""Verify the actual steady cavity, including equations omitted for the pressure gauge."""
import numpy as np
from numpy.polynomial import Polynomial
from flowmllab.high_order_cavity import (
    nodal_derivative, cavity_operators, cavity_residual, cavity_jacobian,
    solve_spectral_cavity, cavity_sample,
)
from flowmllab.ghia import reference


def test_pressure_nodes_and_exact_jacobian():
    ops = cavity_operators(8, 100)
    x = ops['x'][1:-1]
    assert np.max(np.abs(nodal_derivative(x) @ x**4 - 4*x**3)) < 1e-12
    rng = np.random.default_rng(144)
    state = rng.normal(size=3*ops['size']) * 0.05
    direction = rng.normal(size=len(state))
    eps = 2e-7
    measured = (cavity_residual(state+eps*direction,ops)-cavity_residual(state-eps*direction,ops))/(2*eps)
    exact = cavity_jacobian(state,ops) @ direction
    assert np.linalg.norm(exact-measured) / np.linalg.norm(exact) < 1e-8


def test_full_navier_stokes_manufactured_velocity_and_pressure():
    ops = cavity_operators(8, 100, 'stationary')
    y,x = np.meshgrid(ops['x'][1:-1],ops['x'][1:-1],indexing='ij')
    b = Polynomial([0,0,1,-2,1])
    f,fp,fpp,fppp = [b.deriv(k)(x) for k in range(4)]
    g,gp,gpp,gppp = [b.deriv(k)(y) for k in range(4)]
    u,v = 16*f*gp,-16*fp*g
    ux,uy,vx,vy = 16*fp*gp,16*f*gpp,-16*fpp*g,-16*fp*gp
    lapu,lapv = 16*(fpp*gp+f*gppp),-16*(fppp*g+fp*gpp)
    p = x*x+2*y*y-0.75
    force = ((-lapu/100+u*ux+v*uy+2*x).ravel(),(-lapv/100+u*vx+v*vy+4*y).ravel())
    state = np.concatenate([u.ravel(),v.ravel(),p.ravel()])
    assert np.max(np.abs(cavity_residual(state,ops,force))) < 2e-12
    sol = solve_spectral_cavity(8,100,'stationary',tolerance=1e-11,forcing=force)
    assert sol['converged']
    assert np.max(np.abs(sol['state']-state)) < 1e-9


def test_cavity_no_slip_pressure_gauge_ghia_and_refinement():
    low = solve_spectral_cavity(12,100)
    high = solve_spectral_cavity(24,100)
    assert low['converged'] and high['converged']
    assert high['divergence_nodes_max'] < 1e-10
    assert high['divergence_core_rms'] < low['divergence_core_rms']
    assert np.all(high['u'][-1,1:-1] == 1)
    assert np.all(high['u'][:,[0,-1]] == 0)
    assert np.all(high['u'][0] == 0)
    assert np.all(high['v'][[0,-1],:] == 0)
    assert np.all(high['v'][:,[0,-1]] == 0)
    assert abs(cavity_sample(high,'p',[.5],[.5])[0,0]) < 1e-12
    r = reference(100)
    u = cavity_sample(high,'u',[.5],r['y'])[:,0]
    v = cavity_sample(high,'v',r['x'],[.5])[0]
    assert np.sqrt(np.mean((u-r['u'])**2)) < .004
    assert np.sqrt(np.mean((v-r['v'])**2)) < .006


def test_smooth_lid_off_grid_continuity_and_failed_budget():
    low = solve_spectral_cavity(12,100,'smooth')
    high = solve_spectral_cavity(24,100,'smooth')
    assert high['converged']
    assert high['divergence_core_rms'] < low['divergence_core_rms']/5
    limited = solve_spectral_cavity(12,400,max_iterations=1)
    assert not limited['converged']
    assert limited['stop_reason'] == 'iteration budget reached'
