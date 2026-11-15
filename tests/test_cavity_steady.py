"""Independent stencil and analytic-Jacobian checks for the steady CFD solver."""
import numpy as np
from flowmllab.cavity_steady import SteadyCavity


def test_newton_jacobian_directional_derivative():
    solver = SteadyCavity(12)
    rng = np.random.default_rng(71)
    z = np.r_[rng.normal(size=solver.n)*.001, rng.normal(size=solver.n)]
    direction = np.r_[rng.normal(size=solver.n)*.001, rng.normal(size=solver.n)]
    eps = 1e-5
    finite_difference = (solver.residual(z+eps*direction, 3200)-solver.residual(z-eps*direction, 3200))/(2*eps)
    np.testing.assert_allclose(solver.jacobian(z,3200) @ direction, finite_difference, rtol=1e-8, atol=1e-8)


def test_independent_full_grid_spatial_equations():
    solver = SteadyCavity(12)
    rng = np.random.default_rng(91)
    z = np.r_[rng.normal(size=solver.n)*.001, rng.normal(size=solver.n)]
    f = solver.fields(z)
    w, p, u, v, h = f['omega'], f['psi'], f['u'], f['v'], solver.h
    lap = lambda a: (a[1:-1,2:]+a[1:-1,:-2]+a[2:,1:-1]+a[:-2,1:-1]-4*a[1:-1,1:-1])/h**2
    independent = lap(w)/7500-u[1:-1,1:-1]*(w[1:-1,2:]-w[1:-1,:-2])/(2*h)-v[1:-1,1:-1]*(w[2:,1:-1]-w[:-2,1:-1])/(2*h)
    residual = solver.residual(z,7500)
    np.testing.assert_allclose(residual[:solver.n], (lap(p)+w[1:-1,1:-1]).ravel(), atol=1e-12)
    np.testing.assert_allclose(residual[solver.n:], independent.ravel(), atol=1e-12)
    np.testing.assert_allclose(w[-1,1:-1], -2*p[-2,1:-1]/h**2-2/h)


def test_zero_lid_and_time_rhs_consistency():
    solver = SteadyCavity(12, lid=0)
    z, history = solver.solve(np.zeros(2*solver.n),10000)
    assert len(history) == 1
    np.testing.assert_array_equal(solver.time_rhs(z[solver.n:],10000), 0)
    solver = SteadyCavity(16)
    z, _ = solver.solve(np.zeros(2*solver.n),100)
    assert np.max(np.abs(solver.residual(z,100))) < 1e-8
    assert np.max(np.abs(solver.time_rhs(z[solver.n:],100))) < 1e-8
    advanced = solver.rk3(z[solver.n:],100,.001)
    np.testing.assert_allclose(advanced,z[solver.n:],atol=1e-10,rtol=0)
