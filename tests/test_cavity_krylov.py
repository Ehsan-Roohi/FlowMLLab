"""Admission tests for the experimental linear solver, independent of timing claims."""
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flowmllab'))
sys.path.insert(0,str(ROOT/'qa'))
import cavity_steady as base
from cavity_krylov import LinearSolver,KrylovConfig,prolongation
from benchmark_cavity_solvers import injected_linear_solver


def test_transfer_keeps_two_fields_separate():
    p=prolongation(32);coarse=15**2;fine=31**2
    v=np.r_[np.ones(coarse),np.zeros(coarse)]
    result=p@v
    assert np.count_nonzero(result[fine:])==0
    np.testing.assert_allclose(result[:fine].reshape(31,31)[1:-1,1:-1],1.)
    np.testing.assert_allclose(result[0],.25)


def test_original_linear_equation_and_inputs_preserved():
    solver=base.SteadyCavity(16)
    a=solver.jacobian(np.zeros(2*solver.n),100.)
    b=np.random.default_rng(731).normal(size=a.shape[0])
    a_copy=a.copy();b_copy=b.copy()
    expected=spsolve(a,b)
    linear=LinearSolver('gmres_mg',16,KrylovConfig(coarse_cells=8,ilu_fill_factor=8.,ilu_drop_tol=1e-4))
    actual=linear(a,b)
    assert linear.records[-1]['levels']==2
    assert np.linalg.norm(a@actual-b)/np.linalg.norm(b)<1e-7
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8)
    np.testing.assert_array_equal(a.data,a_copy.data)
    np.testing.assert_array_equal(b,b_copy)


def test_outer_newton_is_identical_and_injection_restores():
    solver=base.SteadyCavity(16)
    initial=np.zeros(2*solver.n)
    original=base.spsolve
    direct,_=solver.solve(initial,100.)
    linear=LinearSolver('gmres_mg',16,KrylovConfig(coarse_cells=8,ilu_fill_factor=8.,ilu_drop_tol=1e-4))
    with injected_linear_solver(linear):
        candidate,_=solver.solve(initial,100.)
    assert base.spsolve is original
    assert abs(solver.residual(candidate,100.)).max()<1e-8
    np.testing.assert_allclose(candidate,direct,rtol=1e-6,atol=1e-8)
    with pytest.raises(RuntimeError):
        with injected_linear_solver(linear):
            raise RuntimeError('Intentional restoration check')
    assert base.spsolve is original
