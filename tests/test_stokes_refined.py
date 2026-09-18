import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Week 4.2 regeneration tests need the optional PyTorch dependency")
from common.w4utils import recover_pressure
from flowmllab.stokes_refined import PODCorrection,gradient_metrics
from flowmllab.stokes_correction import solve_stokes


def test_pressure_analytic_stagnation_flow():
    x=np.linspace(0,1,33);xx,yy=np.meshgrid(x,x)
    p,a=recover_pressure(xx,-yy,100,x,x)
    exact=-.5*(xx**2+yy**2);exact-=exact.mean()
    assert np.linalg.norm(p-exact)/np.linalg.norm(exact)<1e-6
    assert abs(p.mean())<1e-12
    assert a['pressure_grad_rel_residual']<1e-5


def test_pressure_velocity_scaling():
    # Doubling dimensional speed with fixed nu means Re doubles; nondimensional
    # analytic stagnation pressure remains the same, while dimensional rho U^2
    # pressure increases fourfold. Here viscous gradients vanish exactly.
    x=np.linspace(0,1,17);xx,yy=np.meshgrid(x,x)
    p1,_=recover_pressure(xx,-yy,100,x,x);p2,_=recover_pressure(xx,-yy,200,x,x)
    assert np.max(abs(p1-p2))<1e-7


def test_sobolev_gram_matches_velocity_field_error():
    rng=np.random.default_rng(4);n=17;b=rng.normal(size=(4,n*n))
    g,_,_=gradient_metrics(b,n);c=rng.normal(size=4)
    p=(c@b).reshape(n,n);h=1/(n-1)
    u=(p[2:,1:-1]-p[:-2,1:-1])/(2*h);v=(p[1:-1,2:]-p[1:-1,:-2])/(2*h)
    assert np.isclose(c@g@c,np.mean(u*u+v*v))


def test_pod_zero_walls_and_finite_gradients():
    cases=[dict(Re=100+30*i,coefficients=[.02*i,0,0,0,0]) for i in range(5)]
    lo=solve_stokes(cases,17)['psi'];target=lo.copy();target[:,1:-1,1:-1]*=.95
    model=PODCorrection(lo,target,np.array([c['Re'] for c in cases]))
    p=model(torch.tensor(lo),torch.tensor([c['Re'] for c in cases],dtype=torch.float64))
    assert torch.isfinite(p).all()
    assert p[:,[0,-1],:].abs().max()<1e-10
    p.square().mean().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters())
