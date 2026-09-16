import numpy as np
from flowmllab.stokes_correction import stokes_operator,solve_stokes,correction_features
from flowmllab.cavity_diversity import solve_cases


def test_linear_residual_and_re_independence():
    cases=[dict(id='a',Re=100,coefficients=[0]*5),dict(id='b',Re=400,coefficients=[0]*5)]
    s=solve_stokes(cases,25)
    assert max(s['linear_relative_residual'])<1e-10
    assert np.max(abs(s['psi'][0]-s['psi'][1]))<1e-13
    assert s['psi'].min()<-.05
    assert np.max(abs(s['psi'][:,:,0]))==0


def test_manufactured_biharmonic_refinement():
    errors=[]
    for n in (17,33,65):
        lu,_,_=stokes_operator(n);m=n-2;x=np.linspace(0,1,n)[1:-1]
        xx,yy=np.meshgrid(x,x);g=lambda z:z*z*(1-z)**2;g2=lambda z:2-12*z+12*z*z
        exact=g(xx)*g(yy)
        forcing=-(24*g(yy)+2*g2(xx)*g2(yy)+24*g(xx))
        rhs=np.concatenate([np.zeros(m*m),forcing.ravel()]);p=lu.solve(rhs)[:m*m].reshape(m,m)
        errors.append(np.linalg.norm(p-exact)/np.linalg.norm(exact))
    assert errors[0]/errors[1]>3
    assert errors[1]/errors[2]>3


def test_low_re_ns_approaches_stokes():
    c=[dict(id='low_re',Re=.1,coefficients=[.1,-.05,0,0,0])]
    s=solve_stokes(c,17);ns=solve_cases(c,n=17,max_steps=10000,tolerance=1e-6)
    assert np.linalg.norm(s['psi']-ns['psi'])/np.linalg.norm(s['psi'])<.002


def test_no_target_in_features_and_matched_capacity():
    c=[dict(id='a',Re=200,coefficients=[.2,0,0,0,0])];s=solve_stokes(c,17)
    a,b=correction_features(c,s,'direct');d,e=correction_features(c,s,'stokes_corrected')
    assert a.shape==d.shape and b.shape==e.shape
    assert np.all(b[:,4:]==0) and np.max(abs(e[:,4:]))>0
    assert a[0,0]==d[0,0]==.5
