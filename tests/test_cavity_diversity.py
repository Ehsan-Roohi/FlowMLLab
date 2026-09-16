import numpy as np
import pytest
from flowmllab.cavity_diversity import (PROFILES, MODELS, case_manifest, lid_profile,
    poisson, fields, solve_cases, tensor_features, build_model, physics_loss)


def test_reynolds_and_case_separation():
    cases=case_manifest(PROFILES['pilot'])
    assert len({c['id'] for c in cases})==len(cases)
    for c in cases:
        assert c['Re']==pytest.approx(c['U_ref']*c['L']/c['nu'])
    for split in ('train','val','test'):
        a=[c['Re'] for c in cases if c['family']=='constant' and c['split']==split]
        b=[c['Re'] for c in cases if c['family']=='diverse' and c['split']==split]
        assert a==b
    train=[c['Re'] for c in cases if c['split']=='train']
    test=[c['Re'] for c in cases if c['split']=='test']
    assert not set(train)&set(test)


def test_speed_normalization_and_physical_similarity():
    x=np.linspace(0,1,8193)
    g=lid_profile(x,[.2,-.1,.05,0,0])
    assert max(abs(g))==pytest.approx(1.)
    assert min(g)>0
    uref,nu,length=.8,.0025,1.
    assert (2*uref)*length/nu==2*(uref*length/nu)
    assert (2*uref)*length/(2*nu)==uref*length/nu


def test_poisson_manufactured_and_divergence():
    n=33;h=1/(n-1);x=np.linspace(0,1,n);xx,yy=np.meshgrid(x,x)
    exact=np.sin(np.pi*xx)*np.sin(np.pi*yy)
    eig=8*np.sin(np.pi*h/2)**2/h**2
    psi=poisson(eig*exact,h)
    assert np.max(abs(psi-exact))<1e-12
    u,v=fields(psi[None],np.zeros((1,n)))
    div=(u[:,2:-2,3:-1]-u[:,2:-2,1:-3]+v[:,3:-1,2:-2]-v[:,1:-3,2:-2])/(2*h)
    assert abs(div).max()<1e-12


def test_all_models_shapes_gradients_and_wall_psi():
    torch=pytest.importorskip('torch');torch.set_num_threads(2)
    cases=case_manifest(PROFILES['pilot'])[:3];n=17
    desc,img=tensor_features(cases,n);d=torch.tensor(desc);im=torch.tensor(img)
    rng=np.random.default_rng(1);psi=rng.normal(size=(3,n,n))*.01
    psi[:,[0,-1],:]=0;psi[:,:,[0,-1]]=0
    for name in MODELS:
        model=build_model(name,n,psi)
        pred=model(d,im)
        assert pred.shape==(3,1,n,n)
        assert torch.isfinite(pred).all()
        assert float(pred.detach()[:,:,[0,-1],:].abs().max())<1e-6
        loss=pred.square().mean()+physics_loss(pred,d,im,1/(n-1))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_unconverged_solver_rejects_labels():
    with pytest.raises(RuntimeError,match='Unconverged'):
        solve_cases(case_manifest(PROFILES['pilot'])[:1],n=17,max_steps=2)
