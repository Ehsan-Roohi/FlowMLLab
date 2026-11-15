import numpy as np
from flowmllab import cavity_field_operator as op


def test_branch_backprop_directional_derivative():
    cfg=op.Config('test',depth=5,width=5)
    theta,shapes,_=op.initialize(cfg,3,17)
    rng=np.random.default_rng(20);x=rng.normal(size=(9,1));y=rng.normal(size=(9,3))
    direction=rng.normal(size=len(theta));direction/=np.linalg.norm(direction)
    value,gradient=op.objective(theta,shapes,x,y)
    eps=1e-5
    numerical=(op.objective(theta+eps*direction,shapes,x,y)[0]-op.objective(theta-eps*direction,shapes,x,y)[0])/(2*eps)
    np.testing.assert_allclose(gradient@direction,numerical,rtol=1e-7,atol=1e-9)


def test_pod_reconstruction_and_same_information_baseline():
    rng=np.random.default_rng(31);fields=rng.normal(size=(len(op.TRAIN),60))
    fields[:,0]=1;fields[:,1]=0
    trunk=op.make_trunk(fields)
    np.testing.assert_allclose(op.reconstruct(trunk,trunk['train_coefficients']),fields,atol=1e-13)
    target=[3200,7500]
    for coordinate in ('linear','log'):
        coeff=op.baseline_coefficients(trunk,target,coordinate)
        weights=op.mixing_weights(trunk,coeff)
        pred=op.reconstruct(trunk,coeff)
        np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-14)
        np.testing.assert_allclose(pred,weights@fields,atol=1e-13)
        train=np.log10(op.TRAIN) if coordinate=='log' else op.TRAIN
        query=np.log10(target) if coordinate=='log' else target
        direct=np.column_stack([np.interp(query,train,col) for col in fields.T])
        np.testing.assert_allclose(pred,direct,atol=1e-13)
        np.testing.assert_allclose(pred[:,0],1,atol=1e-14)
        np.testing.assert_allclose(pred[:,1],0,atol=1e-14)


def test_whole_re_splits_disjoint():
    assert not set(op.TRAIN)&set(op.VALIDATION)
    assert not set(op.TRAIN)&set(op.TEST)
    assert not set(op.TEST)&set(op.VALIDATION)
    assert set(op.TEST)=={3200,7500}
