import numpy as np
import pytest
from flowmllab import ghia, ghia_surrogate as m


def test_all_seven_re_and_raw_anomalies_retained():
    assert tuple(ghia.GHIA)==(100,400,1000,3200,5000,7500,10000)
    assert ghia.GHIA[3200]['u'][9]==-.86636
    assert ghia.GHIA[10000]['u'][8]==.03111
    assert ghia.GHIA[400]['v'][5]==-.23827
    for re in ghia.REYNOLDS:
        for component in ('u','v'):
            assert len(ghia.GHIA[re][component])==17
            n = 16 if (re,component) in ghia.FLAGGED else 17
            assert ghia.valid_mask(re,component).sum()==n
    with pytest.raises(ValueError): ghia.reference(400.1)


def test_profile_evaluation_does_not_extrapolate_or_divide_by_zero():
    ref=ghia.GHIA[400]
    result=ghia.profile_errors(400,'v',ref['x'][::-1],ref['v'][::-1])
    assert result['published_relative_l2']==0
    with pytest.raises(ValueError): ghia.profile_errors(400,'v',[.1,.9],[0,0])


def test_manual_gradient_and_hard_walls():
    cfg=m.ModelConfig(depth=2,width=5)
    theta,shapes=m.initialize(cfg,7);x,y,w=m.training_data('wall')
    value,grad=m.objective(theta,shapes,x,y,w)
    direction=np.random.default_rng(9).normal(size=len(theta));direction/=np.linalg.norm(direction)
    eps=1e-5
    fd=(m.objective(theta+eps*direction,shapes,x,y,w)[0]-m.objective(theta-eps*direction,shapes,x,y,w)[0])/(2*eps)
    assert np.isclose(fd,grad@direction,rtol=1e-6,atol=1e-9)
    model={'theta':theta,'shapes':shapes}
    for re in ghia.REYNOLDS:
        np.testing.assert_array_equal(m.predict(model,re,[0,1],'u'),[0,1])
        np.testing.assert_array_equal(m.predict(model,re,[0,1],'v'),[0,0])


def test_whole_re_holdout_is_excluded_from_fit():
    x,_,_=m.training_data()
    training_re=10**(np.unique(x[:,1])+3)
    np.testing.assert_allclose(training_re,[100,1000,5000,10000])
    assert set(training_re).isdisjoint({400,3200,7500})


def test_legacy_notebook_helper_uses_same_reference_and_scores():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'notebooks/week04/cavity_project_utils.py'
    spec=importlib.util.spec_from_file_location('legacy_cavity_course',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.GHIA is ghia.GHIA
    re=3200;ref=ghia.reference(re);x=ref['x'][::-1];y=ref['y'][::-1]
    u=np.tile(ref['u'][::-1,None],(1,len(x)));v=np.tile(ref['v'][::-1],(len(y),1))
    u[np.isclose(y,.4531),:]+=.5
    result={'Re':re,'U':1,'x':x,'y':y,'u':u,'v':v}
    assert helper.ghia_errors(result)[0]<1e-14
    assert helper.ghia_errors(result,screened=False)[0]>.1
