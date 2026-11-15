import numpy as np
import pytest
from flowmllab.deep_cavity_audit import metrics,masks,horizontal_profile
from flowmllab.cavity_course import compare_field


def test_vector_rms_and_pressure_gauge_decomposition():
    y,x=np.mgrid[0:4,0:5];mask=x>=1
    ref={'u':np.ones_like(x,dtype=float)*3,'v':np.ones_like(x,dtype=float)*4,
         'p':x*.1,'omega':np.ones_like(x,dtype=float)}
    pred={**ref,'u':ref['u']+.3,'v':ref['v']+.4,'p':ref['p']+.001+y*.002}
    a=metrics(pred,ref,mask)
    assert np.isclose(a['velocity_RMS_U'],.5)
    assert np.isclose(a['velocity_NRMSE_percent'],10)
    assert np.isclose(a['pressure_RMS_rhoU2']**2,
        a['pressure_signed_mean_offset_rhoU2']**2+a['pressure_local_shape_RMS_rhoU2']**2)
    shifted=metrics({**pred,'p':pred['p']+.02},ref,mask)
    assert np.isclose(shifted['pressure_local_shape_RMS_rhoU2'],a['pressure_local_shape_RMS_rhoU2'])
    assert np.isclose(shifted['pressure_signed_mean_offset_rhoU2']-a['pressure_signed_mean_offset_rhoU2'],.02)
    with pytest.raises(ValueError):metrics(pred,ref,np.zeros_like(mask))


def test_actual_profile_height_and_no_wall_extrapolation():
    x=np.linspace(.002,.998,11);h=np.linspace(.002,4.998,51)
    regions=masks(x,h)
    assert np.all(~regions['Floor_0.4'] | regions['Floor_0.65'])
    field=np.add.outer(h,x);d={'x':x,'h':h,'re100_cfd_u':field}
    xx,u,height=horizontal_profile(d,100,'cfd','u',.33)
    assert height in h and np.all(u==height+xx)
    with pytest.raises(ValueError):horizontal_profile(d,100,'cfd','u',0)
    with pytest.raises(ValueError):compare_field(100,x,np.linspace(0,1,51),field,field)
