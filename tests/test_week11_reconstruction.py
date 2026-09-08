"""Analytic controls for the optional reconstruction audit."""
import numpy as np
import pytest

torch=pytest.importorskip('torch')
from qa.run_week11_reconstruction import UNet, degrade, diagnostics, mask_scores


def test_rotation_and_shear():
    y,x=np.mgrid[-1:1:17j,-2:2:33j]
    a=np.stack((-y,x))[None]
    swirl,w,div=diagnostics(a,.125,.125)
    np.testing.assert_allclose(swirl,1)
    np.testing.assert_allclose(w,2)
    np.testing.assert_allclose(div,0)
    a=np.stack((y,np.zeros_like(x)))[None]
    np.testing.assert_allclose(diagnostics(a,.125,.125)[0],0)


def test_shape_and_constant_degradation():
    a=torch.ones((2,2,32,78))
    torch.testing.assert_close(degrade(a),a)
    for outputs in (1,2):
        assert UNet(outputs)(a).shape==(2,outputs,32,78)


def test_mask_scores_exclude_edges():
    a=np.zeros((2,10,10),bool); b=a.copy(); b[:,0]=True
    assert mask_scores(a,b)=={'dice':1.,'iou':1.}
    a[:,3:6,3:6]=True
    assert mask_scores(a,b)=={'dice':0.,'iou':0.}
