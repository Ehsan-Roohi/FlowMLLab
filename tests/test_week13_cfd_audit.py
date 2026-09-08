import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'qa/audit_week13_cfd.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def test_analytic_streamfunction_and_vortex():
    x=(np.arange(160)+.5)/160; y=x.copy(); xx,yy=np.meshgrid(x,y)
    psi=-np.sin(np.pi*xx)**2*np.sin(np.pi*yy)**2
    u=-np.pi*np.sin(np.pi*xx)**2*np.sin(2*np.pi*yy)
    v=np.pi*np.sin(2*np.pi*xx)*np.sin(np.pi*yy)**2
    calc,closure=a.streamfunction(np.stack([u,v,u*0],axis=-1),x)
    assert np.max(abs(calc-psi))<2e-4
    assert closure<1e-14
    vortices=a.extrema(calc,x,y)
    assert len(vortices)==1
    assert abs(vortices[0]['x']-.5)<1e-6
    assert abs(vortices[0]['y']-.5)<1e-6


def test_binary_field():
    from types import SimpleNamespace
    values=np.arange(12,dtype='<f8').reshape(4,3)
    raw=b'FoamFile\n{\nformat     binary;\narch "LSB;label=32;scalar=64";\n}\ninternalField nonuniform List<vector>\n4\n('+values.tobytes()+b')'
    path=SimpleNamespace(read_bytes=lambda:raw)
    np.testing.assert_array_equal(a.foam_list(path,field=True),values)
