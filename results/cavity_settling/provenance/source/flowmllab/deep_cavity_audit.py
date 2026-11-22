"""Recompute region-resolved D/W=5 errors from retained author fields.

This is postprocessing, not neural training or a CFD solver. Arrays are [h,x],
h is height above the floor in width units, and all reference/predicted fields
must share coordinates, lid, Reynolds number, and normalization.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def masks(x, h):
    X,H = np.meshgrid(x,h)
    inside=(X>.02)&(X<.98)&(H>.02)&(H<4.98)
    return {'Interior':inside, 'Upper-left':inside&(X<.25)&(H>4.75),
            'Floor_0.4':inside&(H<.4), 'Floor_0.65':inside&(H<.65)}


def load_bundle(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    path=directory/'reynolds_common_fields.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError('Retained field checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    if not np.all(np.diff(data['x'])>0) or not np.all(np.diff(data['h'])>0):
        raise ValueError('Coordinates must increase')
    shape=(len(data['h']),len(data['x']))
    for k,a in data.items():
        if k not in ('x','h') and (a.shape!=shape or not np.isfinite(a).all()):
            raise ValueError(f'Invalid field {k}')
    return data,manifest


def metrics(pred, ref, mask):
    """Uniform sampled weights, vector RMS, one interior pressure gauge.

    Inputs must already use the common interior pressure gauge. Local pressure
    shape and offset are additional diagnostics, never substituted for RMS.
    """
    if not np.any(mask): raise ValueError('Empty evaluation region')
    e2=(pred['u']-ref['u'])**2+(pred['v']-ref['v'])**2
    r2=ref['u']**2+ref['v']**2
    er=float(np.sqrt(np.mean(e2[mask]))); rr=float(np.sqrt(np.mean(r2[mask])))
    dp=(pred['p']-ref['p'])[mask]; offset=float(dp.mean())
    dw=(pred['omega']-ref['omega'])[mask]; wr=ref['omega'][mask]
    wrms=float(np.sqrt(np.mean(wr**2)))
    return {'count':int(mask.sum()), 'velocity_RMS_U':er,'CFD_velocity_RMS_U':rr,
            'velocity_NRMSE_percent':100*er/rr if rr>0 else None,
            'pressure_RMS_rhoU2':float(np.sqrt(np.mean(dp**2))),
            'pressure_signed_mean_offset_rhoU2':offset,
            'pressure_local_shape_RMS_rhoU2':float(np.sqrt(np.mean((dp-offset)**2))),
            'vorticity_RMS_UoverW':float(np.sqrt(np.mean(dw**2))),
            'vorticity_NRMSE_percent':100*float(np.sqrt(np.mean(dw**2)))/wrms if wrms>0 else None}


def evaluate(data):
    regions=masks(data['x'],data['h']);out=[]
    for re in (100,250,500,1000):
        pred={q:data[f're{re}_pinn_{q}'] for q in ('u','v','p','omega')}
        ref={q:data[f're{re}_cfd_{q}'] for q in ('u','v','p','omega')}
        # Stored p already has the frozen interior gauge. Do not re-centre it
        # locally or change its previously frozen sampling convention.
        for name, mask in regions.items():
            out.append({'Re':re,'region':name,**metrics(pred,ref,mask)})
    return out


def horizontal_profile(data, re, method, component, height):
    """Nearest retained physical row, with its actual height returned."""
    if height<data['h'][0] or height>data['h'][-1]:
        raise ValueError('Profile is outside retained samples')
    i=int(np.argmin(np.abs(data['h']-height)))
    return data['x'],data[f're{re}_{method}_{component}'][i],float(data['h'][i])
