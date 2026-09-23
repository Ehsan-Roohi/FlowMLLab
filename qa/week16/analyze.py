"""Dimensional pressure reconstruction and off-body pressure extraction."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import LinearNDInterpolator
ROOT=Path(__file__).resolve().parents[2]
SENSORS=(.25,.5,.75)
X=np.linspace(-.2,2.6,561)

def load_csv(path):
    return np.genfromtxt(path,delimiter=',',names=True,deletechars='')

def analyze(folder):
    folder=Path(folder); meta=json.loads((folder/'metadata.json').read_text())
    u=load_csv(folder/'restart_flow.csv')
    rho=u['Density']; pressure=.4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/rho)
    if np.any(~np.isfinite(pressure)) or min(pressure)<=0 or min(rho)<=0: raise ValueError('Unphysical flow')
    f=LinearNDInterpolator(np.c_[u['x'],u['y']],pressure)
    signatures=np.array([f(X,np.full_like(X,r)) for r in SENSORS])
    if not np.all(np.isfinite(signatures)): raise ValueError('Sensor outside mesh')
    q=.5*1.4*101325*meta['mach']**2
    cp=(signatures-101325)/q
    s=load_csv(folder/'surface_flow.csv'); ids=np.argsort(s['x']); s=s[ids]
    p=.4*(s['Energy']-.5*(s['Momentum_x']**2+s['Momentum_y']**2)/s['Density'])
    # Integrate pressure on surfaces of revolution, no SU2 planar force convention.
    force=float(2*np.pi*np.sum(.5*((p[:-1]-101325)+(p[1:]-101325))*.5*(s['y'][:-1]+s['y'][1:])*np.diff(s['y'])))
    cd=force/q # reference area = L^2 = 1 m^2
    hist=load_csv(folder/'history.csv'); tail=hist['CD'][-100:]
    meta.update(cd_pressure=cd,peak_cp=float(np.max(cp[1])),peak_to_peak_cp=float(np.ptp(cp[1])),signature_l2=float(np.sqrt(np.trapezoid(cp[1]**2,X))),density_residual_log10=float(hist['rms[Rho]'][-1]),residual_drop=float(hist['rms[Rho]'][0]-hist['rms[Rho]'][-1]),drag_tail_relative_range=float(np.ptp(tail)/abs(np.mean(tail))),iterations=len(hist),min_pressure=float(min(pressure)))
    meta['converged']=bool(meta['residual_drop']>=5 and meta['drag_tail_relative_range']<1e-4)
    np.savez_compressed(folder/'extracted.npz',x=X,radii=np.array(SENSORS),cp=cp,surface_x=s['x'],surface_r=s['y'],surface_cp=(p-101325)/q)
    (folder/'metrics.json').write_text(json.dumps(meta,indent=2))
    return meta
if __name__=='__main__':
    import sys
    for name in sys.argv[1:]: print(json.dumps(analyze(ROOT/'results/week16_lowboom/runs'/name),indent=2))
