"""Compare retained author PINN and Nektar exports, including boundary mismatch."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from audit_week13_nektar import structured, integrate_paths
from check_week13_nektar_vtu import read_vtu
from evaluate_chris_deep_original import vortex_candidates


def main():
    p = argparse.ArgumentParser()
    for k in ('pinn', 'vtu', 'previous', 'output'):
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args()
    audit = json.loads((a.pinn.parent/'audit.json').read_text())
    assert audit['case'] == {'Re':1000.0, 'depth_over_width':2.2, 'tri':0.0}
    pn = np.load(a.pinn)
    xn, yn, fn, continuity = structured(read_vtu(a.vtu), depth=2.2)
    xo, yo, fo, _ = structured(read_vtu(a.previous), depth=2.2)
    x, y = pn['x'], pn['y']
    xx, yy = np.meshgrid(x,y)
    q = np.c_[yy.ravel(),xx.ravel()]
    def interp(xs, ys, f):
        return RegularGridInterpolator((ys,xs), f)(q).reshape(xx.shape)
    nf = {k:interp(xn,yn,fn[k]) for k in ('u','v')}
    old = {k:interp(xo,yo,fo[k]) for k in ('u','v')}
    # Equal-area midpoint quadrature avoids boundary/corner sample overweighting.
    mx=(np.arange(160)+.5)/160; my=(np.arange(352)+.5)*2.2/352
    mxx,myy=np.meshgrid(mx,my); mq=np.c_[myy.ravel(),mxx.ravel()]
    def vel(xs,ys,f):
        return np.column_stack([RegularGridInterpolator((ys,xs),f[k])(mq) for k in ('u','v')])
    ref=vel(xn,yn,fn)
    rel=lambda v: float(np.linalg.norm(v-ref)/np.linalg.norm(ref))
    nh,nv=integrate_paths(x,y,nf['u'],nf['v'])
    ph,pv=integrate_paths(x,y,pn['u'],pn['v'])
    report=dict(case=audit['case'],checkpoint=audit['checkpoint'],
        nektar_file=str(a.vtu),previous_file=str(a.previous),
        velocity_relative_L2=rel(vel(x,y,pn)),
        nektar_last_interval_relative_L2=rel(vel(xo,yo,fo)),
        lid_max_difference=float(np.max(abs(nf['u'][-1]-pn['u'][-1]))),
        nektar_vortices=vortex_candidates(nh,x,y),
        pinn_vortices=vortex_candidates(ph,x,y),
        path_disagreement=dict(nektar=float(np.max(abs(nh-nv))),pinn=float(np.max(abs(ph-pv)))),
        continuity=continuity,
        limitations=['Boundary profiles must match before interpreting differences as PINN error.',
            'Last saved interval does not establish steady or mesh convergence.',
            'Streamfunction extrema use export-grid integration, not spectral roots.'])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':13,'axes.titlesize':14})
    fig,ax=plt.subplots(1,3,figsize=(13,9),layout='constrained')
    levels=np.unique(np.r_[-.12,-.10,-.08,-.06,-.04,-.02,-.005,-1e-3,-1e-4,-1e-5,1e-5,1e-4,1e-3,.003,.006,.009,.012,.015])
    for axis,f,title in zip(ax[:2],[nh,ph],['Nektar++ at t=120','Chris PINN checkpoint 55118']):
        axis.contour(x,y,f,levels=levels,colors='black',linewidths=.8)
        axis.set(xlabel='x/W',ylabel='y/W',title=title,aspect='equal')
    im=ax[2].pcolormesh(x,y,np.hypot(nf['u']-pn['u'],nf['v']-pn['v']),shading='auto',cmap='magma')
    ax[2].set(xlabel='x/W',ylabel='y/W',title='Velocity difference / lid speed',aspect='equal')
    fig.colorbar(im,ax=ax[2],shrink=.6)
    fig.savefig(a.output/'fields.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for f,label in [(nf,'Nektar++'),(pn,'PINN')]:
        ax[0].plot(f['u'][:,len(x)//2],y,label=label)
        ax[1].plot(x,f['v'][len(y)//2],label=label)
        ax[2].plot(x,f['u'][-1],label=label)
    for axis in ax: axis.legend(); axis.grid(alpha=.2)
    ax[0].set(xlabel='u/U',ylabel='y/W',title='Vertical centreline')
    ax[1].set(xlabel='x/W',ylabel='v/U',title='Horizontal centreline')
    ax[2].set(xlabel='x/W',ylabel='u/U',title='Lid boundary comparison')
    fig.savefig(a.output/'profiles.png',dpi=180); plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
