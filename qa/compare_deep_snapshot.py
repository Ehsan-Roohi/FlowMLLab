"""Compare an in-progress author PINN snapshot with accepted Nektar and Foam.

Read-only inputs. Full interior and depth-resolved errors are both retained;
small global errors cannot certify the much weaker bottom vortices.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from audit_week13_nektar import structured, integrate_paths
from audit_week13_cfd import audit as foam_audit, extrema
from check_week13_nektar_vtu import read_vtu
from run_week13_nektar_refinement import accepted_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pinn', type=Path, required=True)
    p.add_argument('--chunk', type=Path, required=True)
    p.add_argument('--foam', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.pinn.parent/'audit.json').read_text())
    re, depth = meta['case']['Re'], meta['case']['depth_over_width']
    if depth != 5 or re != 100 or meta['case']['tri'] != 0:
        raise ValueError('This audit is specifically Re100 D5 tri0')
    source, marker, spec = accepted_source(a.chunk, re)
    xn, yn, nf, _ = structured(read_vtu(source/'cavity.vtu'), depth=depth)
    pn = np.load(a.pinn)
    x, y = pn['x'], pn['y']
    if not np.isclose(y[-1], depth) or not np.isclose(x[-1], 1):
        raise ValueError('PINN physical grid mismatch')
    def at(xs, ys, f, q):
        return np.stack([RegularGridInterpolator((ys,xs), f[k])(q) for k in ('u','v')], axis=-1)
    def metrics(value, ref, yy, area):
        out = {}
        for label, mask in [('all', np.ones(yy.shape, bool))]+[(f'y_{i}_{i+1}', (yy>=i)&(yy<i+1)) for i in range(5)]:
            w = area[mask]; d = value[mask]-ref[mask]; r = ref[mask]
            den = np.sum(w[:,None]*r*r)
            out[label] = dict(relative_L2=float(np.sqrt(np.sum(w[:,None]*d*d)/den)) if den>0 else None,
                              absolute_rms_Ulid=float(np.sqrt(np.sum(w[:,None]*d*d)/np.sum(w))))
        return out
    mx=(np.arange(160)+.5)/160; my=(np.arange(800)+.5)/160
    xx,yy=np.meshgrid(mx,my); q=np.c_[yy.ravel(),xx.ravel()]
    ref=at(xn,yn,nf,q); val=at(x,y,pn,q)
    report=dict(case=meta['case'], checkpoint=meta['checkpoint'], nektar_time=marker['time'],
                nektar_grid=[spec['nx'],spec['ny']], pinn_vs_nektar=metrics(val,ref,q[:,0],np.ones(len(q))),
                limitations=['Intermediate PINN checkpoint, not a final trained result.',
                             'Smoothed PINN lid differs from CFD near top corners.',
                             'Streamfunction is integrated from velocity; weak-vortex strength needs resolution checks.',
                             'CFD solver completion alone is not mesh independence.'])
    gx,gy=np.meshgrid(x,y); gq=np.c_[gy.ravel(),gx.ravel()]
    ni=at(xn,yn,nf,gq).reshape(len(y),len(x),2)
    psin,_=integrate_paths(x,y,ni[:,:,0],ni[:,:,1])
    psip,_=integrate_paths(x,y,pn['u'],pn['v'])
    report['nektar_vortices']=extrema(psin,x,y)
    report['pinn_vortices']=extrema(psip,x,y)
    report['openfoam']=[]
    for case in a.foam:
        record,(xf,yf,vf,psi,area)=foam_audit(case)
        if record['re'] != re or not np.isclose(record['depth_over_width'],depth):
            raise ValueError('Foam case mismatch')
        fx,fy=np.meshgrid(xf,yf); fq=np.c_[fy.ravel(),fx.ravel()]
        fv=vf[:,:,:2].reshape(-1,2)
        record['pinn_vs_foam']=metrics(at(x,y,pn,fq),fv,fq[:,0],area.ravel())
        record['foam_vs_nektar']=metrics(fv,at(xn,yn,nf,fq),fq[:,0],area.ravel())
        report['openfoam'].append(record)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    plt.rcParams.update({'font.size':14})
    fig,ax=plt.subplots(1,3,figsize=(11,13),layout='constrained')
    levels=np.unique(np.r_[-.12,-.1,-.08,-.06,-.04,-.02, -np.logspace(-10,-2,17),np.logspace(-10,-2,17)])
    for axis,psi,title in zip(ax[:2],[psin,psip],['Nektar++','PINN snapshot']):
        axis.contour(x,y,psi,levels=levels,colors='black',linewidths=.7)
        axis.set_title(title)
    im=ax[2].pcolormesh(x,y,np.hypot(ni[:,:,0]-pn['u'],ni[:,:,1]-pn['v']),shading='auto',cmap='magma')
    ax[2].set_title('Velocity difference')
    fig.colorbar(im,ax=ax[2],shrink=.45,label='|difference| / lid speed')
    for axis in ax: axis.set(xlabel='x/W',ylabel='y/W',aspect='equal')
    fig.suptitle('Re=100, D/W=5 — intermediate checkpoint comparison')
    fig.savefig(a.output/'streamfunction_comparison.png',dpi=220)
    plt.close(fig)
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__': main()
