"""Case-matched postprocessing on a common interior grid; no training or case writes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from audit_week13_cfd import audit as foam_audit, extrema
from audit_week13_nektar import structured, integrate_paths, diagnostics
from check_week13_nektar_vtu import read_vtu
from run_week42_deepplasma import load_upstream
from run_week13_rectangular_pinn import build_model, fields


def main():
    p = argparse.ArgumentParser()
    for name in ('vtu', 'previous', 'checkpoint', 'source', 'foam', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg = saved['config']; depth = cfg['aspect_ratio']; re = cfg['reynolds_number']
    xn, yn, fn, continuity = structured(read_vtu(a.vtu), depth=depth)
    xp, yp, fp, _ = structured(read_vtu(a.previous), depth=depth)
    report = dict(case=dict(Re=re, depth_over_width=depth), nektar=diagnostics(xn, yn, fn, re, depth), continuity=continuity)
    # Midpoint quadrature: equal physical cell areas, excluding singular wall corners.
    x = (np.arange(160)+.5)/160; y = (np.arange(352)+.5)*depth/352
    xx, yy = np.meshgrid(x,y); q = np.c_[yy.ravel(),xx.ravel()]
    def interp(x, y, f):
        return RegularGridInterpolator((y,x),f)(q)
    reference = np.column_stack([interp(xn,yn,fn[k]) for k in ('u','v')])
    def error(v):
        return dict(velocity_relative_L2=float(np.linalg.norm(v-reference)/np.linalg.norm(reference)),
                    u_rms=float(np.sqrt(np.mean((v[:,0]-reference[:,0])**2))),
                    v_rms=float(np.sqrt(np.mean((v[:,1]-reference[:,1])**2))))
    old = np.column_stack([interp(xp,yp,fp[k]) for k in ('u','v')])
    report['nektar_last_interval'] = error(old)
    report['openfoam'] = []
    # Compare each mesh on its own cell centres to avoid extrapolation.
    for case in sorted(a.foam.glob('re*-n*')):
        r, (xc,yc,vc,ps,area) = foam_audit(case)
        if r['re'] != re or not np.isclose(r['depth_over_width'],depth):
            raise ValueError('CFD/PINN case mismatch')
        xc2,yc2=np.meshgrid(xc,yc); qc=np.c_[yc2.ravel(),xc2.ravel()]
        nr=np.stack([RegularGridInterpolator((yn,xn),fn[k])(qc).reshape(len(yc),len(xc)) for k in ('u','v')],axis=-1)
        r['relative_velocity_L2_vs_nektar']=float(np.sqrt(np.sum(area[:,:,None]*(vc[:,:,:2]-nr)**2)/np.sum(area[:,:,None]*nr**2)))
        report['openfoam'].append(r)
    module, digest = load_upstream(a.source)
    module.device = torch.device('cpu')
    model=build_model(module,cfg.get('hidden_width',50),cfg.get('hidden_layers',3),cfg.get('activation','tanh'))
    model.load_state_dict(saved['model']); model.eval()
    values=[]
    xy=np.c_[xx.ravel(),yy.ravel()/depth]
    for start in range(0,len(xy),2048):
        z=torch.tensor(xy[start:start+2048],dtype=torch.float64,requires_grad=True)
        values.append(np.column_stack([v.detach().numpy().ravel() for v in fields(module,model,z,depth)]))
    values=np.concatenate(values)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite PINN fields')
    report['pinn']=error(values[:,:2])
    report['pinn']['vortices']=extrema(values[:,3].reshape(len(y),len(x)),x,y)
    report['pinn']['selection']=saved.get('selection_row')
    report['limitations']=['Nektar temporal change covers only the last saved interval.', 'Nektar p/h independence is not established by this comparison.', 'Vortex extraction excludes x<0.1 and x>0.9 and cannot certify corner eddies.', 'No paper reference is available in the table for D/W=2.2.']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__': main()
