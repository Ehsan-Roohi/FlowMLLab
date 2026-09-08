"""Read-only field audit of the structured OpenFOAM cavity campaign.

Writes derived JSON/plots only into a separate --output directory.
Cheng & Hung (2006), Table 2, journal p1060; D=H/W, Re=U*W/nu.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import maximum_filter
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.optimize import root

# Rows: x, y, psi, omega (paper's positive clockwise convention).
REF = {
 (100,1): [[.6162,.7372,-.1018,3.1703]],
 (1000,1): [[.5315,.5672,-.1131,2.0500]],
 (100,5): [[.6194,4.7362,-.1009,3.1599],[.5321,3.5844,7.9446e-4,-3.0721e-2],
           [.5049,2.1957,-2.2038e-6,8.2211e-5],[.5015,.8175,6.1200e-9,-2.3431e-7]],
 (500,5): [[.5523,4.6055,-.1070,2.2430],[.4115,3.8588,9.3455e-3,-.3426],
           [.4943,2.5406,-3.3516e-5,1.2876e-3],[.5041,1.1571,9.3654e-8,-3.5355e-6]],
 (1000,5): [[.5361,4.5854,-.1105,2.0404],[.3497,3.8528,1.2196e-2,-.4322],
            [.4671,2.8055,-1.5158e-4,5.9247e-3],[.5056,1.4141,4.1837e-7,-1.5787e-5]],
 (500,7): [[.5531,6.6024,-.1070,2.2432],[.4076,5.8555,9.3446e-3,-.3422],
           [.49209,4.5401,-3.3509e-5,1.2804e-3],[.5025,3.1596,9.3648e-8,-3.5297e-6],
           [.5004,1.7792,-2.6196e-10,9.9351e-9],[.5001,.4274,5.5424e-13,-2.9746e-11]],
 (1000,7): [[.53207,6.5822,-.1103,2.0401],[.3432,5.8492,1.2206e-3,-.4321],
            [.4634,4.8062,-1.6158e-4,6.0193e-3],[.5038,3.4101,4.4573e-7,-1.7281e-5],
            [.5006,2.0248,-1.2469e-9,4.5929e-8],[.5001,.6456,3.3861e-12,-1.4091e-10]],
}


def foam_list(path, field=False, components=3, dtype='<f8'):
    b = path.read_bytes()
    if not re.search(rb'format\s+binary;',b[:900]) or b'LSB;label=32;scalar=64' not in b[:900]:
        raise ValueError(f'Unsupported OpenFOAM encoding: {path}')
    offset = b.index(b'internalField') if field else b.index(b'\n}', b.index(b'FoamFile')) + 2
    m = re.search(rb'\n\s*(\d+)\s*\n\(', b[offset:])
    if m is None:
        raise ValueError(f'No nonuniform list: {path}')
    n = int(m[1]); start = offset + m.end()
    a = np.frombuffer(b, dtype=dtype, count=n*components, offset=start).copy()
    if not np.isfinite(a).all():
        raise ValueError(f'Nonfinite array: {path}')
    return a.reshape(n, components) if components > 1 else a


def mesh(case, spec):
    nx, ny = spec['nx'], spec['ny']
    p = foam_list(case/'constant/polyMesh/points').reshape(2,ny+1,nx+1,3)
    xe, ye = p[0,0,:,0], p[0,:,0,1]
    assert np.allclose(p[0,:,:,0], xe[None,:])
    assert np.allclose(p[0,:,:,1], ye[:,None])
    assert np.all(np.diff(xe)>0) and np.all(np.diff(ye)>0)
    assert np.isclose(xe[-1],1) and np.isclose(ye[-1],spec['depth_over_width'])
    return (xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2, np.outer(np.diff(ye),np.diff(xe))


def streamfunction(velocity, x):
    # Left-wall psi=0; integrate locally across x, not through many weak layers.
    v = np.pad(velocity[:,:,1], ((0,0),(1,1)))
    ps = -cumulative_trapezoid(v, np.r_[0,x,1], axis=1, initial=0)
    return ps[:,1:-1], float(np.max(np.abs(ps[:,-1])))


def extrema(psi, x, y):
    a = abs(psi)
    mask = (a == maximum_filter(a, size=5)) & (a>1e-15)
    mask[:,(x<.1)|(x>.9)] = False
    mask[:2]=False; mask[-2:]=False
    spline = RectBivariateSpline(y,x,psi)
    out=[]
    for j,i in zip(*np.where(mask)):
        def grad(q):
            return [spline.ev(q[1],q[0],dy=1),spline.ev(q[1],q[0],dx=1)]
        r=root(grad,[x[i],y[j]])
        xx,yy = r.x
        if not r.success or abs(xx-x[i])>3*np.max(np.diff(x)) or abs(yy-y[j])>3*np.max(np.diff(y)):
            xx,yy=x[i],y[j]
        if not (x[0]<=xx<=x[-1] and y[0]<=yy<=y[-1]):
            continue
        if any(np.hypot(xx-v['x'],yy-v['y'])<1e-5 for v in out):
            continue
        out.append(dict(x=float(xx),y=float(yy),psi=float(spline.ev(yy,xx))))
    return sorted(out,key=lambda v:-v['y'])


def audit(case):
    spec=json.loads((case/'case-spec.json').read_text())
    x,y,area=mesh(case,spec)
    times=sorted([p for p in case.iterdir() if p.is_dir() and p.name.isdigit() and (p/'U').exists()],key=lambda p:int(p.name))
    time=times[-1]
    vel=foam_list(time/'U',field=True).reshape(len(y),len(x),3)
    psi,closure=streamfunction(vel,x)
    vortices=extrema(psi,x,y)
    omega=np.gradient(vel[:,:,0],y,axis=0,edge_order=2)-np.gradient(vel[:,:,1],x,axis=1,edge_order=2)
    interp=RegularGridInterpolator((y,x),omega,bounds_error=True)
    for v in vortices:
        v['omega_paper_convention']=float(interp([[v['y'],v['x']]])[0])
    comparisons=[]
    for k,ref in enumerate(REF.get((spec['re'],spec['depth_over_width']),[])):
        rx,ry,rpsi,rw=ref
        candidates=[v for v in vortices if v['psi']*rpsi>0 and abs(v['y']-ry)<.45]
        if not candidates:
            comparisons.append(dict(vortex=k+1,reference=ref,status='not_resolved_in_reference_neighborhood'))
            continue
        v=min(candidates,key=lambda v:np.hypot(v['x']-rx,v['y']-ry))
        comparisons.append(dict(vortex=k+1,reference=ref,computed=v,
            centre_distance_W=float(np.hypot(v['x']-rx,v['y']-ry)),
            psi_error_percent=100*abs(v['psi']/rpsi-1),
            omega_error_percent=100*abs(v['omega_paper_convention']/rw-1),
            status='paper_psi_exponent_suspect' if (spec['re'],spec['depth_over_width'],k)==(1000,7,1) else 'compared_not_certified'))
    logs=sorted(case.glob('log.simpleFoam.*'),key=lambda p:p.stat().st_mtime)
    log=logs[-1].read_text(errors='replace')
    residuals={key:float(re.findall(rf'Solving for {key}, Initial residual = ([^,]+)',log)[-1]) for key in ['Ux','Uy','p']}
    result=dict(case=case.name,**spec,time=int(time.name),residuals=residuals,
        right_wall_psi_closure=closure,vortices=vortices,paper=comparisons,
        field_sha256=hashlib.sha256((time/'U').read_bytes()).hexdigest())
    # Previous processor snapshots allow an iterative change check without writing into cases.
    proc0=case/'processor0'
    previous=sorted([p.name for p in proc0.iterdir() if p.is_dir() and p.name.isdigit() and 0<int(p.name)<int(time.name)],key=int)
    if previous:
        old=np.empty_like(vel.reshape(-1,3)); counts=np.zeros(len(old),dtype=int)
        for proc in sorted(case.glob('processor[0-9]*')):
            ids=foam_list(proc/'constant/polyMesh/cellProcAddressing',components=1,dtype='<i4')
            old[ids]=foam_list(proc/previous[-1]/'U',field=True); counts[ids]+=1
        assert np.all(counts==1)
        old=old.reshape(vel.shape)
        result['previous_time']=int(previous[-1])
        result['velocity_relative_change']=float(np.sqrt(np.sum(area[:,:,None]*(vel-old)**2)/np.sum(area[:,:,None]*vel**2)))
        oldpsi,_=streamfunction(old,x)
        oldinterp=RectBivariateSpline(y,x,oldpsi)
        for v in vortices:
            v['psi_relative_change']=abs(v['psi']-float(oldinterp.ev(v['y'],v['x'])))/max(abs(v['psi']),1e-300)
    return result,(x,y,vel,psi,area)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--runs',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    records=[]; arrays={}; errors=[]
    for case in sorted(args.runs.glob('production-*'),key=lambda p:int(p.name.split('-')[-1])):
        if not (case/'solver-finished-unvalidated').exists(): continue
        try:
            record,data=audit(case); records.append(record); arrays[case.name]=data
        except Exception as exc:
            errors.append(dict(case=case.name,error=str(exc)))
    groups={}
    for r in records: groups.setdefault((r['re'],r['depth_over_width']),[]).append(r)
    grid=[]
    for key,rows in groups.items():
        rows.sort(key=lambda r:r['nx'])
        if len(rows)<2:continue
        fine=rows[-1]; x,y,v,psi,area=arrays[fine['case']]
        for coarse in rows[:-1]:
            xc,yc,vc,pc,ac=arrays[coarse['case']]
            xx,yy=np.meshgrid(xc,yc)
            vi=RegularGridInterpolator((y,x),v)(np.c_[yy.ravel(),xx.ravel()]).reshape(vc.shape)
            grid.append(dict(re=key[0],depth=key[1],coarse=coarse['nx'],fine=fine['nx'],
                velocity_relative_L2=float(np.sqrt(np.sum(ac[:,:,None]*(vc-vi)**2)/np.sum(ac[:,:,None]*vi**2)))))
    report=dict(reference='Cheng and Hung 2006 Table 2, DOI 10.1016/j.compfluid.2005.08.006',
        caveats=['Cell-centre trapezoidal streamfunction reconstruction; not a paper-discrete operator.',
                 'Vortex search excludes x<0.1 and x>0.9; this is a large-vortex audit, not certification of corner eddies.',
                 'Table 2 D7 Re1000 vortex2 psi=1.2206e-3 is retained as printed and flagged, not corrected.',
                 'No temporal-stability validation or formal GCI certification.'],cases=records,grid=grid,errors=errors)
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
