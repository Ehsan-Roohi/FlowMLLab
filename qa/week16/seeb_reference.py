"""Rebuild NASA SEEB-ALR axisymmetric geometry, mesh and SU2 Euler evidence.

Run from repository root: python qa/week16/seeb_reference.py --level 1
Original CAD is kept unchanged. Mesh geometry is a densely sampled CAD meridian.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time
import numpy as np
from scipy.interpolate import PchipInterpolator, LinearNDInterpolator
from cfd import config
from analyze import load_csv
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'cases/week16_lowboom/reference/SEEB-ALR-as-built.stp'
L_IN=17.667

def cad_meridian():
    import gmsh
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.model.occ.importShapes(str(SOURCE));gmsh.model.occ.synchronize()
        candidates=[]
        for _,tag in gmsh.model.getEntities(1):
            bounds=gmsh.model.getParametrizationBounds(1,tag)
            u=np.linspace(float(bounds[0][0]),float(bounds[1][0]),2001)
            p=np.asarray(gmsh.model.getValue(1,tag,u)).reshape(-1,3)
            if np.ptp(p[:,0])>400 and np.max(abs(p[:,1]))<2e-5:
                candidates.append(p)
        if not candidates: raise ValueError('Expected x-z CAD meridian in millimetres not found')
        p=max(candidates,key=lambda p:np.ptp(p[:,0]));p=p[np.argsort(p[:,0])]
        x,idx=np.unique(p[:,0],return_index=True);r=np.sqrt(p[idx,1]**2+p[idx,2]**2)
        return x/(L_IN*25.4),r/(L_IN*25.4)
    finally: gmsh.finalize()

def mesh(folder,level=1.,shear=1.,height=2.,end=3.5):
    import gmsh
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    x,r=cad_meridian(); np.savez_compressed(folder/'cad_meridian.npz',x=x,r=r)
    f=PchipInterpolator(x,r);xn=float(x[0]);xt=float(x[-1]);rn=float(r[0]);rt=float(r[-1])
    if not 0<rn<.002 or not 1.5<xt<2.: raise ValueError('Unexpected CAD nose/tail dimensions')
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('SEEB_ALR');g=gmsh.model.geo
    try:
        # Refine the leading 8% of the body separately. Geometric progression
        # on these curves puts extra streamwise resolution near the finite nose.
        xs=[-.5,xn,xn+.08,xt,end]
        rs=[0,rn,float(f(xn+.08)),rt,rt]
        lo=[g.addPoint(v,w,0) for v,w in zip(xs,rs)]
        axisnose=g.addPoint(xn,0,0)
        hi=[g.addPoint(v+shear*(height-w),height,0) for v,w in zip(xs,rs)]
        radial=[g.addLine(lo[i],hi[i]) for i in range(len(xs))]
        axis=g.addLine(lo[0],axisnose);cap=g.addLine(axisnose,lo[1])
        bottom=[axis]
        for a,b in [(1,2),(2,3)]:
            pts=[lo[a]]+[g.addPoint(float(z),float(f(z)),0) for z in np.linspace(xs[a],xs[b],401)[1:-1]]+[lo[b]]
            bottom.append(g.addSpline(pts))
        bottom.append(g.addLine(lo[3],lo[4]))
        top=[g.addLine(hi[i],hi[i+1]) for i in range(4)]
        surfaces=[]
        first=g.addPlaneSurface([g.addCurveLoop([axis,cap,radial[1],-top[0],-radial[0]])]);surfaces.append(first)
        for i in range(1,4): surfaces.append(g.addPlaneSurface([g.addCurveLoop([bottom[i],radial[i+1],-top[i],-radial[i]])]))
        g.synchronize()
        nr=round(160*level);nc=2
        for i,c in enumerate(radial):gmsh.model.mesh.setTransfiniteCurve(c,nr+nc+1 if i==0 else nr+1,'Progression',1.025**(1/level))
        gmsh.model.mesh.setTransfiniteCurve(cap,nc+1)
        nx=[round(v*level) for v in (100,100,300,220)]
        for i in range(4):
            for c in [bottom[i],top[i]]:gmsh.model.mesh.setTransfiniteCurve(c,nx[i]+1,'Progression',1.015**(1/level) if i==1 else 1.)
            corners=[lo[i],axisnose if i==0 else lo[i+1],hi[i+1],hi[i]]
            gmsh.model.mesh.setTransfiniteSurface(surfaces[i],'Left',corners);gmsh.model.mesh.setRecombine(2,surfaces[i])
        for name,curves in [('body',[cap]+bottom[1:]),('axis',[axis]),('farfield',top+[radial[0],radial[-1]])]:
            t=gmsh.model.addPhysicalGroup(1,curves);gmsh.model.setPhysicalName(1,t,name)
        gmsh.model.addPhysicalGroup(2,surfaces,1);gmsh.model.mesh.generate(2)
        gmsh.write(str(folder/'mesh.su2'));gmsh.write(str(folder/'mesh.msh'))
        _,tags,_=gmsh.model.mesh.getElements(2);tags=np.concatenate(tags)
        quality=gmsh.model.mesh.getElementQualities(tags,'minSJ');nodes=gmsh.model.mesh.getNodes()[0]
        result={'cells':len(tags),'nodes':len(nodes),'min_scaled_jacobian':float(min(quality)),'nose_x_over_L':xn,'nose_radius_over_L':rn,'tail_x_over_L':xt,'tail_radius_over_L':rt,'height_over_L':height,'end_over_L':end,'nose_cap_cells':nc,'shear_dx_dr':shear,'level':level,'cad_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
        if result['min_scaled_jacobian']<=0: raise ValueError('Invalid mesh')
        return result
    finally:gmsh.finalize()

def extract(folder):
    folder=Path(folder);u=load_csv(folder/'restart_flow.csv');rho=u['Density']
    p=.4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/rho)
    if not np.all(np.isfinite(p)) or p.min()<=0 or rho.min()<=0:raise ValueError('Unphysical pressure/density')
    interp=LinearNDInterpolator(np.c_[u['x'],u['y']],p)
    x_in=np.linspace(24.5,47,901);signal=(interp(x_in/L_IN,np.full_like(x_in,21.2/L_IN))-101325)/101325
    if not np.all(np.isfinite(signal)):raise ValueError('Sampling line outside domain')
    np.savez_compressed(folder/'signature.npz',x_inches=x_in,dp_pinf=signal,height_inches=21.2)
    hist=load_csv(folder/'history.csv')
    return {'iterations':len(hist),'density_residual_log10':float(hist['rms[Rho]'][-1]),'residual_drop':float(hist['rms[Rho]'][0]-hist['rms[Rho]'][-1]),'min_pressure':float(p.min()),'min_density':float(rho.min())}

def run(level=1.,iterations=1800,shear=1.,mesh_only=False,max_cfl=20.,threads=1):
    folder=ROOT/'results/week16_lowboom/reference'/f'seeb_level_{level:g}'
    meta=mesh(folder,level,shear);meta.update(mach=1.6,axisymmetric=True,model='Euler',geometry='NASA SEEB-ALR as-built sampled meridian with sting extension')
    config(folder,mach=1.6,iterations=iterations)
    cfg=folder/'flow.cfg';cfg.write_text(cfg.read_text().replace('CONV_STARTITER= 500','CONV_STARTITER= 800').replace('CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, 100.0)',f'CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, {max_cfl})')+'ENTROPY_FIX_COEFF= 0.001\n')
    meta['max_cfl']=max_cfl
    meta['threads']=threads
    meta['status']='meshed' if mesh_only else 'running'
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if not mesh_only:
        start=time.time()
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.run([os.environ.get('SU2_CFD','SU2_CFD'),'flow.cfg','-t',str(threads)],cwd=folder,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':'1'})
        meta.update(returncode=proc.returncode,wall_seconds=time.time()-start)
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        if proc.returncode:raise RuntimeError((folder/'solver.log').read_text()[-4000:])
        meta.update(extract(folder));meta['converged']=bool(meta['density_residual_log10']<=-9 and meta['residual_drop']>=5)
    meta['status']='mesh_only' if mesh_only else ('converged' if meta.get('converged') else 'not_converged')
    for name in ['mesh.su2','flow.cfg']:meta[name+'_sha256']=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--level',type=float,default=1.);parser.add_argument('--iterations',type=int,default=1800);parser.add_argument('--shear',type=float,default=1.);parser.add_argument('--mesh-only',action='store_true');parser.add_argument('--max-cfl',type=float,default=20.);parser.add_argument('--threads',type=int,default=1)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
