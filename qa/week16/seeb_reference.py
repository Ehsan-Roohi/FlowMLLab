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

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path); tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if json.loads(tmp.read_text())!=value:raise ValueError('JSON readback failed')
    tmp.replace(path)


def atomic_npz(path, **arrays):
    path=Path(path);tmp=path.with_name(path.stem+'.partial.npz')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    with np.load(tmp) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    tmp.replace(path)


def check_su2_mesh(path, expected_cells=None, expected_nodes=None):
    """Read every declared element, node and boundary record before launch."""
    with Path(path).open() as stream:
        def header(key):
            line=stream.readline()
            if not line.startswith(key+'='):raise ValueError(f'Missing {key} in {path}')
            return int(line.split('=',1)[1].split()[0])
        if header('NDIME')!=2:raise ValueError('Expected axisymmetric 2D meridian mesh')
        cells=header('NELEM');used_nodes=set()
        for _ in range(cells):
            row=stream.readline().split()
            if len(row)<5 or row[0]!='9':raise ValueError('Truncated/non-quadrilateral element')
            [int(v) for v in row];used_nodes.update(int(v) for v in row[1:5])
        nodes=header('NPOIN')
        if used_nodes!=set(range(nodes)):raise ValueError('Fluid connectivity does not cover exported node IDs')
        for _ in range(nodes):
            row=stream.readline().split()
            if len(row)!=3:raise ValueError('Truncated coordinate record')
            if not all(np.isfinite(float(v)) for v in row[:2]):raise ValueError('Nonfinite mesh coordinate')
            int(row[2])
        markers=header('NMARK');names=[];edges=0
        for _ in range(markers):
            line=stream.readline()
            if not line.startswith('MARKER_TAG='):raise ValueError('Missing boundary marker')
            names.append(line.split('=',1)[1].strip());count=header('MARKER_ELEMS');edges+=count
            for _ in range(count):
                row=stream.readline().split()
                if len(row)!=3 or row[0]!='3':raise ValueError('Truncated boundary edge')
                if any(not 0<=int(v)<nodes for v in row[1:]):raise ValueError('Invalid boundary node')
        if set(names)!={'body','axis','farfield'} or markers!=3:raise ValueError('Unexpected markers')
        if stream.read().strip():raise ValueError('Unexpected data after boundary records')
    if expected_cells is not None and cells!=expected_cells:raise ValueError('Element count mismatch')
    if expected_nodes is not None and nodes!=expected_nodes:raise ValueError('Node count mismatch')
    return {'cells':cells,'nodes':nodes,'boundary_edges':edges}


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
    x,r=cad_meridian(); atomic_npz(folder/'cad_meridian.npz',x=x,r=r)
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
        _,tags,connectivity=gmsh.model.mesh.getElements(2);tags=np.concatenate(tags)
        # getNodes() also includes the unused spline control-point entities.
        # SU2 exports the physical fluid mesh, whose nodes are defined by its
        # element connectivity; use that same set for all count checks.
        nodes=np.unique(np.concatenate(connectivity))
        all_gmsh_nodes=gmsh.model.mesh.getNodes()[0]
        quality=gmsh.model.mesh.getElementQualities(tags,'minSJ')
        result={'gmsh_version':gmsh.__version__,'cells':len(tags),'nodes':len(nodes),'gmsh_all_nodes':len(all_gmsh_nodes),'unused_geometry_nodes':len(all_gmsh_nodes)-len(nodes),'min_scaled_jacobian':float(min(quality)),'nose_x_over_L':xn,'nose_radius_over_L':rn,'tail_x_over_L':xt,'tail_radius_over_L':rt,'height_over_L':height,'end_over_L':end,'nose_cap_cells':nc,'shear_dx_dr':shear,'level':level,'cad_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
        if result['min_scaled_jacobian']<=0: raise ValueError('Invalid mesh')
        for attempt in range(3):
            temporary=folder/'mesh.partial.su2';gmsh.write(str(temporary))
            with temporary.open('rb') as stream:os.fsync(stream.fileno())
            try:
                check_su2_mesh(temporary,len(tags),len(nodes));break
            except ValueError:
                if attempt==2:raise
        temporary.replace(folder/'mesh.su2')
        temporary=folder/'mesh.partial.msh';gmsh.write(str(temporary))
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        if not temporary.read_bytes().rstrip().endswith(b'$EndElements'):raise ValueError('Incomplete Gmsh export')
        temporary.replace(folder/'mesh.msh')
        return result
    finally:gmsh.finalize()

def extract(folder):
    folder=Path(folder);u=load_csv(folder/'restart_flow.csv');rho=u['Density']
    for field in ['x','y','Density','Energy','Momentum_x','Momentum_y']:
        if not np.all(np.isfinite(u[field])):raise ValueError(f'Nonfinite field: {field}')
    if rho.min()<=0:raise ValueError('Nonpositive density')
    pressure=.4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/rho)
    if not np.all(np.isfinite(pressure)) or pressure.min()<=0:raise ValueError('Nonpositive pressure')
    interp=LinearNDInterpolator(np.c_[u['x'],u['y']],pressure)
    x_in=np.linspace(24.5,47,901)
    signal=(interp(x_in/L_IN,np.full_like(x_in,21.2/L_IN))-101325)/101325
    if not np.all(np.isfinite(signal)):raise ValueError('Sampling line outside mesh')
    atomic_npz(folder/'signature.npz',x_inches=x_in,dp_pinf=signal,height_inches=21.2)
    hist=load_csv(folder/'history.csv')
    if len(hist)<100:raise ValueError('Insufficient convergence history')
    for field in ['rms[Rho]','CD']:
        if not np.all(np.isfinite(hist[field])):raise ValueError('Nonfinite convergence history')
    tail=hist['CD'][-100:];mean=abs(float(np.mean(tail)))
    if mean<=0:raise ValueError('Cannot normalize force-stability metric')
    return {'iterations':len(hist),'density_residual_log10':float(hist['rms[Rho]'][-1]),
            'residual_drop':float(hist['rms[Rho]'][0]-hist['rms[Rho]'][-1]),
            'drag_tail_relative_range':float(np.ptp(tail)/mean),
            'min_pressure':float(pressure.min()),'min_density':float(rho.min())}


def verify_run(folder):
    """Check a completed run without trusting its summary alone."""
    folder=Path(folder);meta=json.loads((folder/'metadata.json').read_text())
    check_su2_mesh(folder/'mesh.su2',meta['cells'],meta['nodes'])
    for name in ['mesh.su2','flow.cfg','history.csv','restart_flow.csv','signature.npz']:
        if sha256(folder/name)!=meta['sha256'][name]:raise ValueError(f'Hash mismatch: {name}')
    hist=load_csv(folder/'history.csv')
    if len(hist)!=meta['iterations'] or float(hist['rms[Rho]'][-1])!=meta['density_residual_log10']:
        raise ValueError('History and metadata disagree')
    cfg=dict(line.split('=',1) for line in (folder/'flow.cfg').read_text().splitlines() if '=' in line and not line.lstrip().startswith('%'))
    cfg={k.strip():v.strip() for k,v in cfg.items()}
    if cfg['CONV_NUM_METHOD_FLOW']!=meta['convective_flux']:raise ValueError('Flux mismatch')
    if int(cfg['LIMITER_ITER'])!=meta['limiter_freeze_iteration']:raise ValueError('Limiter mismatch')
    if meta['fixed_cfl'] and (cfg['CFL_ADAPT']!='NO' or float(cfg['CFL_NUMBER'])!=meta['max_cfl']):
        raise ValueError('CFL metadata mismatch')
    u=load_csv(folder/'restart_flow.csv');rho=u['Density']
    if len(u)!=meta['nodes']:raise ValueError('Restart node count mismatch')
    pressure=.4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/rho)
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(rho)) or pressure.min()<=0 or rho.min()<=0:
        raise ValueError('Unphysical completed fields')
    if float(pressure.min())!=meta['min_pressure'] or float(rho.min())!=meta['min_density']:
        raise ValueError('Field and metadata minima disagree')
    return meta


def run(level=1.,iterations=8000,shear=1.,mesh_only=False,max_cfl=5.,threads=1,
        entropy=.05,flux='ROE',fixed_cfl=True,limiter_iter=2000,name=None):
    if level<=0 or threads<1 or max_cfl<=0:raise ValueError('Level, threads and CFL must be positive')
    if name is not None and Path(name).name!=name:raise ValueError('Name must be one directory name')
    folder=ROOT/'results/week16_lowboom/reference'/(name or f'seeb_level_{level:g}')
    folder.mkdir(parents=True,exist_ok=False)  # never overwrite previous evidence
    meta={'status':'meshing','converged':False,'level':level}
    try:
        meta.update(mesh(folder,level,shear))
        meta.update(mach=1.6,axisymmetric=True,model='Euler',geometry='NASA SEEB-ALR as-built sampled meridian with sting extension',
                    max_cfl=max_cfl,threads=threads,entropy_fix_coeff=entropy if flux=='ROE' else None,
                    convective_flux=flux,fixed_cfl=fixed_cfl,limiter_freeze_iteration=limiter_iter,
                    convergence_start_iteration=max(800,limiter_iter+500))
        config(folder,mach=1.6,iterations=iterations)
        cfg=folder/'flow.cfg'
        contents=cfg.read_text().replace('CONV_STARTITER= 500',f"CONV_STARTITER= {meta['convergence_start_iteration']}")
        contents=contents.replace('LIMITER_ITER= 300',f'LIMITER_ITER= {limiter_iter}').replace('CONV_NUM_METHOD_FLOW= ROE',f'CONV_NUM_METHOD_FLOW= {flux}')
        contents=contents.replace('CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, 100.0)',f'CFL_ADAPT_PARAM= (0.5, 1.5, 1.0, {max_cfl})')
        if fixed_cfl:contents=contents.replace('CFL_ADAPT= YES','CFL_ADAPT= NO').replace('CFL_NUMBER= 5.0',f'CFL_NUMBER= {max_cfl}')
        cfg.write_text(contents+f'ENTROPY_FIX_COEFF= {entropy}\n')
        with cfg.open('rb') as stream:os.fsync(stream.fileno())
        meta['source_sha256']={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('cfd.py'),Path(__file__).with_name('analyze.py')]}
        meta['sha256']={n:sha256(folder/n) for n in ['mesh.su2','flow.cfg','cad_meridian.npz']}
        # Legacy keys retained for report compatibility.
        for n in ['mesh.su2','flow.cfg']:meta[n+'_sha256']=meta['sha256'][n]
        meta['status']='mesh_only' if mesh_only else 'running';atomic_json(folder/'metadata.json',meta)
        if mesh_only:return meta
        start=time.time()
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.run([os.environ.get('SU2_CFD','SU2_CFD'),'flow.cfg','-t',str(threads)],cwd=folder,
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':'1'})
            log.flush();os.fsync(log.fileno())
        meta.update(returncode=proc.returncode,wall_seconds=time.time()-start)
        if proc.returncode:raise RuntimeError(f"SU2 exited {proc.returncode}: "+(folder/'solver.log').read_text()[-1500:])
        meta.update(extract(folder))
        meta['converged']=bool(meta['density_residual_log10']<=-9 and meta['residual_drop']>=5 and meta['drag_tail_relative_range']<1e-4)
        meta['sha256'].update({n:sha256(folder/n) for n in ['history.csv','restart_flow.csv','signature.npz','solver.log']})
        meta['status']='converged' if meta['converged'] else 'not_converged'
        atomic_json(folder/'metadata.json',meta);verify_run(folder)
        if not meta['converged']:raise RuntimeError('Numerical convergence gate failed; retained evidence is not validated')
        return meta
    except Exception as exc:
        meta['converged']=False
        if meta.get('status')!='not_converged':meta['status']='failed'
        meta['error']=str(exc);atomic_json(folder/'metadata.json',meta)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--level',type=float,default=1.);parser.add_argument('--iterations',type=int,default=8000)
    parser.add_argument('--shear',type=float,default=1.);parser.add_argument('--mesh-only',action='store_true')
    parser.add_argument('--max-cfl',type=float,default=5.);parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--entropy',type=float,default=.05);parser.add_argument('--flux',choices=['ROE','AUSMPLUSUP2','HLLC'],default='ROE')
    parser.add_argument('--fixed-cfl',action='store_true',default=True);parser.add_argument('--adaptive-cfl',dest='fixed_cfl',action='store_false')
    parser.add_argument('--limiter-iter',type=int,default=2000);parser.add_argument('--name')
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))
