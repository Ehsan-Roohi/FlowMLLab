"""SEEB-ALR mesh with resolved finite cap and local upstream shock region.

Uses unchanged geometry/solver extraction helpers; this mesh family is separate
from previous coarse-nose pilots. A mesh-only run must precede CFD validation.
"""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
import seeb_reference as base


def mesh(folder,level=1.,shear=1.,height=2.,end=3.5):
    import gmsh
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    x,r=base.cad_meridian();base.atomic_npz(folder/'cad_meridian.npz',x=x,r=r)
    f=PchipInterpolator(x,r);xn=float(x[0]);xt=float(x[-1]);rn=float(r[0]);rt=float(r[-1])
    target=rn/12
    # Grading is selected geometrically, without consulting pressure data.
    def end_spacing(q):return (xn+.5)*q**199/np.sum(q**np.arange(200))
    upstream_q=brentq(lambda q:end_spacing(q)-target,.8,.999999)
    def first_spacing(q):return .08/np.sum(q**np.arange(200))
    leading_q=brentq(lambda q:first_spacing(q)-target,1.000001,1.1)
    nx=[round(v*level) for v in (200,200,300,220)]
    nr=round(310*level);nc=round(10*level)
    if min(nx)<2 or nr<2 or nc<2:raise ValueError('Refinement level too small')
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0);gmsh.model.add('SEEB_resolved_cap');g=gmsh.model.geo
    try:
        xs=[-.5,xn,xn+.08,xt,end];rs=[0,rn,float(f(xn+.08)),rt,rt]
        lo=[g.addPoint(v,w,0) for v,w in zip(xs,rs)];axisnose=g.addPoint(xn,0,0)
        hi=[g.addPoint(v+shear*(height-w),height,0) for v,w in zip(xs,rs)]
        radial=[g.addLine(lo[i],hi[i]) for i in range(5)]
        axis=g.addLine(lo[0],axisnose);cap=g.addLine(axisnose,lo[1]);bottom=[axis]
        for a,b in [(1,2),(2,3)]:
            pts=[lo[a]]+[g.addPoint(float(z),float(f(z)),0) for z in np.linspace(xs[a],xs[b],401)[1:-1]]+[lo[b]]
            bottom.append(g.addSpline(pts))
        bottom.append(g.addLine(lo[3],lo[4]));top=[g.addLine(hi[i],hi[i+1]) for i in range(4)]
        surfaces=[g.addPlaneSurface([g.addCurveLoop([axis,cap,radial[1],-top[0],-radial[0]])])]
        for i in range(1,4):surfaces.append(g.addPlaneSurface([g.addCurveLoop([bottom[i],radial[i+1],-top[i],-radial[i]])]))
        g.synchronize()
        for i,c in enumerate(radial):gmsh.model.mesh.setTransfiniteCurve(c,nr+nc+1 if i==0 else nr+1,'Progression',1.025**(1/level))
        gmsh.model.mesh.setTransfiniteCurve(cap,nc+1)
        for i in range(4):
            q=upstream_q if i==0 else leading_q if i==1 else 1.
            for c in [bottom[i],top[i]]:gmsh.model.mesh.setTransfiniteCurve(c,nx[i]+1,'Progression',q**(1/level))
            corners=[lo[i],axisnose if i==0 else lo[i+1],hi[i+1],hi[i]]
            gmsh.model.mesh.setTransfiniteSurface(surfaces[i],'Left',corners);gmsh.model.mesh.setRecombine(2,surfaces[i])
        for name,curves in [('body',[cap]+bottom[1:]),('axis',[axis]),('farfield',top+[radial[0],radial[-1]])]:
            tag=gmsh.model.addPhysicalGroup(1,curves);gmsh.model.setPhysicalName(1,tag,name)
        gmsh.model.addPhysicalGroup(2,surfaces,1);gmsh.model.mesh.generate(2)
        _,tags,connectivity=gmsh.model.mesh.getElements(2);tags=np.concatenate(tags);nodes=np.unique(np.concatenate(connectivity))
        quality=gmsh.model.mesh.getElementQualities(tags,'minSJ')
        def coordinates(curve):
            return np.asarray(gmsh.model.mesh.getNodes(1,curve,includeBoundary=True)[1]).reshape(-1,3)
        axisx=np.sort(coordinates(axis)[:,0]);leadx=np.sort(coordinates(bottom[1])[:,0]);capr=np.sort(coordinates(cap)[:,1]);radialr=np.sort(coordinates(radial[1])[:,1])
        spacing={'upstream_last_dx_over_R':float((axisx[-1]-axisx[-2])/rn),
                 'leading_body_first_dx_over_R':float((leadx[1]-leadx[0])/rn),
                 'first_off_wall_dr_over_R':float((radialr[1]-radialr[0])/rn),
                 'max_cap_dr_over_R':float(np.diff(capr).max()/rn),
                 'upstream_points_within_one_R_of_nose':int(np.sum(axisx>xn-rn)),
                 'upstream_points_within_five_R_of_nose':int(np.sum(axisx>xn-5*rn))}
        if min(quality)<=0:raise ValueError('Invalid resolved mesh')
        if max(spacing[k] for k in ['upstream_last_dx_over_R','leading_body_first_dx_over_R','first_off_wall_dr_over_R'])>.1/level*1.005:
            raise ValueError('Local spacing is too coarse for prescribed R/10 target')
        result={'mesh_family':'finite_cap_resolved_v1','gmsh_version':gmsh.__version__,'cells':len(tags),'nodes':len(nodes),
                'min_scaled_jacobian':float(min(quality)),'level':level,'nose_x_over_L':xn,'nose_radius_over_L':rn,
                'tail_x_over_L':xt,'tail_radius_over_L':rt,'nose_cap_cells':nc,'height_over_L':height,'end_over_L':end,
                'shear_dx_dr':shear,'upstream_progression_base':upstream_q,'leading_progression_base':leading_q,
                'radial_progression_base':1.025,'spacing_diagnostics':spacing,'cad_sha256':base.sha256(base.SOURCE)}
        for attempt in range(3):
            tmp=folder/'mesh.partial.su2';gmsh.write(str(tmp))
            with tmp.open('rb') as stream:os.fsync(stream.fileno())
            try:base.check_su2_mesh(tmp,len(tags),len(nodes));break
            except ValueError:
                if attempt==2:raise
        tmp.replace(folder/'mesh.su2');tmp=folder/'mesh.partial.msh';gmsh.write(str(tmp))
        with tmp.open('rb') as stream:os.fsync(stream.fileno())
        if not tmp.read_bytes().rstrip().endswith(b'$EndElements'):raise ValueError('Incomplete Gmsh file')
        tmp.replace(folder/'mesh.msh')
        return result
    finally:gmsh.finalize()


def physical_audit(folder):
    """Check thermodynamic bounds and stagnation pressure independently of residuals."""
    folder=Path(folder);cfg={}
    for line in (folder/'flow.cfg').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('%'):
            k,v=line.split('=',1);cfg[k.strip()]=v.strip()
    gamma=1.4;gas_constant=287.058
    pinf=float(cfg['FREESTREAM_PRESSURE']);tinf=float(cfg['FREESTREAM_TEMPERATURE']);mach=float(cfg['MACH_NUMBER'])
    rhoinf=pinf/(gas_constant*tinf);factor=1+.5*(gamma-1)*mach**2
    h0inf=gamma/(gamma-1)*gas_constant*tinf*factor
    rho0=rhoinf*factor**(1/(gamma-1))
    m2sq=(1+.5*(gamma-1)*mach**2)/(gamma*mach**2-.5*(gamma-1))
    expected=(1+2*gamma/(gamma+1)*(mach**2-1))*(1+.5*(gamma-1)*m2sq)**(gamma/(gamma-1))
    u=base.load_csv(folder/'restart_flow.csv')
    for field in ['x','y','Density','Momentum_x','Momentum_y','Energy']:
        if not np.all(np.isfinite(u[field])):raise ValueError(f'Nonfinite physical-audit field {field}')
    rho=u['Density']
    if np.min(rho)<=0:raise ValueError('Nonpositive density in physical audit')
    speed2=(u['Momentum_x']**2+u['Momentum_y']**2)/rho**2
    pressure=(gamma-1)*(u['Energy']-.5*rho*speed2)
    if not np.isfinite(pressure).all() or pressure.min()<=0:raise ValueError('Invalid pressure in physical audit')
    h0=gamma/(gamma-1)*pressure/rho+.5*speed2
    with np.load(folder/'cad_meridian.npz') as cad:xnose=float(cad['x'][0]);rnose=float(cad['r'][0])
    axis=np.flatnonzero((abs(u['x']-xnose)<1e-10)&(abs(u['y'])<1e-10))
    if len(axis)!=1:raise ValueError('Nose-axis junction is not uniquely represented')
    i=int(axis[0]);cap=np.flatnonzero((abs(u['x']-xnose)<1e-10)&(u['y']>=0)&(u['y']<=rnose+1e-10))
    cap=cap[np.argsort(u['y'][cap])]
    stagnation_ratio=float(pressure[i]/pinf);stagnation_error=abs(stagnation_ratio/expected-1)
    h_error=float(np.max(abs(h0/h0inf-1)));rho_ratio=float(np.max(rho/rho0))
    checks={'stagnation_pressure_within_10pct':bool(stagnation_error<=.10),
            'global_total_enthalpy_within_10pct':bool(h_error<=.10),
            'density_below_1p1_isentropic_stagnation_bound':bool(rho_ratio<=1.1)}
    return {'passed':all(checks.values()),'checks':checks,
            'scope':'Physical plausibility checks; distinct from numerical convergence and experimental waveform validation.',
            'gamma':gamma,'gas_constant_J_kg_K':gas_constant,'freestream_pressure_Pa':pinf,'freestream_temperature_K':tinf,'mach':mach,
            'reference_h0_J_kg':h0inf,'reference_isentropic_stagnation_density_kg_m3':rho0,
            'normal_shock_stagnation_pressure_ratio_expected':expected,
            'nose_stagnation_pressure_ratio_actual':stagnation_ratio,'stagnation_pressure_relative_error':float(stagnation_error),
            'global_h0_normalized_min':float(np.min(h0/h0inf)),'global_h0_normalized_max':float(np.max(h0/h0inf)),
            'global_max_abs_h0_relative_deviation':h_error,'global_max_rho_over_isentropic_stagnation_density':rho_ratio,
            'cap_samples':[{'r_over_nose_radius':float(u['y'][j]/rnose),'p_over_pinf':float(pressure[j]/pinf),
                            'u_m_s':float(u['Momentum_x'][j]/rho[j]),'v_m_s':float(u['Momentum_y'][j]/rho[j])} for j in cap],
            'thresholds':{'stagnation_pressure_relative_error_max':.10,'global_h0_relative_deviation_max':.10,'rho_over_isentropic_stagnation_max':1.1},
            'input_sha256':{n:base.sha256(folder/n) for n in ['flow.cfg','restart_flow.csv','cad_meridian.npz']}}


def run(level=1.,threads=4,iterations=8000,mesh_only=False,name=None):
    name=name or f'seeb_resolved_level_{level:g}'
    folder=base.ROOT/'results/week16_lowboom/reference'/name
    if folder.exists():raise FileExistsError(f'Choose a fresh output name: {folder}')
    sources=[Path(base.__file__),Path(base.__file__).with_name('cfd.py'),Path(base.__file__).with_name('analyze.py')]
    before={p.name:base.sha256(p) for p in sources};own_hash=base.sha256(__file__)
    original_mesh=base.mesh;original_config=base.config
    def checkpoint_config(folder,*args,**kwargs):
        original_config(folder,*args,**kwargs)
        path=Path(folder)/'flow.cfg'
        lines=path.read_text().splitlines()
        lines=['OUTPUT_WRT_FREQ= 500' if line.startswith('OUTPUT_WRT_FREQ=') else line for line in lines]
        with path.open('w') as stream:
            stream.write('\n'.join(lines)+'\n');stream.flush();os.fsync(stream.fileno())
    base.mesh=mesh;base.config=checkpoint_config
    failure=None;result=None;audit=None
    try:
        result=base.run(level=level,threads=threads,iterations=iterations,mesh_only=mesh_only,name=name,
                        max_cfl=5.,fixed_cfl=True,entropy=.05,flux='ROE',limiter_iter=round(2000*level))
    except BaseException as exc:
        failure=exc
    finally:
        base.mesh=original_mesh;base.config=original_config
        meta=folder/'metadata.json'
        if meta.exists():
            value=json.loads(meta.read_text());value['resolved_driver_sha256']=own_hash
            value['resolved_driver_file']='seeb_resolved.py';value.setdefault('source_sha256',{})['seeb_resolved.py']=own_hash
            value['full_reference_validation_at_generation']=False
            value['scope']='New mesh family; convergence, physical plausibility and experimental validation are separate requirements.'
            if not mesh_only:
                try:audit=physical_audit(folder)
                except Exception as exc:audit={'passed':False,'checks':{'physical_fields_available_and_readable':False},'error':str(exc)}
                base.atomic_json(folder/'physical_audit.json',audit)
                value['physical_plausibility_passed']=audit['passed']
                value['physical_audit_sha256']=base.sha256(folder/'physical_audit.json')
                if value.get('converged') and not audit['passed']:value['status']='physical_validation_failed'
            base.atomic_json(meta,value);result=value
        assert before=={p.name:base.sha256(p) for p in sources},'Base sources changed'
    if failure is not None:raise failure
    if not mesh_only and (audit is None or not audit['passed']):
        raise RuntimeError('Physical plausibility gate failed; see retained physical_audit.json')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--level',type=float,default=1.);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--iterations',type=int,default=8000);p.add_argument('--name');p.add_argument('--mesh-only',action='store_true')
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
