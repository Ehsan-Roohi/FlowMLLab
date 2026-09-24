"""Uniform mesh refinement of the unchanged Taylor-Maccoll cone, SU2 8.0.1.

The original 24k-cell result remains in the report even if its Cp test fails.
Only the finest Cp error and last-two-level Cp change control family accuracy.
"""
from pathlib import Path
import argparse,json,os,subprocess,time
import numpy as np
import cfd
from benchmark import cone_exact
from analyze import load_csv
from weakwall_cfd import physical_checks,sha
from install_su2_801 import BINARY_SHA256,ASSET_SHA256,URL
ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'results/week16_lowboom/runs'
REF=ROOT/'results/week16_lowboom/reference'
SOURCES=['cone_refinement_v801.py','benchmark.py','cfd.py','analyze.py','weakwall_cfd.py','install_su2_801.py']


def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if json.loads(tmp.read_text())!=value:raise ValueError('JSON readback failed')
    tmp.replace(path)


def equivalent(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
    return a==b


def mesh_counts(path):
    with Path(path).open() as f:
        def head(key):
            line=f.readline()
            if not line.startswith(key+'='):raise ValueError(f'Incomplete mesh: {key}')
            return int(line.split('=')[1].split()[0])
        assert head('NDIME')==2
        nc=head('NELEM');used=set()
        for _ in range(nc):
            row=f.readline().split();assert row[0]=='9' and len(row)>=5
            used.update(map(int,row[1:5]))
        nn=head('NPOIN');assert used==set(range(nn))
        for i in range(nn):
            row=f.readline().split();assert len(row)==3 and np.isfinite([float(z) for z in row[:2]]).all()
        nm=head('NMARK');names=[]
        for _ in range(nm):
            line=f.readline();assert line.startswith('MARKER_TAG=');names.append(line.split('=')[1].strip())
            n=head('MARKER_ELEMS')
            for _ in range(n):
                row=f.readline().split();assert len(row)==3 and row[0]=='3'
                assert all(0<=int(z)<nn for z in row[1:])
        assert nm==2 and set(names)=={'body','farfield'} and not f.read().strip()
    return {'cells':nc,'nodes':nn}


def make_mesh(folder,level):
    import gmsh
    gmsh.initialize();gmsh.option.setNumber('General.Terminal',0)
    try:
        gmsh.model.add('unchanged_seven_degree_cone');g=gmsh.model.geo
        points=[g.addPoint(*p) for p in [(0,0,0),(1,float(np.tan(np.deg2rad(7))),0),(1,1,0),(0,1,0)]]
        lines=[g.addLine(points[i],points[(i+1)%4]) for i in range(4)]
        surface=g.addPlaneSurface([g.addCurveLoop(lines)]);g.synchronize()
        for i,line in enumerate(lines):gmsh.model.mesh.setTransfiniteCurve(line,(200 if i%2==0 else 120)*level+1)
        gmsh.model.mesh.setTransfiniteSurface(surface);gmsh.model.mesh.setRecombine(2,surface)
        for name,curves in [('body',[lines[0]]),('farfield',lines[1:])]:
            tag=gmsh.model.addPhysicalGroup(1,curves);gmsh.model.setPhysicalName(1,tag,name)
        gmsh.model.addPhysicalGroup(2,[surface],1);gmsh.model.mesh.generate(2)
        _,tags,connections=gmsh.model.mesh.getElements(2);tags=np.concatenate(tags);nodes=np.unique(np.concatenate(connections))
        quality=float(np.min(gmsh.model.mesh.getElementQualities(tags,'minSJ')))
        if quality<=0:raise ValueError('Invalid cone mesh')
        tmp=folder/'mesh.partial.su2';gmsh.write(str(tmp))
        with tmp.open('rb') as f:os.fsync(f.fileno())
        counts=mesh_counts(tmp)
        assert counts=={'cells':24000*level**2,'nodes':len(nodes)}
        tmp.replace(folder/'mesh.su2')
        return dict(counts,min_scaled_jacobian=quality,gmsh_version=gmsh.__version__,streamwise_cells=200*level,radial_cells=120*level)
    finally:gmsh.finalize()


def inspect(folder,level):
    folder=Path(folder);counts=mesh_counts(folder/'mesh.su2')
    assert counts['cells']==24000*level**2
    history=load_csv(folder/'history.csv');tail=history['CD'][-100:]
    finite=bool(np.isfinite(history['rms[Rho]']).all() and len(tail)==100 and np.isfinite(tail).all() and abs(tail.mean())>0)
    if not finite:raise ValueError('Invalid cone convergence history')
    residual=float(history['rms[Rho]'][-1]);drop=float(history['rms[Rho]'][0]-residual)
    stability=float(np.ptp(tail)/abs(tail.mean()))
    numerical={'iterations':len(history),'density_residual_log10':residual,'residual_drop':drop,'drag_tail_relative_range':stability,
               'checks':{'density_residual_at_most_minus9':residual<=-9,'density_residual_drop_at_least5':drop>=5,'drag_tail_relative_range_below_1e_minus4':stability<1e-4}}
    numerical['passed']=all(numerical['checks'].values())
    assert len(load_csv(folder/'restart_flow.csv'))==counts['nodes']
    physical=physical_checks(folder)
    surface=load_csv(folder/'surface_flow.csv');mask=(surface['x']>.3)&(surface['x']<.8)
    if not mask.any():raise ValueError('No cone pressure samples in unchanged comparison interval')
    pressure=.4*(surface['Energy']-.5*(surface['Momentum_x']**2+surface['Momentum_y']**2)/surface['Density'])
    cp=(pressure-101325)/(.5*1.4*101325*1.8**2)
    if not np.isfinite(cp).all():raise ValueError('Nonfinite surface Cp')
    exact=cone_exact();measured=float(np.mean(cp[mask]));error=float(abs(measured/exact['cp']-1))
    return {'level':level,'mesh':counts,'numerical':numerical,'physical':physical,
            'analytical_comparison':{'exact':exact,'cfd_cp_mean':measured,'relative_error':error,'passed':error<.03,
                                     'sampling':'Arithmetic mean at original surface nodes with 0.3<x<0.8; unchanged from benchmark.py','samples':int(mask.sum())}}


def run(level,iterations=4000):
    if level not in (2,3):raise ValueError('Only predeclared refinement levels 2 and 3')
    exe=Path(os.environ.get('SU2_CFD',str(ROOT/'.tools/week16_su2_801/bin/SU2_CFD'))).resolve()
    if sha(exe)!=BINARY_SHA256:raise ValueError('Solver is not the pinned official SU2 8.0.1 binary')
    folder=RUNS/f'cone_v801_level_{level}';folder.mkdir(parents=True,exist_ok=False)
    source={n:sha(ROOT/'qa/week16'/n) for n in SOURCES}
    evidence={'status':'meshing','solver_version':'8.0.1','solver_binary_sha256':BINARY_SHA256,'official_asset_sha256':ASSET_SHA256,
              'official_asset_url':URL,'source_sha256':source,'level':level,'iteration_ceiling':iterations,
              'method':'Original benchmark geometry, domain, uniform transfinite topology, Mach1.8, cone7deg and solver config. Only uniform mesh density and predeclared iteration ceiling differ.'}
    try:
        evidence['mesh']=make_mesh(folder,level)
        cfd.config(folder,1.8,iterations)
        cfg=folder/'flow.cfg';cfg.write_text(cfg.read_text().replace('MARKER_SYM= (axis)','MARKER_SYM= (NONE)'))
        with cfg.open('rb') as f:os.fsync(f.fileno())
        evidence['status']='running';save(folder/'run_evidence.json',evidence)
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'};start=time.time()
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.run([str(exe),'flow.cfg'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.flush();os.fsync(log.fileno())
        evidence.update(returncode=proc.returncode,wall_seconds=time.time()-start)
        if proc.returncode:raise RuntimeError('SU2 failed; see retained solver.log')
        result=inspect(folder,level);evidence['checks']=result
        save(folder/'benchmark.json',result['analytical_comparison'])
        evidence['status']='completed'
        evidence['numerical_and_physical_passed']=bool(result['numerical']['passed'] and result['physical']['passed'])
        if not evidence['numerical_and_physical_passed']:raise RuntimeError('Cone numerical or physical check failed')
    except BaseException as exc:
        evidence['status']='failed';evidence['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        evidence['source_unchanged']=source=={n:sha(ROOT/'qa/week16'/n) for n in SOURCES}
        evidence['raw_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name not in ['run_evidence.json','run_evidence.json.partial']}
        save(folder/'run_evidence.json',evidence)
        assert evidence['source_unchanged'],'Immutable generating sources changed'
    return evidence


def build_report(write=True,coarse_folder=None,coarse_report=None):
    coarse_folder=Path(coarse_folder or RUNS/'cone_benchmark')
    coarse_report=Path(coarse_report or REF/'cone_verification_v801.json')
    old=json.loads(coarse_report.read_text());assert old['binary_sha256']==BINARY_SHA256
    for name,value in old['raw_sha256'].items():assert sha(coarse_folder/name)==value,f'Changed archived coarse file {name}'
    for name,value in old['source_sha256'].items():assert sha(ROOT/'qa/week16'/name)==value,f'Changed coarse source {name}'
    rows=[inspect(coarse_folder,1)]
    for level in (2,3):
        folder=RUNS/f'cone_v801_level_{level}';saved=json.loads((folder/'run_evidence.json').read_text())
        assert saved['solver_binary_sha256']==BINARY_SHA256 and saved['returncode']==0
        for name,value in saved['raw_sha256'].items():assert sha(folder/name)==value,f'Changed raw file {name}'
        for name,value in saved['source_sha256'].items():assert sha(ROOT/'qa/week16'/name)==value,f'Changed generating source {name}'
        row=inspect(folder,level);assert equivalent(row,saved['checks']),'Stale refinement report'
        rows.append(row)
    cp2=rows[1]['analytical_comparison']['cfd_cp_mean'];cp3=rows[2]['analytical_comparison']['cfd_cp_mean']
    change=float(abs(cp2-cp3)/abs(cp3))
    checks={'all_three_levels_numerically_converged':all(r['numerical']['passed'] for r in rows),
            'all_three_levels_physically_admissible':all(r['physical']['passed'] for r in rows),
            'finest_analytical_cp_error_below_3pct':rows[-1]['analytical_comparison']['relative_error']<.03,
            'last_two_mean_cp_change_below_3pct':change<.03}
    report={'case':'Mach1.8 seven-degree axisymmetric cone','solver_version':'8.0.1','solver_binary_sha256':BINARY_SHA256,
            'levels':rows,'last_two_cp_relative_change':change,'checks':checks,'passed':all(checks.values()),
            'thresholds':{'finest_analytical_cp_error_max':.03,'last_two_cp_relative_change_max':.03},
            'scope':'Independent Taylor-Maccoll pressure verification. Original coarse Cp failure retained; refinement sensitivity is not a formal GCI.'}
    if write:REF.mkdir(parents=True,exist_ok=True);save(REF/'cone_refinement_v801.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--level',type=int,choices=[2,3]);group.add_argument('--report',action='store_true')
    p.add_argument('--iterations',type=int,choices=[2000,4000],default=4000)
    p.add_argument('--coarse-folder',type=Path);p.add_argument('--coarse-report',type=Path)
    a=p.parse_args();result=build_report(coarse_folder=a.coarse_folder,coarse_report=a.coarse_report) if a.report else run(a.level,a.iterations)
    print(json.dumps(result,indent=2))
    if a.report and not result['passed']:raise SystemExit('Cone refinement acceptance gate failed')
