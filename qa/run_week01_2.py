"""Execute the initial Re=100 comparison; never alter Week 1 assets."""
from pathlib import Path
import hashlib,json,sys,time,platform
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flowmllab'))
sys.path.insert(0,str(ROOT/'common'))
from pressure_velocity import CavityConfig,run_cavity,ghia_errors
from w4utils import run_cavity as streamfunction_cavity

OUT=ROOT/'results/week01_2_pressure_velocity'
OUT.mkdir(parents=True,exist_ok=True)


def save(name,result):
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(OUT/(name+'.npz'),**arrays)
    record={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
    record['field_sha256']=hashlib.sha256((OUT/(name+'.npz')).read_bytes()).hexdigest()
    (OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    return {'name':name,**record}


def main():
    rows=[]
    summarize_only='--summarize-only' in sys.argv
    execution_versions={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}
    if summarize_only:
        existing=json.loads((OUT/'manifest.json').read_text())
        for path,key in [(ROOT/'flowmllab/pressure_velocity.py','source_sha256'),(ROOT/'common/w4utils.py','reference_source_sha256')]:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==existing[key], 'Source changed: recompute results instead of summarizing'
        execution_versions={key:existing[key] for key in execution_versions}
    for n in [32,64]:
        for method in ['simple','piso','pimple']:
            name=f'{method}-re100-n{n}'
            if summarize_only:
                record=json.loads((OUT/(name+'.json')).read_text())
                assert hashlib.sha256((OUT/(name+'.npz')).read_bytes()).hexdigest()==record['field_sha256']
                assert record['converged'] and record['momentum_residual_linf']<1e-7 and record['continuity_linf']<1e-9
                rows.append({'name':name,**record})
                continue
            r=run_cavity(CavityConfig(method=method,cells=n,max_iterations=4000),progress=True)
            assert r['converged'],(name,'not converged',r['momentum_residual_linf'])
            assert r['continuity_linf']<1e-9 and abs(r['p'].mean())<1e-12
            assert max(r['ghia_u_relative_l2'],r['ghia_v_relative_l2'])<.05
            rows.append(save(name,r))
            print('COMPLETED',name,r['iterations'],r['runtime_seconds'],flush=True)
        # Reuse the public Week-1-family reference implementation unchanged.
        if summarize_only:
            name=f'streamfunction-re100-n{n}'
            record=json.loads((OUT/(name+'.json')).read_text())
            assert hashlib.sha256((OUT/(name+'.npz')).read_bytes()).hexdigest()==record['field_sha256']
            assert record['converged'];rows.append({'name':name,**record})
            continue
        start=time.perf_counter()
        r=streamfunction_cavity(Re=100,N=n+1,dt=.001,max_steps=60000,
            check_every=500,min_steps=5000,tol=1e-8,consecutive_required=3,verbose=False)
        r['runtime_seconds']=time.perf_counter()-start
        assert r['converged'],('streamfunction',n)
        r.update(config={'method':'streamfunction-vorticity','cells':n,'nodes':n+1,
                         'reynolds':100,'dt':.001,'space':'central finite difference'},**ghia_errors(r))
        rows.append(save(f'streamfunction-re100-n{n}',r))
        print('COMPLETED streamfunction',n,r['steps'],r['runtime_seconds'],flush=True)
    # Matched steady discrete equations must agree before comparing efficiency.
    comparisons=[]
    for n in [32,64]:
        fields={m:np.load(OUT/f'{m}-re100-n{n}.npz') for m in ['simple','piso','pimple']}
        for method in ['piso','pimple']:
            errors={k:float(np.max(np.abs(fields[method][k]-fields['simple'][k]))) for k in ['u','v','p']}
            assert max(errors.values())<2e-6,(n,method,errors)
            comparisons.append({'cells':n,'method':method,'relative_to':'simple','linf':errors})
        for f in fields.values():f.close()
    trends=[]
    for method in ['simple','piso','pimple','streamfunction-vorticity']:
        selected=[r for r in rows if r['config']['method']==method]
        trends.append({'method':method,**{key+'_decreases':selected[1][key]<selected[0][key] for key in ['ghia_u_relative_l2','ghia_v_relative_l2']}})
    manifest={'date':'2026-10-09','reynolds':100,'grids_cells':[32,64],
        **execution_versions,
        'source_sha256':hashlib.sha256((ROOT/'flowmllab/pressure_velocity.py').read_bytes()).hexdigest(),
        'reference_source_sha256':hashlib.sha256((ROOT/'common/w4utils.py').read_bytes()).hexdigest(),
        'runs':rows,'matched_discrete_steady_comparisons':comparisons,
        'all_new_methods_converged':True,'benchmark_refinement_trends':trends,
        'ghia_velocity_errors_decrease_on_refinement':all(t['ghia_u_relative_l2_decreases'] and t['ghia_v_relative_l2_decreases'] for t in trends),
        'refinement_limit':'The FD reference errors decrease. FV Ghia errors are nonmonotone across these two grids; no asymptotic accuracy gate is claimed.',
        'scope':'Re=100 steady cavity; PISO/PIMPLE time-marched to steady state. No transient-accuracy qualification or higher-Re claim.',
        'timings':'One local CPU solve each; not a hardware-independent ranking.'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'completed_runs':len(rows),'comparisons':comparisons},indent=2))


if __name__=='__main__':main()
