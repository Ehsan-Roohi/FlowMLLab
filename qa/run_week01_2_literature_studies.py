"""Run small original CPU teaching experiments motivated by four readings."""
from pathlib import Path
import argparse, hashlib, json, os, platform, sys
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from flowmllab.pressure_velocity_studies import pressure_work_study, startup_study


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()): raise FileExistsError('Use an empty results directory; never overwrite a run')
    p,pf=pressure_work_study(); print('Pressure batches complete',flush=True)
    t,tf,checks=startup_study(); print('Startup trajectories complete',flush=True)
    np.savez_compressed(args.output/'fields.npz',**{'pressure_'+k:v for k,v in pf.items()},**{'time_'+k:v for k,v in tf.items()})
    report={'started_from_rest':True,'pressure_trial_fields':'Analytic synthetic trial velocities; not cavity CFD snapshots.',
        'time_config':'Re100, 16 MAC cells/side, unit lid/square, central convection, backward Euler, t=0.2.',
        'pressure_config':'Fixed L=-0.05*D*G, impermeable walls, pinned solve then zero mean; three analytic RHS; Jacobi GMRES restart50 max200 cycles.',
        'pressure_rows':p,'time_rows':t,'reference_checks':checks,
        'provenance':{'completed_utc':datetime.now(timezone.utc).isoformat(),'host':platform.node(),
            'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'slurm_job':os.environ.get('SLURM_JOB_ID'),'source_commit':os.environ.get('SOURCE_COMMIT'),
            'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ('flowmllab/pressure_velocity.py','flowmllab/pressure_velocity_studies.py','qa/run_week01_2_literature_studies.py')},
            'fields_sha256':hashlib.sha256((args.output/'fields.npz').read_bytes()).hexdigest()},
        'limits':['One host, small fixed meshes, three repeat timings; no universal speed ranking.',
                  'Fixed-operator LU reuse is invalid when momentum coefficients, time step or boundary conditions change.',
                  'Reference sensitivity is measured; neither continuum/grid independence nor second-order time accuracy is claimed.']}
    (args.output/'study.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pressure_rows':len(p),'startup_cases':len(t),'reference_checks':checks}),flush=True)


if __name__=='__main__': main()
