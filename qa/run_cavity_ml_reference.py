"""Matched low-Re roots for the full-field neural training/validation dataset.
Uses the same frozen spatial equations as the high-Re campaign, no Ghia values.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, socket
from datetime import datetime, timezone
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'flowmllab'),str(ROOT/'qa')]
from cavity_steady import SteadyCavity
from run_cavity_highre import atomic_npz, atomic_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--re',type=int,choices=(100,400,1000),required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    original=dict(np.load(a.seed,allow_pickle=False));initial=original
    history=[];started=time.perf_counter()
    hashes={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in ('flowmllab/cavity_steady.py','qa/run_cavity_ml_reference.py')}
    for n in (128,256,512):
        out=a.output/f'n{n}';out.mkdir(parents=True,exist_ok=True)
        config=dict(cells=n,reynolds=a.re,method='steady-newton-central-thom',lid=1.,length=1.,
                    nonlinear_tolerance=1e-8,acceptance_steady_linf=1e-6,
                    source_sha256_lf=hashes,seed_sha256=digest(a.seed),mesh_sequence=[128,256,512])
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        field_path=out/f're{a.re}.npz';meta_path=field_path.with_suffix('.json')
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta['signature']!=signature or meta['field_sha256']!=digest(field_path):
                raise ValueError('Existing low-Re root identity mismatch')
            initial=dict(np.load(field_path,allow_pickle=False))
            history=meta['continuation_history'];continue
        solver=SteadyCavity(n);z=solver.interpolate_seed(initial)
        z,records=solver.solve(z,a.re,callback=lambda z,r:print(json.dumps(dict(cells=n,reynolds=a.re,**r)),flush=True))
        f=solver.fields(z);res=solver.residual(z,a.re);h=solver.h
        div=(f['u'][1:-1,2:]-f['u'][1:-1,:-2]+f['v'][2:,1:-1]-f['v'][:-2,1:-1])/(2*h)
        j,i=np.unravel_index(np.argmin(f['psi']),f['psi'].shape)
        atomic_npz(field_path,**f)
        history.append(dict(reynolds=a.re,cells=n,success=True,iterations=len(records)-1,final=records[-1]))
        meta=dict(config=config,signature=signature,reynolds=a.re,cells=n,converged=bool(np.max(np.abs(res))<1e-6),
            poisson_linf=float(np.abs(res[:solver.n]).max()),steady_vorticity_linf=float(np.abs(res[solver.n:]).max()),
            continuity_linf=float(np.abs(div).max()),vortex_x=float(f['x'][i]),vortex_y=float(f['y'][j]),psi_min=float(f['psi'][j,i]),
            field_sha256=digest(field_path),newton_history=records,continuation_history=history,
            cumulative_compute_seconds=time.perf_counter()-started,host=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            utc=datetime.now(timezone.utc).isoformat(),environment=dict(numpy=np.__version__,scipy=scipy.__version__),
            ghia_used_during_solution=False,branch='2-D steady Navier-Stokes root')
        atomic_json(meta_path,meta)
        initial=f
    print(json.dumps(dict(event='complete',reynolds=a.re,grids=[128,256,512])),flush=True)


if __name__=='__main__':main()
