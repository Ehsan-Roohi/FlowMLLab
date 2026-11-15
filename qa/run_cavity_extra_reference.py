"""Additional CFD parameter cases for a frozen full-field neural split."""
from pathlib import Path
import argparse,hashlib,json,os,socket,sys,time
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'flowmllab'),str(ROOT/'qa')]
from cavity_steady import SteadyCavity
from run_cavity_highre import atomic_npz,atomic_json,digest
CASES={250:(100,(128,256,512)),2500:(1000,(128,256,512)),4000:(5000,(128,256,512)),
       6000:(5000,(128,256)),6500:(5000,(128,256)),9000:(10000,(128,256))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--re',type=int,choices=CASES,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    start_re,grids=CASES[a.re]
    source_hashes={name:hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                  for name in ('flowmllab/cavity_steady.py','qa/run_cavity_extra_reference.py')}
    for n in grids:
        solver=SteadyCavity(n);out=a.root/'results'/f'n{n}'
        seed_path=out/f're{start_re}.npz';seed_meta=json.loads(seed_path.with_suffix('.json').read_text())
        if digest(seed_path)!=seed_meta['field_sha256'] or not seed_meta['converged']:
            raise ValueError('Initial CFD root is unverified')
        config=dict(reynolds=a.re,cells=n,method='steady-newton-central-thom',lid=1.,length=1.,
                    nonlinear_tolerance=1e-8,acceptance_steady_linf=1e-6,start_reynolds=start_re,
                    seed_sha256=digest(seed_path),source_sha256_lf=source_hashes)
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        final_path=out/f're{a.re}.npz';final_meta=final_path.with_suffix('.json')
        if final_meta.exists():
            saved=json.loads(final_meta.read_text())
            if saved['signature']!=signature or saved['field_sha256']!=digest(final_path):raise ValueError('Final root mismatch')
            continue
        checkpoint_dir=a.root/'extra_checkpoints';checkpoint_dir.mkdir(exist_ok=True)
        checkpoint=checkpoint_dir/f're{a.re}-n{n}.npz'
        history=[];current=float(start_re);started=time.perf_counter()
        if checkpoint.exists():
            data=np.load(checkpoint,allow_pickle=False);meta=json.loads(str(data['metadata']))
            if meta['signature']!=signature:raise ValueError('Continuation restart mismatch')
            current=meta['completed_reynolds'];history=meta['history'];state=data['state'].copy()
        else:
            state=solver.interpolate_seed(dict(np.load(seed_path,allow_pickle=False)))
        increment=500.
        while current!=a.re:
            proposed=current+np.sign(a.re-current)*min(increment,abs(a.re-current))
            try:
                candidate,records=solver.solve(state,proposed,callback=lambda z,r:print(json.dumps(dict(target=a.re,cells=n,reynolds=proposed,**r)),flush=True))
            except ArithmeticError as exc:
                history.append(dict(reynolds=proposed,success=False,error=str(exc)))
                if increment<=15.625:raise
                increment/=2;continue
            state=candidate;current=float(proposed)
            history.append(dict(reynolds=current,success=True,iterations=len(records)-1,final=records[-1]))
            atomic_npz(checkpoint,state=state,metadata=np.array(json.dumps(dict(signature=signature,config=config,completed_reynolds=current,history=history))))
        f=solver.fields(state);res=solver.residual(state,a.re);h=solver.h
        div=(f['u'][1:-1,2:]-f['u'][1:-1,:-2]+f['v'][2:,1:-1]-f['v'][:-2,1:-1])/(2*h)
        j,i=np.unravel_index(np.argmin(f['psi']),f['psi'].shape)
        atomic_npz(final_path,**f)
        atomic_json(final_meta,dict(config=config,signature=signature,reynolds=a.re,cells=n,converged=bool(np.max(np.abs(res))<1e-6),
            poisson_linf=float(np.abs(res[:solver.n]).max()),steady_vorticity_linf=float(np.abs(res[solver.n:]).max()),
            continuity_linf=float(np.abs(div).max()),vortex_x=float(f['x'][i]),vortex_y=float(f['y'][j]),psi_min=float(f['psi'][j,i]),
            field_sha256=digest(final_path),newton_history=records,continuation_history=history,
            cumulative_compute_seconds=time.perf_counter()-started,host=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            utc=datetime.now(timezone.utc).isoformat(),ghia_used_during_solution=False,branch='2-D steady Navier-Stokes root'))
    print(json.dumps(dict(event='complete',reynolds=a.re,grids=grids)),flush=True)


if __name__=='__main__':main()
