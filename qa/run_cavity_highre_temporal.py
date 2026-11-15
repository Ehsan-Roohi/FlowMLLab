"""Paired finite-horizon physical-time audit of converged steady CFD roots.

Both dt values start from the same deterministic smooth streamfunction
perturbation and retain identical observation times. This is not a test for
asymptotic temporal stability, and cannot validate a three-dimensional flow.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import numpy as np
from scipy.fft import dstn, idstn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flowmllab'))
sys.path.insert(0,str(ROOT/'qa'))
from cavity_steady import SteadyCavity
from run_cavity_highre import atomic_npz, atomic_json, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--re',type=int,required=True)
    p.add_argument('--dt',type=float,required=True)
    a=p.parse_args()
    cells=256; duration=10.0; amplitude=1e-4
    source=a.root/'results'/f'n{cells}'/f're{a.re}.npz'
    metadata=json.loads(source.with_suffix('.json').read_text())
    if not metadata['converged'] or digest(source)!=metadata['field_sha256']:
        raise ValueError('Unverified initial steady field')
    root=dict(np.load(source,allow_pickle=False))
    s=SteadyCavity(cells)
    X,Y=np.meshgrid(root['x'],root['y'])
    perturbation=amplitude*np.sin(np.pi*X)**2*np.sin(np.pi*Y)**2
    h=s.h
    dw=-(perturbation[1:-1,2:]+perturbation[1:-1,:-2]+perturbation[2:,1:-1]+perturbation[:-2,1:-1]-4*perturbation[1:-1,1:-1])/h**2
    w=(root['omega'][1:-1,1:-1]+dw).ravel()
    config=dict(reynolds=a.re,cells=cells,dt=a.dt,duration=duration,
                perturbation_streamfunction_amplitude=amplitude,
                perturbation='amplitude*sin(pi*x)^2*sin(pi*y)^2',
                method='SSPRK3; fresh DST Poisson/Thom boundaries each stage',
                initial_omega_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                steady_field_sha256=digest(source),
                source_sha256_lf={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                    for f in [Path(__file__),ROOT/'flowmllab/cavity_steady.py']})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    out=a.root/'temporal'/f're{a.re}-dt{a.dt:g}'
    out.mkdir(parents=True,exist_ok=True)
    nsteps=round(duration/a.dt); stride=round(.5/a.dt)
    if abs(nsteps*a.dt-duration)>1e-12 or abs(stride*a.dt-.5)>1e-12:
        raise ValueError('Time steps must exactly match the observation window')
    checkpoint=out/'checkpoint.npz'
    rows=[];start_step=0
    if checkpoint.exists():
        z=np.load(checkpoint,allow_pickle=False)
        cm=json.loads(str(z['metadata']))
        if cm['signature']!=signature: raise ValueError('Temporal restart mismatch')
        w=z['omega'].copy();start_step=cm['step'];rows=cm['rows']
    started=time.perf_counter()
    def fields(state):
        psi=idstn(dstn(-state.reshape(s.m,s.m),type=1,norm='ortho')/s.poisson_eigenvalues,type=1,norm='ortho')
        return s.fields(np.r_[psi.ravel(),state])
    def observe(step):
        f=fields(w)
        du=f['u']-root['u'];dv=f['v']-root['v']
        row=dict(time=step*a.dt,velocity_departure_rms_U=float(np.sqrt(np.mean(du**2+dv**2))),
                 velocity_departure_linf_U=float(max(np.abs(du).max(),np.abs(dv).max())),
                 advective_cfl=float(a.dt/h*np.max(np.abs(f['u'])+np.abs(f['v']))),
                 u_center=float(f['u'][cells//2,cells//2]),v_center=float(f['v'][cells//2,cells//2]))
        if not np.isfinite(w).all() or row['advective_cfl']>1:
            raise FloatingPointError('Temporal finite/CFL gate failed')
        rows.append(row)
        atomic_npz(checkpoint,omega=w,metadata=np.array(json.dumps(dict(signature=signature,step=step,rows=rows))))
        print(json.dumps(dict(reynolds=a.re,dt=a.dt,**row)),flush=True)
        return f
    if start_step==0 and not rows: observe(0)
    for step in range(start_step+1,nsteps+1):
        w=s.rk3(w,a.re,a.dt)
        if step%stride==0:
            final=observe(step)
    if start_step==nsteps: final=fields(w)
    atomic_npz(out/'final.npz',**final)
    atomic_json(out/'audit.json',dict(config=config,signature=signature,observations=rows,
                final_sha256=digest(out/'final.npz'),elapsed_seconds_this_attempt=time.perf_counter()-started,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                claim='matched finite-horizon time-discretization audit, not asymptotic stability'))


if __name__=='__main__': main()
