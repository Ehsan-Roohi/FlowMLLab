"""Independent streamfunction-vorticity adapter; Week-1 source stays unchanged."""
from pathlib import Path
import sys,time
import numpy as np
from .pressure_velocity_experiments import benchmark_errors
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
import w4utils as sv

def run_streamfunction(reynolds=400,cells=32,dt=.001,max_steps=400000,progress=False):
    start=time.perf_counter()
    r=sv.run_cavity(Re=reynolds,N=cells+1,dt=dt,max_steps=max_steps,
        check_every=500,min_steps=5000,tol=1e-12,consecutive_required=3,verbose=progress)
    r['runtime_seconds']=time.perf_counter()-start
    r['config']={'method':'streamfunction-vorticity','cells':cells,'nodes':cells+1,
        'reynolds':reynolds,'dt':dt,'convection':'central','max_steps':max_steps,
        'relative_update_tolerance':1e-12,'check_every':500,'consecutive_required':3}
    h=1/cells
    updated,_=sv.advance_vorticity(r['omega'].copy(),r['u'],r['v'],reynolds,dt,h,h)
    r['steady_vorticity_linf']=float(np.max(np.abs((updated-r['omega'])[1:-1,1:-1]))/dt)
    du=(r['u'][1:-1,2:]-r['u'][1:-1,:-2])/(2*h)
    dv=(r['v'][2:,1:-1]-r['v'][:-2,1:-1])/(2*h)
    r['continuity_linf']=float(np.abs(du+dv).max())
    r['update_converged']=bool(r['converged'])
    r['converged']=bool(r['converged'] and r['steady_vorticity_linf']<1e-6)
    r['status']='converged' if r['converged'] else 'limit_or_failed_steady_gate'
    r['poisson_solves']=int(r['steps']+1)
    r['history_rate']=r['residual_values']/dt
    j,i=np.unravel_index(np.argmin(r['psi']),r['psi'].shape)
    r.update(vortex_x=float(r['x'][i]),vortex_y=float(r['y'][j]),psi_min=float(r['psi'][j,i]))
    r.update(benchmark_errors(r))
    return r
