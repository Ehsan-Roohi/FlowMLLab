"""Retain higher-Re successes and failures, without modifying Re=100 evidence."""
from pathlib import Path
import sys,json,hashlib,platform,argparse
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flowmllab.pressure_velocity_experiments import ExperimentalConfig,run_experiment

OUT=ROOT/'results/week01_2_higher_re'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cells',type=int,default=32)
    parser.add_argument('--re',type=int,nargs='+',default=[400,1000])
    parser.add_argument('--methods',nargs='+',default=['simple','piso','pimple'])
    parser.add_argument('--scheme',default='central',choices=['central','upwind'])
    parser.add_argument('--dt',type=float,default=.05)
    parser.add_argument('--iterations',type=int,default=2500)
    args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    for re in args.re:
        for method in args.methods:
            name=f'{method}-re{re}-n{args.cells}-{args.scheme}-dt{args.dt:g}'
            target=OUT/(name+'.json')
            if target.exists():
                print('EXISTS',name,flush=True);continue
            cfg=ExperimentalConfig(method=method,cells=args.cells,reynolds=re,convection=args.scheme,dt=args.dt,
                velocity_relaxation=.5,pressure_relaxation=.2,tolerance=1e-6,max_iterations=args.iterations,check_every=50)
            print('START',name,flush=True)
            try:
                r=run_experiment(cfg,progress=True)
                arrays={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
                archive=OUT/(name+'.npz');np.savez_compressed(archive,**arrays)
                record={k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
                record['field_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
                record['status']='converged' if r['converged'] else 'iteration_limit'
            except Exception as error:
                from dataclasses import asdict
                record={'config':asdict(cfg),'status':'failed','error':type(error).__name__+': '+str(error)}
            record['date']='2026-10-15'
            record['python']=platform.python_version();record['numpy']=np.__version__;record['scipy']=scipy.__version__
            record['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['flowmllab/pressure_velocity.py','flowmllab/pressure_velocity_experiments.py']}
            target.write_text(json.dumps(record,indent=2)+'\n')
            print('DONE',name,json.dumps({k:record[k] for k in ['status','iterations','momentum_residual_linf','ghia_u_relative_l2','ghia_v_relative_l2'] if k in record}),flush=True)

if __name__=='__main__':main()
