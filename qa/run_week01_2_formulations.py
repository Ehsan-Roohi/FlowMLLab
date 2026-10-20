"""Sequential, single-thread higher-Re formulation study from rest."""
from pathlib import Path
import sys,time,json,hashlib,platform,argparse,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,scipy
from flowmllab.streamfunction_experiments import run_streamfunction
from flowmllab.pressure_velocity_experiments import ExperimentalConfig,run_experiment
OUT=ROOT/'results/week01_2_formulations'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--re',type=int,nargs='+',default=[400,1000]);ap.add_argument('--cells',type=int,nargs='+',default=[32]);ap.add_argument('--methods',nargs='+',default=['simple','piso','pimple','streamfunction-vorticity']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    sources=['common/w4utils.py','flowmllab/pressure_velocity.py','flowmllab/pressure_velocity_experiments.py','flowmllab/streamfunction_experiments.py','qa/run_week01_2_formulations.py']
    loaded_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    for n in args.cells:
      for re in args.re:
       for method in args.methods:
        name=f'{method}-re{re}-n{n}';target=OUT/(name+'.json')
        if target.exists():print('Retained',name,flush=True);continue
        print('START',name,flush=True)
        start=time.perf_counter()
        try:
          if method=='streamfunction-vorticity':r=run_streamfunction(re,n,progress=True)
          else:r=run_experiment(ExperimentalConfig(method=method,cells=n,reynolds=re,dt=.05,velocity_relaxation=.5,pressure_relaxation=.2,tolerance=1e-6,max_iterations=6000,check_every=50),progress=True)
          total=time.perf_counter()-start
          arrays={k:v for k,v in r.items() if isinstance(v,np.ndarray)}
          archive=OUT/(name+'.npz');np.savez_compressed(archive,**arrays)
          record={k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
          record.update(field_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),status='converged' if r['converged'] else 'not_converged',end_to_end_seconds=total)
        except Exception as e:record={'config':{'method':method,'reynolds':re,'cells':n},'status':'failed','error':repr(e),'end_to_end_seconds':time.perf_counter()-start}
        record.update(date='2026-10-20',timing_protocol='Sequential runs from rest; one sample; one BLAS/OMP thread; no parallel study jobs',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,machine=platform.processor(),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},source_sha256=loaded_hashes)
        target.write_text(json.dumps(record,indent=2)+'\n')
        print('DONE',name,record['status'],record.get('end_to_end_seconds'),flush=True)
if __name__=='__main__':main()
