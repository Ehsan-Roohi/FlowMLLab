"""Run paired methods sequentially on one Slurm allocation; no duplicate live trials."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,signal,hashlib,socket
ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    tmp=path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['pilot','production'],required=True)
    p.add_argument('--cells',type=int,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection',type=Path)
    args=p.parse_args()
    out=args.root/args.phase/f'n{args.cells}';out.mkdir(parents=True,exist_ok=True)
    lock=out/'writer.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID')}).encode());os.close(fd)
    try:
        if args.phase=='pilot':
            trials=[dict(method='direct',fill=4.,drop=1e-3,scenario=s,repeat=0)
                for s in ['warm7500','warm10000']]
            for fill,drop in [(4.,1e-3),(8.,1e-4),(12.,1e-5)]:
                for s in ['warm7500','warm10000']:
                    trials.append(dict(method='gmres_mg',fill=fill,drop=drop,scenario=s,repeat=0))
            timeout=240
        else:
            if args.selection is None: raise ValueError('Frozen pilot selection required')
            selected=json.loads(args.selection.read_text())
            fill,drop=selected['fill'],selected['drop']
            trials=[]
            for s in ['warm7500','warm10000']:
                for repeat in range(3):
                    methods=['direct','gmres_mg'] if repeat%2==0 else ['gmres_mg','direct']
                    trials.extend(dict(method=m,fill=fill,drop=drop,scenario=s,repeat=repeat) for m in methods)
            # End-to-end continuation is separately paired on the medium grid.
            if args.cells==256:
                trials.extend(dict(method=m,fill=fill,drop=drop,scenario='continuation',repeat=0)
                    for m in ['direct','gmres_mg'])
            timeout=2400 if args.cells==256 else 1200
        plan=dict(phase=args.phase,cells=args.cells,trials=trials,timeout_seconds_per_trial=timeout,
                  paired_same_allocation=True,order='AB BA AB within each warm scenario',
                  checkpoint='one complete root per trial',
                  source_hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in [ROOT/'qa/benchmark_cavity_solvers.py',ROOT/'flowmllab/cavity_krylov.py',ROOT/'flowmllab/cavity_steady.py']})
        write(out/'plan.json',plan)
        completed=[]
        for i,t in enumerate(trials):
            tag=f"{i:02d}_{t['scenario']}_{t['method']}_r{t['repeat']}"
            folder=out/tag;folder.mkdir(exist_ok=True)
            result=folder/'result.json'
            if result.exists():
                raise FileExistsError('Existing trial requires explicit reconciliation: '+str(result))
            cmd=[sys.executable,'-u',str(ROOT/'qa/benchmark_cavity_solvers.py'),
                '--cells',str(args.cells),'--method',t['method'],'--scenario',t['scenario'],
                '--repeat',str(t['repeat']),'--data-root',str(args.root/'input'),
                '--output',str(folder),'--fill-factor',str(t['fill']),'--drop-tol',str(t['drop'])]
            print(json.dumps({'event':'trial_start','tag':tag,'command_args':cmd[3:]}),flush=True)
            start=time.perf_counter()
            with (folder/'run.log').open('w') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try: code=child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                    code=124
            if not result.exists():
                write(result,t|dict(cells=args.cells,success=False,error=f'Process exited {code} without a result',
                    process_wall_seconds=time.perf_counter()-start,host=socket.gethostname(),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID')))
            data=json.loads(result.read_text())
            completed.append(dict(tag=tag,returncode=code,success=data.get('success',False),
                process_wall_seconds=time.perf_counter()-start))
            write(out/'status.json',dict(state='running',completed=completed,planned=len(trials)))
            print(json.dumps({'event':'trial_complete'}|completed[-1]),flush=True)
        write(out/'status.json',dict(state='complete',completed=completed,planned=len(trials)))
    finally:
        lock.unlink()


if __name__=='__main__': main()
