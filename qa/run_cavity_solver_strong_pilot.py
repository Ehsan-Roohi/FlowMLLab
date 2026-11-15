"""Bounded 128-cell preconditioner-strength follow-up; unchanged equations/gates."""
from pathlib import Path
import os,json,sys,subprocess,time,signal,socket,hashlib
r=Path(sys.argv[1]);out=r/'pilot_strong/n128';out.mkdir(parents=True,exist_ok=True)
lock=out/'writer.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID')}).encode());os.close(fd)
try:
    trials=[dict(method='direct',fill=4.,drop=1e-3,scenario=s,repeat=0) for s in ['warm7500','warm10000']]
    for fill,drop in [(20.,1e-6),(40.,1e-8)]:
        trials.extend(dict(method='gmres_mg',fill=fill,drop=drop,scenario=s,repeat=0) for s in ['warm7500','warm10000'])
    plan={'reason':'All original 128-cell pilot candidate settings failed factorization or the original-equation gate. Test stronger ILUT on pilot grid only; no production-data tuning.','cells':128,'trials':trials,'timeout_seconds_per_trial':180,'host':socket.gethostname(),'job':os.environ.get('SLURM_JOB_ID'),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    completed=[]
    for i,t in enumerate(trials):
        tag=f"{i:02d}_{t['scenario']}_{t['method']}_r0";folder=out/tag;folder.mkdir(exist_ok=False)
        cmd=[sys.executable,'-u',str(r/'source/qa/benchmark_cavity_solvers.py'),'--cells','128','--method',t['method'],'--scenario',t['scenario'],'--repeat','0','--data-root',str(r/'input'),'--output',str(folder),'--fill-factor',str(t['fill']),'--drop-tol',str(t['drop'])]
        print(json.dumps({'start':tag,'config':t}),flush=True);start=time.perf_counter()
        with (folder/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=124
        result=folder/'result.json'
        if not result.exists():result.write_text(json.dumps(t|{'cells':128,'success':False,'error':f'Process exited {code} without result','process_wall_seconds':time.perf_counter()-start,'host':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
        data=json.loads(result.read_text());completed.append({'tag':tag,'returncode':code,'success':data['success'],'process_wall_seconds':time.perf_counter()-start})
        tmp=out/'status.pending.json';tmp.write_text(json.dumps({'state':'running','completed':completed,'planned':len(trials)},indent=2)+'\n');os.replace(tmp,out/'status.json')
        print(json.dumps(completed[-1]),flush=True)
    tmp=out/'status.pending.json';tmp.write_text(json.dumps({'state':'complete','completed':completed,'planned':len(trials)},indent=2)+'\n');os.replace(tmp,out/'status.json')
finally:lock.unlink()
