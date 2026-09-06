#!/usr/bin/env python3
"""Controlled CPU boundary-window comparison. Never approves data from exit status."""
import argparse, fcntl, hashlib, importlib.util, json, os, re, shutil
import statistics, subprocess, tempfile, time
from pathlib import Path

spec=importlib.util.spec_from_file_location('cpu_campaign',Path(__file__).with_name('cpu_campaign.py'))
cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu)
camp,bench,ck,pilot=cpu.campaign,cpu.bench,cpu.checkpoint,cpu.pilot
NAMES=['boundary_probe.py','boundary_probe_job.sh','patch_face_window.py']+cpu.FILES
CASES=['cpu_h21_s20260917','cpu_h16_s20260905','cpu_h16_s20260917']
WINDOWS=[0,50,200]

def save(p,d): ck.atomic_json(p,d)
def read(p): return json.loads(Path(p).read_text())

def run_arm(root,row,binary,window,restart=None,serial=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=ck.completed(root)
        if done: return done
        attempts=root/'attempts';attempts.mkdir(exist_ok=True)
        n=max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()],default=0)+1
        path=attempts/f'{n:04d}'
        restart=ck.latest_warm(root) or restart
        camp.generate_case(path,row,restart,'smoke' if restart else None,smoke=row.get('smoke',False))
        m=read(path/'case.json');m.update(boundary_window=window,experimental_boundary=True,training_data_approved=False)
        save(path/'case.json',m)
        deck=(path/'in.step').read_text()
        lines=deck.splitlines()
        for i,line in enumerate(lines):
            if line.startswith(('fix inlet emit/face','fix outlet emit/face')):
                lines[i]=line+f' window {window}'
        deck='\n'.join(lines)+'\n'
        # Keep standard 10000-step field blocks; diagnose open-boundary flux
        # every 100 steps without multiplying large field outputs.
        fine=10 if row.get('smoke') else 100
        anchor='print "SPARTA_STEP_SAMPLING_BEGIN"'
        deck=deck.replace(anchor,anchor+f'\nfix fluxfine ave/time 1 {fine} {fine} f_inlet[1] f_outlet[1] v_exits_in v_exits_out v_inventory file flux.fine')
        # Do not split the sampling run: init() rebuilds emit tasks and resets
        # the exponential velocity average. Warm/sample boundary still resets;
        # both windows are << first 10000-step output block.
        b=row['block_steps']; sample=row['sampling_steps']
        deck=deck.replace(f'run {sample-b}\ndump final', 'dump final')
        deck=deck.replace(f'run {b}\nwrite_restart restart.final',f'run {sample}\nwrite_restart restart.final')
        (path/'in.step').write_text(deck);ck.checkpoint_deck(path)
        timing=bench.execute(path,bench.command(binary,'cpu',ranks=row['ranks'],serial=serial),timeout=23*3600)
        if [r['steps'] for r in timing['loops']] != [row['warmup_steps'],sample]:
            raise ValueError('Incomplete solver loops')
        report=bench.validate_case(path)
        report.update(experimental_boundary=True,boundary_window=window,training_data_approved=False)
        save(path/'report.json',report);save(path/'timing.json',timing)
        ck.commit_result(root,path,timing,report)
        print(f'BOUNDARY_ARM_COMPLETE={root.name} WINDOW={window}',flush=True)
        return path,timing,report

def preflight(out,binary,original,serial):
    out=Path(out); outputs={}
    for w in WINDOWS:
        row=bench.smoke_row(21);row.update(ranks=2,smoke=True)
        outputs[w]=run_arm(out/f'w{w}',row,binary,w,serial=serial)
    # A restart with smoothing enabled exercises changed MPI rank counts too.
    row=bench.smoke_row(21);row.update(ranks=4,smoke=True)
    run_arm(out/'restart',row,binary,50,outputs[50][0]/'restart.warm',serial)
    # Exact numerical default preservation using the original executable.
    row=bench.smoke_row(21);row.update(ranks=2,smoke=True)
    old=out/f'original-{time.time_ns()}';camp.generate_case(old,row,smoke=True)
    # Compare identical diagnostics/run segmentation, only the executable and
    # the newly supported explicit window=0 keyword differ.
    reference=(outputs[0][0]/'in.step').read_text().replace(' window 0','')
    reference=reference.replace(str(outputs[0][0].resolve()),str(old.resolve()))
    (old/'in.step').write_text(reference)
    bench.execute(old,bench.command(original,'cpu',ranks=2,serial=serial),timeout=300)
    import gzip
    if gzip.open(old/'grid.final.gz','rb').read()!=gzip.open(outputs[0][0]/'grid.final.gz','rb').read():
        raise ValueError('Window=0 altered default numerical result')
    save(out/'PREFLIGHT_PASS.json',{'window0_exact_match':True,'positive_windows':WINDOWS[1:],'restart_pass':True})

def submit(base,source,ref):
    base=Path(base).resolve();source=Path(source).resolve()
    if not re.fullmatch('[0-9a-f]{40}',ref):raise ValueError('Require immutable commit')
    meta=read(source/'manifest.json')
    with (base/'boundary-submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pointer=base/'LATEST_SPARTA_STEP_BOUNDARY_PROBE'
        if pointer.exists():raise ValueError('Boundary probe already exists: '+pointer.read_text().strip())
        jobs=cpu.account_jobs('pi_roohie_umass_edu')
        reserved=sum(j['cpus'] for j in jobs)
        parallel=min(3,(1000-reserved-3)//16)
        if parallel<1:raise ValueError('Insufficient remaining CPU quota')
        if shutil.disk_usage(base).free<60*1024**3:raise ValueError('Need 60 GiB free scratch')
        out=Path(tempfile.mkdtemp(prefix='step-boundary-'+time.strftime('%Y%m%dT%H%M%SZ-',time.gmtime()),dir=base/'runs'))
        code=out/'code';code.mkdir()
        for name in NAMES:shutil.copy2(Path(__file__).with_name(name),code/name)
        (out/'code.sha256').write_text(''.join(f'{ck.sha(code/n)}  code/{n}\n' for n in NAMES))
        arms=[]
        for name in CASES:
            row=next(r.copy() for r in meta['cases'] if r['id']==name)
            warm=ck.latest_warm(source/'cases'/name)
            if not warm:raise ValueError('Missing verified warm checkpoint: '+name)
            if row['nx']!=1000 or row['ny']!=200:raise ValueError('Wrong grid')
            bench.check_particle_budget(row)
            for w in WINDOWS:
                arm=dict(row,id=f'{name}_w{w}',window=w,original_case=name,restart=str(warm),restart_sha256=ck.sha(warm))
                arm.update(warmup_steps=40000,sampling_steps=120000,block_steps=10000,sample_every=10)
                arms.append(arm)
        config=dict(source_campaign=str(source),source_pilot=meta['source'],ref=ref,arms=arms,parallel=parallel,reserved_cpus=reserved,jobs={})
        save(out/'probe.json',config);pointer.write_text(str(out)+'\n')
        def queue(phase,ranks,dependency=None,array=None):
            cmd=['sbatch','--parsable','--partition=cpu','--account=pi_roohie_umass_edu','--constraint=x86_64','--nodes=1',f'--ntasks={ranks}','--cpus-per-task=1','--mem=16G','--time=24:00:00','--requeue','--job-name=step-bc-'+phase,'--kill-on-invalid-dep=yes',f'--output={out}/slurm-%A_%a.out',f'--error={out}/slurm-%A_%a.err',f'--export=ALL,SPARTA_BOUNDARY_OUT={out}']
            if dependency:cmd+=['--dependency=afterok:'+dependency]
            if array:cmd+=['--array='+array]
            cmd +=[str(code/'boundary_probe_job.sh'),phase]
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            if not job.isdigit():raise ValueError('Unexpected sbatch response')
            config['jobs'][phase]=job;save(out/'probe.json',config);print(phase+'='+job,flush=True);return job
        first=queue('build',16)
        second=queue('arms',16,first,f'0-8%{parallel}')
        queue('review',1,second)
        print('OUT='+str(out),flush=True)

def review(out):
    out=Path(out);m=read(out/'probe.json');data={}
    for row in m['arms']:
        done=ck.completed(out/'arms'/row['id'])
        if not done:raise ValueError('Incomplete comparison')
        r=done[2]; u=[b['bulk_u_m_s'] for b in r['blocks']]
        data[row['id']]={'window':row['window'],'mean_bulk_u':statistics.fmean(u),'bulk_half_drift':r['bulk_velocity_half_drift_fraction'],'reverse_half_drift':r['regional_reverse_half_drift_fraction'],'mass_imbalance':r['mass_imbalance_fraction'],'pressure_errors':r['boundary_pressure_error_fraction'],'checks':r['checks']}
    save(out/'boundary_comparison.json',{'arms':data,'scientific_approval':False,'next':'Review stability, pressure fidelity, and agreement of windows 50/200 before accepting or extending.'})
    print(json.dumps(data,indent=2));print('BOUNDARY_COMPARISON_COMPLETE SCIENTIFIC_APPROVAL=False')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['submit','preflight','run','review']);p.add_argument('--base',default=pilot.DEFAULT_BASE);p.add_argument('--source');p.add_argument('--ref');p.add_argument('--out');p.add_argument('--binary');p.add_argument('--original');p.add_argument('--serial',action='store_true');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='submit':submit(a.base,a.source,a.ref)
    elif a.action=='preflight':preflight(a.out,a.binary,a.original,a.serial)
    elif a.action=='review':review(a.out)
    else:
        out=Path(a.out);row=read(out/'probe.json')['arms'][a.index]
        if ck.sha(row['restart'])!=row['restart_sha256']:raise ValueError('Source checkpoint changed')
        if int(os.environ.get('SLURM_NTASKS',0))!=16:raise ValueError('Requires 16 CPU ranks')
        if not (out/'preflight'/'PREFLIGHT_PASS.json').exists():raise ValueError('Missing preflight')
        run_arm(out/'arms'/row['id'],row,a.binary,row['window'],row['restart'])
