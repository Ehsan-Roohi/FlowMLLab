#!/usr/bin/env python3
"""Isolated DSMC mesh/time controls and gated, frozen-weight Bezier evaluation."""
import argparse, contextlib, csv, fcntl, hashlib, importlib.util, json, os
from pathlib import Path
import shutil, subprocess, sys, tempfile, time

def read(p): return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n'); t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()
def load_module(p,name):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def submit(base):
    base=Path(base).resolve(); pointer=base/'LATEST_STEP_CONTROLS_20260908'
    with (base/'controls-20260908.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if pointer.exists(): raise RuntimeError('Already submitted: '+pointer.read_text())
        prev=Path((base/'LATEST_SPARTA_FOLLOWUP16_20260908_CAMPAIGN').read_text().strip())
        meta=read(prev/'manifest.json'); cpu=load_module(prev/'code/cpu_campaign.py','existing_cpu')
        jobs=cpu.account_jobs(meta['account']); reserved=sum(j['cpus'] for j in jobs)
        concurrency=min(4,(1000-reserved-5)//64)
        if concurrency<1: raise RuntimeError('Insufficient account CPU headroom')
        if shutil.disk_usage(base).free<300*1024**3: raise RuntimeError('Need 300 GiB scratch headroom')
        out=Path(tempfile.mkdtemp(prefix='step-controls-20260908-',dir=base/'runs'))
        shutil.copytree(prev/'code',out/'code',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(__file__,out/'code/step_controls_20260908.py')
        for name in ['step_interpolation_train_v2.py']:
            shutil.copy2(base/name,out/'code'/name)
        reps=[meta['cases'][i] for i in [12,14,15,1]]
        rows=[]
        for r in reps:
            for level,nx,factor in [('coarse',1000,3.5),('medium',2000,3.5),('fine',3500,3.5),('halfdt',3500,7.)]:
                v=dict(r);v.update(id=r['id']+'__'+level,representative=r['id'],control=level,nx=nx,ny=nx//5,dt_s=cpu.campaign.REFERENCE_DT/factor,ranks=64,phase='mesh_time_control',warmup_steps=round(80000*factor),sampling_steps=round(240000*factor),block_steps=round(10000*factor),sample_every=round(10*factor))
                v.pop('fluid_cell_equivalents',None);v.pop('initial_particle_estimate',None);rows.append(v)
        old=dict(rows[0]);old.update(id=reps[0]['id']+'__continuous_original_dt',control='continuous_original_dt',dt_s=cpu.campaign.REFERENCE_DT,warmup_steps=80000,sampling_steps=240000,block_steps=10000,sample_every=10);rows.append(old)
        old=dict(rows[0]);old.update(id=reps[0]['id']+'__coarse_seed2',control='coarse_seed2',seed=20260929);rows.append(old)
        controls=len(rows)
        for r in meta['cases']:
            if r['split']=='test':
                v=dict(r);v.update(id=r['id']+'__refined_test',representative=r['id'],control='refined_test',nx=3500,ny=700,dt_s=cpu.campaign.REFERENCE_DT/7,ranks=64,phase='bezier_test',warmup_steps=560000,sampling_steps=1680000,block_steps=70000,sample_every=70)
                v.pop('fluid_cell_equivalents',None);v.pop('initial_particle_estimate',None);rows.append(v)
        assert controls==18 and len(rows)==24
        for i,r in enumerate(rows):r['index']=i
        m=dict(source=meta['source'],binary=meta['binary'],binary_sha256=meta['binary_sha256'],source_flowmllab_commit=meta['source_flowmllab_commit'],previous_campaign=str(prev),base=str(base),cases=rows,control_count=controls,jobs={},concurrency=concurrency,account=meta['account'],reserved_jobs=jobs,training_data_approved=False,policy='User authorized mesh/time independence, old_h25 diagnosis and six Bezier tests. Continuous sampling; unchanged collision and pressure boundary models. Constant physical warmup and sampling duration across timestep controls. Frozen test identities and network weights. Threshold pass is a sensitivity screen, not complete validation.')
        save(out/'manifest.json',m)
        wrapper='''#!/bin/bash
set -euo pipefail
cd "$STEP_CONTROL_OUT"
sha256sum -c code.sha256
module purge
module load openmpi/5.0.3
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TMPDIR=$(mktemp -d /tmp/step-control-${SLURM_JOB_ID}-XXXXXX)
export PRTE_MCA_prte_tmpdir_base="$TMPDIR" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
if [[ "$1" == review || "$1" == evaluate ]]; then
  exec "$STEP_CONTROL_PYTHON" code/step_controls_20260908.py "$1" --out "$STEP_CONTROL_OUT"
else
  exec python3 -I code/step_controls_20260908.py "$1" --out "$STEP_CONTROL_OUT"
fi
'''
        (out/'job.sh').write_text(wrapper)
        (out/'code.sha256').write_text(''.join(sha(p)+' '+str(p.relative_to(out))+'\n' for p in sorted((out/'code').glob('*')) if p.is_file())+sha(out/'job.sh')+' job.sh\n')
        with (out/'run_matrix.csv').open('w') as f:
            keys=['index','id','control','nx','ny','ppc','dt_s','warmup_steps','sampling_steps','block_steps','sample_every','seed'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
        pointer.write_text(str(out)+'\n')
        env=dict(os.environ,STEP_CONTROL_OUT=str(out),STEP_CONTROL_PYTHON=str(base/'runs/step-interpolation-20260908/venv/bin/python'))
        def queue(name,ranks,mem,wall,array=None,dep=None):
            cmd=['sbatch','--parsable','--account='+m['account'],'--partition=cpu-preempt','--constraint=x86_64','--nodes=1','--ntasks='+str(ranks),'--cpus-per-task=1','--mem='+mem,'--time='+wall,'--requeue','--job-name=step-control-'+name,'--chdir='+str(out),'--output='+str(out/(name+'-%A_%a.out')),'--error='+str(out/(name+'-%A_%a.err')),'--export=ALL']
            if array:cmd+=['--array='+array+'%'+str(concurrency)]
            if dep:cmd+=['--dependency=afterok:'+dep,'--kill-on-invalid-dep=yes']
            jid=subprocess.check_output(cmd+[str(out/'job.sh'),name],env=env,text=True).strip().split(';')[0];assert jid.isdigit();m['jobs'][name]=jid;save(out/'manifest.json',m);print(name,jid,flush=True);return jid
        a=queue('preflight',64,'128G','02:00:00')
        b=queue('controls',64,'128G','7-00:00:00','0-17',a)
        c=queue('review',1,'32G','06:00:00',dep=b)
        d=queue('bezier',64,'128G','7-00:00:00','18-23',c)
        queue('evaluate',1,'32G','06:00:00',dep=d)
        print('OUT='+str(out),flush=True)

def run_case(out,row,smoke=False):
    cpu=load_module(out/'code/cpu_campaign.py','existing_cpu');camp,bench,ck=cpu.campaign,cpu.bench,cpu.checkpoint
    m=read(out/'manifest.json');binary=m['binary']
    assert sha(binary)==m['binary_sha256']
    root=out/('preflight' if smoke else 'cases')/row['id'];root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=ck.completed(root)
        if done:return done
        a=root/'attempts';a.mkdir(exist_ok=True);idx=max([int(p.name) for p in a.iterdir() if p.name.isdigit()],default=0)+1;p=a/f'{idx:04d}'
        r=dict(row);warm=ck.latest_warm(root)
        if smoke:r.update(ppc=2,warmup_steps=20,sampling_steps=40,block_steps=10,sample_every=1,smoke=True)
        if warm:r['warmup_steps']=r['block_steps']
        camp.generate_case(p,r,warm,'retry' if warm else None,smoke=smoke)
        deck=(p/'in.step').read_text();b=r['block_steps'];s=r['sampling_steps']
        first=f'run {s-b}\ndump final';last=f'run {b}\nwrite_restart restart.final'
        assert deck.count(first)==1 and deck.count(last)==1
        deck=deck.replace(first,'dump final').replace(last,f'run {s}\nwrite_restart restart.final')
        assert [x for x in deck.splitlines() if x.startswith('run ')]==[f'run {r["warmup_steps"]}',f'run {s}']
        (p/'in.step').write_text(deck);ck.checkpoint_deck(p)
        timing=bench.execute(p,bench.command(binary,'cpu',ranks=64),timeout=6.9*86400)
        assert [v['steps'] for v in timing['loops']]==[r['warmup_steps'],s]
        report=bench.validate_case(p);save(p/'timing.json',timing)
        ck.commit_result(root,p,timing,report);print('COMPLETE',row['id'],flush=True)
        return p,timing,report

def field(p,area_weighted=True):
    import numpy as np,pandas as pd
    m=read(p/'case.json');L,H=m['L_m'],m['H_m'];d=pd.read_csv(p/'fields.csv.gz');cols=['rho_kg_m-3','u_m_s','v_m_s','Ttrans_K','p_Pa'];a=d[cols].to_numpy(float)
    good=(d.mean_particles.to_numpy()>0)&np.isfinite(a).all(1)
    ix=np.clip((d.x_m.to_numpy()/L*200).astype(int),0,199);iy=np.clip((d.y_m.to_numpy()/H*40).astype(int),0,39);bins=(iy*200+ix)[good]
    weight=d.area_m2.to_numpy()[good] if area_weighted else np.ones(good.sum())
    den=np.bincount(bins,weights=weight,minlength=8000);v=np.stack([np.bincount(bins,weights=a[good,k]*weight,minlength=8000) for k in range(5)],-1)/np.maximum(den[:,None],1e-100)
    return v.reshape(40,200,5),den.reshape(40,200)>0,m

def review(out):
    import numpy as np
    m=read(out/'manifest.json');cpu=load_module(out/'code/cpu_campaign.py','existing_cpu');results={}
    for r in m['cases'][:m['control_count']]:
        p,_,report=cpu.checkpoint.completed(out/'cases'/r['id']);results[r['id']]=(p,report)
    comparisons=[];passed=True
    for rep in dict.fromkeys(r['representative'] for r in m['cases'][:16]):
        for a,b in [('coarse','medium'),('medium','fine'),('fine','halfdt')]:
            pa,ra=results[rep+'__'+a];pb,rb=results[rep+'__'+b];fa,ma,_=field(pa);fb,mb,_=field(pb);mask=ma&mb
            norm=np.sqrt((fb[mask]**2).sum(0));err=np.sqrt(((fa-fb)[mask]**2).sum(0))/np.maximum(norm,1e-100)
            flow=abs(ra['mass_out_kg_per_m_s']/rb['mass_out_kg_per_m_s']-1)
            vabs=float(np.sqrt(np.mean(((fa-fb)[...,2][mask])**2))/max(np.sqrt(np.mean(fb[...,1][mask]**2)),1e-12))
            check=bool(np.all(err[[0,3,4]]<.01) and err[1]<.02 and vabs<.01 and flow<.01)
            if a!='coarse':passed=passed and check
            comparisons.append(dict(representative=rep,pair=[a,b],relative_l2=err.tolist(),v_rms_difference_over_reference_u_rms=vabs,mass_flow_relative_difference=flow,sensitivity_screen_pass=check))
        for level in ['fine','halfdt']:passed=passed and all(results[rep+'__'+level][1]['checks'].values())
    old_id=m['cases'][0]['representative']
    previous_root=Path(m['previous_campaign'])/'cases'/old_id
    previous=cpu.checkpoint.completed(previous_root)
    old_diagnostics={}
    def compact_report(r):
        blocks=[x['bulk_u_m_s'] for x in r['blocks']]
        return dict(mass_imbalance=r['mass_imbalance_fraction'],half_drift=r['bulk_velocity_half_drift_fraction'],pressure_errors=r['boundary_pressure_error_fraction'],last_block_u=blocks[-1],median_previous_block_u=float(np.median(blocks[:-1])),last_block_relative_deviation=float(abs(blocks[-1]/np.median(blocks[:-1])-1)))
    if previous:old_diagnostics['previous_segmented']=compact_report(previous[2])
    for level in ['continuous_original_dt','coarse','coarse_seed2','fine','halfdt']:
        old_diagnostics[level]=compact_report(results[old_id+'__'+level][1])
    old_diagnostics['interpretation']='Stochastic runs and differing MPI ranks are not bitwise paired. Compare pressure, stationarity and the final-block excursion. No automatic claim that removing the run boundary fixes the physical boundary condition.'
    reports={k:{a:b for a,b in v[1].items() if a in ['checks','mass_imbalance_fraction','bulk_velocity_half_drift_fraction','boundary_pressure_error_fraction','maximum_cell_over_minimum_sampled_lambda','maximum_dt_over_sampled_tau']} for k,v in results.items()}
    save(out/'control_review.json',dict(sensitivity_screen_pass=bool(passed),comparisons=comparisons,old_h25_diagnostics=old_diagnostics,reports=reports,training_data_approved=False,note='Single-seed comparisons except old_h25 coarse repeat. Neither threshold pass nor global norms establish complete DSMC validation. Bezier refinement/evaluation proceeds only after this screen. Equal-area bins used for physical comparisons; original frozen-model preprocessing retained for ML evaluation.'))
    print('SENSITIVITY_SCREEN_PASS='+str(bool(passed)),flush=True)
    if not passed:raise SystemExit(2)

def evaluate(out):
    import numpy as np,pandas as pd,torch
    m=read(out/'manifest.json');assert read(out/'control_review.json')['sensitivity_screen_pass']
    mod=load_module(out/'code/step_interpolation_train_v2.py','frozen_models');cpu=load_module(out/'code/cpu_campaign.py','existing_cpu')
    old=Path(m['base'])/'runs/step-interpolation-20260908';info=read(old/'data/bundle_manifest.json');z=np.load(old/'data/bundle.npz');torch.set_num_threads(1)
    nets={};scales={};checksums={}
    for name,cls in [('geom',mod.GeomDeepONet2D),('deeponet',mod.PlainDeepONet2D),('ufno',mod.UFNO2D)]:
        n=len(info['parameter_names']);net=cls(channels=4+n) if name=='ufno' else cls(n);net.load_state_dict(torch.load(old/name/'best.pt',map_location='cpu',weights_only=True));net.eval();nets[name]=net;scales[name]=np.load(old/name/'scaling.npz');checksums[name]=sha(old/name/'best.pt')
    rows=[]
    for r in m['cases'][18:]:
        p,_,report=cpu.checkpoint.completed(out/'cases'/r['id'])
        if not all(report['checks'].values()):raise RuntimeError('Bezier reference failed numerical gate: '+r['id'])
        truth,coverage,meta=field(p,False);g=meta['geometry'];L,H=meta['L_m'],meta['H_m'];section=(p/'step.surf').read_text().split('Points',1)[1].split('Lines',1)[0]
        wall=np.array([[float(t) for t in line.split()[1:3]] for line in section.splitlines() if line.strip()]);poly=np.vstack([wall,[[L,0],[L,H],[0,H]]])/L;xy=z['coordinates'];sdf=mod.polygon_sdf(xy.reshape(-1,2),poly).reshape(40,200);mask=sdf>0
        assert np.all(coverage[mask]);params=np.array([float(g['family']==k[7:]) if k.startswith('family_') else g[k] if k in g else meta[k] if k in meta else 0. for k in info['parameter_names']])
        train=[i for i in info['split']['train'] if info['records'][i]['geometry']['family']==g['family']];w=mod.hull_witness(z['parameters'][train],params);assert w is not None,'Not interpolation'
        q=np.concatenate([xy,sdf[...,None]],-1).astype(np.float32)
        for name,net in nets.items():
            s=scales[name];pt=torch.tensor(((params-s['parameter_mean'])/s['parameter_std'])[None],dtype=torch.float32);qt=torch.tensor(q[None]);mt=torch.tensor(mask[None],dtype=torch.float32)
            with torch.no_grad():
                if name=='ufno':pred=net(torch.cat([qt,mt[...,None],pt[:,None,None,:].expand(-1,40,200,-1)],-1).permute(0,3,1,2)).permute(0,2,3,1).numpy()[0]
                else:pred=net(pt,qt.flatten(1,2),mt.flatten(1,2)).reshape(40,200,5).numpy()
            pred=pred*s['std']+s['mean'];e=np.sqrt((((pred-truth)*mask[...,None])**2).sum((0,1)))/np.maximum(np.sqrt(((truth*mask[...,None])**2).sum((0,1))),1e-12)
            rows.append(dict(case=r['id'],model=name,relative_l2=e.tolist(),interpolation_witness=dict(train_indices=train,weights=w.tolist())))
            np.savez_compressed(out/(r['id']+'_'+name+'.npz'),prediction=pred,truth=truth,mask=mask,coordinates=xy)
    save(out/'bezier_frozen_evaluation.json',dict(results=rows,weights_sha256=checksums,training_data_approved=False,note='Frozen coarse-data-trained networks evaluated on six preselected refined references. No retraining or test-based selection. Sensitivity screen is not complete physical validation.'))
    print('SIX_BEZIER_FROZEN_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['submit','preflight','controls','review','bezier','evaluate']);ap.add_argument('--base');ap.add_argument('--out');a=ap.parse_args()
    if a.action=='submit':submit(a.base)
    else:
        out=Path(a.out).resolve();m=read(out/'manifest.json')
        if a.action=='preflight':
            for r in m['cases']:run_case(out,r,True)
            save(out/'PREFLIGHT_PASS.json',dict(cases=len(m['cases']),continuous_sampling=True))
        elif a.action in ['controls','bezier']:
            assert (out/'PREFLIGHT_PASS.json').exists();run_case(out,m['cases'][int(os.environ['SLURM_ARRAY_TASK_ID'])])
        elif a.action=='review':review(out)
        else:evaluate(out)
