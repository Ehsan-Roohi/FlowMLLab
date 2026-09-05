#!/usr/bin/env python3
"""Pinned SPARTA step campaign: validation, geometry sweep, status and review packs.

Submission and postprocessing use Python's standard library only. This module
does not infer scientific success from a Slurm collector's exit status.
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

_spec = importlib.util.spec_from_file_location('step_pilot', Path(__file__).with_name('pilot.py'))
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)
BASE = pilot.DEFAULT_BASE
PILOT_RUN = 'step-pilot-20260905T142106Z-8xo_trzs'
SEEDS = [20260905, 20260917, 20260929]
HEIGHTS = [16, 21, 25, 30, 33, 36, 40, 44, 50, 58, 60, 64, 67, 70, 75]
REFERENCE_DT = 3.589201579675864e-11
COMMON_DT = REFERENCE_DT / 3.5
CODE_FILES = ['campaign.py', 'campaign_job.sh', 'submit_campaign.sh', 'campaign_matrix.csv',
              'verify_campaign.py', 'CAMPAIGN.md', 'pilot.py', 'unity_job.sh',
              'verify_local.py', 'README.md', 'VALIDATION.md']
POINTER = 'LATEST_SPARTA_STEP_CAMPAIGN'


def read_json(path):
    return json.loads(Path(path).read_text())


def matrix():
    rows = []
    for level, nx, ny, ranks, memory, wall in [
        ('coarse', 1000, 200, 16, 48, '1-00:00:00'),
        ('medium', 2000, 400, 32, 96, '3-00:00:00'),
        ('fine', 3500, 700, 64, 192, '7-00:00:00'),
    ]:
        for seed in SEEDS:
            rows.append(dict(id=f'grid_{level}_s{seed}', phase='validation',
                             group=level, height_percent=50, seed=seed, nx=nx,
                             ny=ny, ppc=20, dt_s=COMMON_DT, ranks=ranks,
                             memory_gib=memory, walltime=wall))
    fine = rows[-3].copy()
    for name, change in [('halfdt', {'dt_s': COMMON_DT/2}), ('ppc40', {'ppc':40})]:
        rows.append(dict(fine, id=f'fine_{name}_s{SEEDS[0]}', **change))
    for height in HEIGHTS:
        if height == 50:
            continue  # already has three fine-grid independent seeds
        for seed in SEEDS[:2]:
            rows.append(dict(fine, id=f'geometry_h{height:02d}_s{seed}',
                             phase='geometry', group='geometry',
                             height_percent=height, seed=seed, memory_gib=128))
    for i, row in enumerate(rows):
        scale = REFERENCE_DT/row['dt_s']
        row.update(index=i, warmup_steps=round(80000*scale),
                   sampling_steps=round(120000*scale), block_steps=round(10000*scale),
                   sample_every=round(10*scale))
        fluid = row['nx']*row['ny']*(1-.3*row['height_percent']/100)
        row['initial_simulated_particles_estimate'] = round(fluid*row['ppc']*1.5)
        # A sizing estimate from job 64013621; ideal MPI scaling is NOT a measurement.
        row['ideal_scaling_wall_hours_estimate'] = (3732.684/3600 * fluid/170000
            * row['ppc']/20 * (row['warmup_steps']+row['sampling_steps'])/60000
            * 16/row['ranks'])
    assert len(rows) == 39 and len({r['id'] for r in rows}) == 39
    return rows


def print_matrix(rows):
    print('CASE                           H%    GRID        PPC  RANKS  EST_HOURS')
    for r in rows:
        print(f"{r['id']:<30} {r['height_percent']:3d} {r['nx']:4d}x{r['ny']:<4d} "
              f"{r['ppc']:3d} {r['ranks']:6d} {r['ideal_scaling_wall_hours_estimate']:10.1f}")
    core_hours = sum(r['ideal_scaling_wall_hours_estimate']*r['ranks'] for r in rows)
    print(f'CASES={len(rows)} VALIDATION=11 GEOMETRY=28 DISTINCT_HEIGHTS={len(HEIGHTS)}')
    print(f'IDEAL_SCALING_CORE_HOURS_ESTIMATE={core_hours:.0f}')
    print('ESTIMATE_NOTE=pilot extrapolation only; queue time and MPI scaling are unmeasured')


def sampling_deck(m):
    b, stride, warm, sample = (m[k] for k in ['block_steps', 'sample_every', 'warmup_steps', 'sampling_steps'])
    final = warm+sample
    return f'''compute boundary boundary gas n press shx ke
variable exits_in equal c_boundary[1][1]
variable exits_out equal c_boundary[2][1]
variable inventory equal np
fix check grid/check {b} error
stats {b}
stats_style step cpu np ncoll nattempt nexit f_inlet[2] f_outlet[2]
print "SPARTA_STEP_WARMUP_BEGIN"
run {warm}
write_restart restart.warm
print "SPARTA_CAMPAIGN_WARM_RESTART_READY"
print "SPARTA_STEP_SAMPLING_BEGIN"
fix flux ave/time 1 {b} {b} f_inlet[1] f_outlet[1] v_exits_in v_exits_out v_inventory file flux.blocks
compute gasfield grid all gas n nrho massrho u v w trot pxrho pyrho
compute thermal thermal/grid all gas temp press
fix blocks ave/grid all {stride} {b//stride} {b} c_gasfield[*] c_thermal[*] ave one
fix average ave/grid all {stride} {b//stride} {b} c_gasfield[*] c_thermal[*] ave running
dump blocks grid all {b} grid.block.*.gz id xc yc vol f_blocks[*]
dump_modify blocks first no format float %.16g
compute surface surf all gas n press shx shy ke
fix wallavg ave/surf all 1 {b} {b} c_surface[*] ave running
dump walls surf all {b} wall.running.*.gz id f_wallavg[*]
dump_modify walls first no format float %.16g
fix boxwalls ave/time 1 {b} {b} c_boundary[*] mode vector file boundary.blocks
run {sample-b}
dump final grid all {final} grid.final.gz id xc yc vol f_average[*]
dump_modify final first no format float %.16g
run {b}
write_restart restart.final
print "SPARTA_STEP_SOLVER_COMPLETE"
'''


def generate_case(out, row, restart=None, restart_kind=None, smoke=False):
    """Restart means a new sampling window; no running tally is silently continued."""
    out = Path(out)
    if out.exists():
        raise ValueError(f'Refusing to overwrite case: {out}')
    m = pilot.generate(out, smoke=smoke, ratio=row['height_percent']/100,
                       nx=row['nx'], ny=row['ny'], ppc=row['ppc'],
                       warmup=row['warmup_steps'], sample=row['sampling_steps'], seed=row['seed'])
    maximum_dt = m['dt_s']
    if not 0 < row['dt_s'] <= maximum_dt*(1+1e-12):
        raise ValueError('Campaign timestep exceeds the pilot collision/flight bound')
    m.update(level='campaign_smoke' if smoke else f"campaign_{row['phase']}",
             campaign_case_id=row['id'], campaign_index=row.get('index'),
             dt_s=row['dt_s'], block_steps=row['block_steps'],
             sample_every=row['sample_every'], warmup_steps=row['warmup_steps'],
             sampling_steps=row['sampling_steps'], fresh_seed=row['seed'],
             independent_seed_replicate=True, restart_source=None)
    if restart:
        restart = Path(restart).resolve()
        if not restart.is_file() or restart.stat().st_size == 0:
            raise ValueError(f'Restart not found: {restart}')
        previous = read_json(restart.parent/'case.json')
        for key in ['nx','ny','h_over_H','p_in_Pa','p_out_Pa','fnum_unit_depth',
                    'T_in_K','T_wall_K','sparta_commit','physics']:
            if previous[key] != m[key]:
                raise ValueError(f'Restart case mismatch: {key}')
        if restart_kind == 'pilot':
            if previous['final_step'] != 60000 or previous['seed'] != SEEDS[0]:
                raise ValueError('Unexpected pilot restart provenance')
            m['warmup_steps'] = round(20000*REFERENCE_DT/m['dt_s'])
        elif restart_kind == 'retry':
            m['warmup_steps'] = m['block_steps']
        elif restart_kind != 'smoke':
            raise ValueError('Unknown restart kind')
        m.update(restart_source=str(restart), restart_sha256=pilot.sha(restart),
                 restart_kind=restart_kind, independent_seed_replicate=(restart_kind=='pilot'),
                 restart_note='Clock reset; collisions and all tallies recreated; new warmup and sampling window.')
        # Restart restores species/mixtures/geometry, but NOT collision or wall models.
        # The read-time balance option avoids the known dispersed-grid ghost failure.
        prefix = f'''seed {row['seed']}
read_restart "{restart}" balance rcb cell
reset_timestep 0
surf_collide wall diffuse {m['T_wall_K']} 1.0
surf_modify all collide wall
bound_modify ylo yhi collide wall
collide vss gas nitrogen.vss
collide_modify rotate smooth vibrate no
fix inlet emit/face gas xlo subsonic {m['p_in_Pa']:.17g} {m['T_in_K']}
fix outlet emit/face gas xhi subsonic {m['p_out_Pa']:.17g} {m['T_out_injected_K']}
'''
    else:
        old = (out/'in.step').read_text()
        prefix, tail = old.split('fix check grid/check ', 1)
        assert 'create_grid ' in prefix and 'balance_grid rcb cell\n' in prefix
    b, stride = m['block_steps'], m['sample_every']
    assert m['warmup_steps'] % b == 0 and m['sampling_steps'] % b == 0
    assert b % stride == 0 and m['sampling_steps'] >= 4*b
    m['final_step'] = m['warmup_steps']+m['sampling_steps']
    m['sampling_time_s'] = m['sampling_steps']*m['dt_s']
    m['warmup_time_s'] = m['warmup_steps']*m['dt_s']
    m['cell_moment_sampling_time_s'] = stride*m['dt_s']
    pilot.write_json(out/'case.json', m)
    (out/'in.step').write_text(prefix+f"timestep {m['dt_s']:.17g}\n"+sampling_deck(m))
    return m


def diagnostic_probes(path, m, write_path=None):
    """Finite-volume bins common to every campaign mesh; no fitted/interpolated field."""
    bins = {}
    for row in pilot.read_dump(path):
        if isinstance(row, tuple):
            continue
        _, x, y, a, n, nrho, rho, u, v, w, tr, ru, rv, t, p = row
        if a <= 0 or (x < m['step_x_m'] and y < m['step_height_m']):
            continue
        if rho <= 0:
            raise ValueError('Nonpositive sampled fluid density in comparison probes')
        i = min(99, int(100*x/m['L_m'])); j = min(99, int(100*y/m['H_m']))
        z = bins.setdefault((i,j), [0.]*5)
        z[0] += a; z[1] += rho*a; z[2] += ru*a; z[3] += rv*a; z[4] += p*a
    result=[]
    for (i,j), (a,mass,px,py,pres) in sorted(bins.items()):
        result.append([i,j,(i+.5)*m['L_m']/100,(j+.5)*m['H_m']/100,a,px/mass,py/mass,pres/a])
    if write_path:
        with open(write_path,'w',newline='') as f:
            writer=csv.writer(f); writer.writerow(['i','j','x_m','y_m','area_m2','u_m_s','v_m_s','p_Pa']); writer.writerows(result)
    return result


def summarize_case(out):
    out=Path(out); m=read_json(out/'case.json')
    with contextlib.redirect_stdout(io.StringIO()):
        result=pilot.report(out)
    diagnostic_probes(out/'grid.final.gz', m, out/'comparison_probes.csv')
    regional=[]
    for b in result['blocks']:
        reverse=0.
        for row in pilot.read_dump(out/f"grid.block.{b['step']}.gz"):
            if isinstance(row,tuple): continue
            if 0 < row[1]-m['step_x_m'] < 3*m['step_height_m'] and row[2] < m['step_height_m']:
                reverse += max(-row[11],0)*row[3]
        regional.append(reverse)
        b['reverse_axial_momentum_integral']=reverse
    half=len(regional)//2; average=statistics.fmean(regional)
    reverse_drift=(abs(statistics.fmean(regional[:half])-statistics.fmean(regional[half:]))/average
                   if average > 1e-30 else None)
    result.update(status='campaign_case_execution_complete',
                  campaign_case_id=m['campaign_case_id'],
                  regional_reverse_half_drift_fraction=reverse_drift,
                  regional_diagnostic='Integral of max(-rho*u,0) over 0<x-x_s<3h, 0<y<h; not a reattachment length.',
                  sampling_time_s=m['sampling_time_s'], restart_source=m['restart_source'],
                  training_data_approved=False, bird_parity_validated=False)
    pilot.write_json(out/'report.json',result)
    (out/'CASE_COMPLETE').write_text(m['campaign_case_id']+'\n')
    paths=[p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS']
    (out/'SHA256SUMS').write_text(''.join(f'{pilot.sha(p)}  {p.name}\n' for p in sorted(paths)))
    print(f"CAMPAIGN_CASE_COMPLETE id={m['campaign_case_id']} mass_imbalance={result['mass_imbalance_fraction']:.6g} "
          f"bulk_drift={result['bulk_velocity_half_drift_fraction']:.6g} TRAINING_DATA_APPROVED=False",flush=True)
    return result


def run_case(out, index, binary, launcher='mpirun', retry_from=None):
    out=Path(out); meta=read_json(out/'manifest.json'); row=meta['cases'][index]
    case_root=out/'cases'/row['id']; case_root.mkdir(parents=True,exist_ok=True)
    lock=case_root/'RUNNING.lock'
    try: fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError: raise ValueError(f'Case already locked: {lock}; inspect the owning job before retrying')
    try:
        with os.fdopen(fd,'w') as f: f.write(os.environ.get('SLURM_JOB_ID','local')+'\n')
        if (case_root/'COMPLETED_ATTEMPT').exists():
            raise ValueError('Refusing to rerun an already completed case')
        if not retry_from and os.environ.get('SPARTA_CAMPAIGN_RETRY')=='1' and (case_root/'LATEST_ATTEMPT').exists():
            prior=Path((case_root/'LATEST_ATTEMPT').read_text().strip())
            if ((prior/'restart.warm').is_file() and (prior/'log.sparta').is_file()
                and 'SPARTA_CAMPAIGN_WARM_RESTART_READY' in (prior/'log.sparta').read_text()):
                retry_from=prior
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=case_root))
        (case_root/'LATEST_ATTEMPT').write_text(str(attempt)+'\n')
        restart=None; kind=None
        if retry_from:
            restart=Path(retry_from)/'restart.warm'; kind='retry'
            previous=read_json(Path(retry_from)/'case.json')
            if previous['campaign_case_id'] != row['id']:
                raise ValueError('Retry belongs to a different campaign case')
            if 'SPARTA_CAMPAIGN_WARM_RESTART_READY' not in (Path(retry_from)/'log.sparta').read_text():
                raise ValueError('Warm restart completion was not confirmed')
        elif row['id'] == f'grid_coarse_s{SEEDS[0]}':
            candidate=Path(meta['pilot_run'])/'pilot'/'restart.final'
            if candidate.is_file():
                restart=candidate; kind='pilot'
        generate_case(attempt,row,restart,kind)
        ranks=int(os.environ.get('SLURM_NTASKS',row['ranks']))
        if ranks != row['ranks']: raise ValueError('Allocated MPI ranks differ from the manifest')
        pilot.write_json(attempt/'execution.json',{'job_id':os.environ.get('SLURM_JOB_ID'),
                         'ranks':ranks,'binary':str(binary),'binary_sha256':pilot.sha(binary),
                         'mpi_launcher':str(launcher),'campaign_commit':meta['flowmllab_commit']})
        with open(attempt/'in.step') as inp:
            subprocess.run([str(launcher),'-np',str(ranks),str(binary)],cwd=attempt,stdin=inp,check=True)
        summarize_case(attempt)
        (case_root/'COMPLETED_ATTEMPT').write_text(str(attempt)+'\n')
    finally:
        lock.unlink(missing_ok=True)


def completed_attempt(out, row):
    root=Path(out)/'cases'/row['id']
    p=Path((root/'COMPLETED_ATTEMPT').read_text().strip())
    if not (p/'CASE_COMPLETE').is_file(): raise ValueError(f'Missing case marker: {p}')
    report=read_json(p/'report.json')
    if report['campaign_case_id'] != row['id'] or report['status']!='campaign_case_execution_complete':
        raise ValueError(f'Wrong report identity/status: {p}')
    return p, report


def ensemble_probes(out, rows):
    values=[]
    for row in rows:
        path,_=completed_attempt(out,row)
        with open(path/'comparison_probes.csv') as f:
            values.append({(int(v['i']),int(v['j'])):v for v in csv.DictReader(f)})
    keys=set(values[0])
    if any(set(v)!=keys for v in values): raise ValueError('Probe grid mismatch')
    result={}
    for key in sorted(keys):
        vals=[v[key] for v in values]
        result[key]={name:statistics.fmean(float(z[name]) for z in vals)
                     for name in ['x_m','y_m','area_m2','u_m_s','v_m_s','p_Pa']}
    return result


def field_difference(a,b):
    if set(a)!=set(b): raise ValueError('Cannot compare different probe domains')
    sums={name:[0.,0.] for name in ['velocity_global','velocity_region','pressure']}
    h=85.47e-6/10; sx=.3*85.47e-6
    for key in b:
        p,q=a[key],b[key]; area=q['area_m2']
        diff=(p['u_m_s']-q['u_m_s'])**2+(p['v_m_s']-q['v_m_s'])**2
        norm=q['u_m_s']**2+q['v_m_s']**2
        for name in ['velocity_global']+(['velocity_region'] if 0<q['x_m']-sx<3*h and q['y_m']<h else []):
            sums[name][0]+=area*diff; sums[name][1]+=area*norm
        sums['pressure'][0]+=area*(p['p_Pa']-q['p_Pa'])**2
        sums['pressure'][1]+=area*q['p_Pa']**2
    return {name:math.sqrt(v[0]/max(v[1],1e-100)) for name,v in sums.items()}


def validation_gate(out):
    out=Path(out); meta=read_json(out/'manifest.json'); rows=meta['cases'][:11]
    reports={row['id']:completed_attempt(out,row)[1] for row in rows}
    checks={}; thresholds=meta['validation_thresholds']
    for row in rows:
        r=reports[row['id']]; prefix=row['id']
        checks[prefix+':flow'] = r['checks']['positive_net_flow']
        checks[prefix+':mass'] = r['mass_imbalance_fraction']<thresholds['mass_imbalance_fraction']
        checks[prefix+':stationarity_bulk'] = r['bulk_velocity_half_drift_fraction']<thresholds['bulk_half_drift_fraction']
        rd=r['regional_reverse_half_drift_fraction']
        checks[prefix+':stationarity_regional'] = rd is not None and rd<thresholds['reverse_half_drift_fraction']
        checks[prefix+':boundary'] = max(r['boundary_pressure_error_fraction'])<.1
        checks[prefix+':timestep'] = r['maximum_dt_over_sampled_tau']<.25
        if row['group']=='fine':
            checks[prefix+':spatial_resolution'] = r['maximum_cell_over_minimum_sampled_lambda']<=1/3
    differences={}
    fine=ensemble_probes(out,rows[6:9])
    for name, selected, reference in [
        ('coarse_vs_fine',rows[:3],fine),('medium_vs_fine',rows[3:6],fine),
        ('halfdt_vs_fine_seed0',[rows[9]],ensemble_probes(out,[rows[6]])),
        ('ppc40_vs_fine_seed0',[rows[10]],ensemble_probes(out,[rows[6]])),
    ]:
        differences[name]=field_difference(ensemble_probes(out,selected),reference)
        if name!='coarse_vs_fine':
            for metric,value in differences[name].items():
                checks[name+':'+metric]=value<thresholds[metric]
    passed=all(checks.values())
    result={'status':'validation_screen_passed' if passed else 'validation_screen_failed',
            'geometry_sweep_released':passed,'checks':checks,'differences':differences,
            'thresholds':thresholds,'training_data_approved':False,'bird_parity_validated':False,
            'interpretation':'Predeclared engineering screen, not statistical equivalence or Bird validation. Geometry rows still require their own diagnostics.'}
    pilot.write_json(out/'validation_report.json',result)
    print(json.dumps(result,indent=2))
    if not passed:
        print('GEOMETRY_SWEEP_NOT_RELEASED: inspect validation_report.json',file=sys.stderr)
        return 3
    (out/'VALIDATION_PASS').write_text(meta['flowmllab_commit']+'\n')
    print('SPARTA_CAMPAIGN_VALIDATION_PASS')
    return 0


def submit(root, base, ref, account='pi_roohie_umass_edu', module='openmpi/5.0.3',
           new_campaign=False, min_free_gib=500):
    if not re.fullmatch('[0-9a-f]{40}',ref): raise ValueError('A full commit SHA is required')
    root,base=Path(root).resolve(),Path(base).resolve()
    (base/'runs').mkdir(parents=True,exist_ok=True)
    for name in CODE_FILES:
        if not (root/name).is_file(): raise ValueError(f'Missing immutable code file: {name}')
    pointer=base/POINTER
    if pointer.exists() and not new_campaign:
        raise ValueError(f'Campaign already recorded at {pointer}; use status/retry instead of duplicate submission, or --new-campaign intentionally')
    free=shutil.disk_usage(base).free/2**30
    if free<min_free_gib: raise ValueError(f'Only {free:.1f} GiB filesystem space available; campaign requires {min_free_gib} GiB headroom')
    out=Path(tempfile.mkdtemp(prefix=time.strftime('step-campaign-%Y%m%dT%H%M%SZ-',time.gmtime()),dir=base/'runs'))
    code=out/'code'; code.mkdir()
    for name in CODE_FILES: shutil.copy2(root/name,code/name)
    (out/'code.sha256').write_text(''.join(f'{pilot.sha(code/name)}  code/{name}\n' for name in CODE_FILES))
    rows=matrix()
    meta={'status':'submitting','flowmllab_commit':ref,'sparta_commit':pilot.SPARTA_COMMIT,
          'out':str(out),'pilot_run':str(base/'runs'/PILOT_RUN),'account':account,
          'openmpi_module':module,'cases':rows,'jobs':{},'training_data_approved':False,
          'validation_thresholds':{'mass_imbalance_fraction':.02,'bulk_half_drift_fraction':.02,
              'reverse_half_drift_fraction':.10,'velocity_global':.05,'velocity_region':.10,'pressure':.03},
          'storage_note':'500 GiB filesystem headroom; this is not a user-quota check. Preserve raw block fields and restart files.',
          'runtime_estimate_note':'Extrapolated from Unity job 64013621 with ideal rank scaling; not a throughput promise.'}
    pilot.write_json(out/'manifest.json',meta); pointer.write_text(str(out)+'\n')
    with open(out/'run_matrix.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print_matrix(rows)
    env=dict(os.environ,SPARTA_CAMPAIGN_OUT=str(out),SPARTA_STEP_OUT=str(out),SPARTA_STEP_MPI_MODULE=module)
    common=['sbatch','--parsable',f'--account={account}','--partition=cpu','--nodes=1',
            '--constraint=x86_64','--export=ALL','--cpus-per-task=1',f'--chdir={out}']
    def queue(phase,ranks,memory,wall,indices=None,parallel=None,dependencies=()):
        cmd=common+[f'--job-name=step-camp-{phase}',f'--ntasks={ranks}',f'--mem={memory}G',
                    f'--time={wall}',f'--output={out}/{phase}-%A_%a.out',f'--error={out}/{phase}-%A_%a.err']
        if indices is not None: cmd += ['--array='+','.join(map(str,indices))+f'%{parallel}']
        if dependencies: cmd += ['--dependency=afterok:'+':'.join(meta['jobs'][k] for k in dependencies),'--kill-on-invalid-dep=yes']
        raw=subprocess.check_output(cmd+[str(code/'campaign_job.sh'),phase],env=env,text=True).strip()
        job=raw.split(';')[0]
        if not job.isdigit(): raise ValueError(f'Invalid sbatch job ID: {raw!r}')
        meta['jobs'][phase]=job;pilot.write_json(out/'manifest.json',meta)
        print(f'{phase.upper()}_JOB={job}',flush=True)
    try:
        queue('build',8,16,'00:45:00')
        queue('coarse',16,48,'1-00:00:00',[0,1,2],1,['build'])
        queue('medium',32,96,'3-00:00:00',[3,4,5],1,['build'])
        queue('fine',64,192,'7-00:00:00',[6,7,8,9,10],3,['build'])
        queue('gate',1,8,'00:30:00',dependencies=['coarse','medium','fine'])
        queue('geometry',64,128,'7-00:00:00',list(range(11,len(rows))),4,['gate'])
        queue('collect',1,8,'01:00:00',dependencies=['geometry'])
        meta['status']='submitted';pilot.write_json(out/'manifest.json',meta)
    except Exception:
        meta['status']='partial_submission';pilot.write_json(out/'manifest.json',meta)
        print(f'PARTIAL_SUBMISSION_RECORDED={out}',file=sys.stderr)
        raise
    print(f'OUT={out}\nSPARTA_CAMPAIGN_SUBMITTED VALIDATION=11 GEOMETRY=28')


def collect(out):
    out=Path(out); meta=read_json(out/'manifest.json'); summaries=[]
    for row in meta['cases']:
        path,r=completed_attempt(out,row)
        summaries.append({'id':row['id'],'height_percent':row['height_percent'],'seed':row['seed'],
                          'path':str(path),'mass_imbalance_fraction':r['mass_imbalance_fraction'],
                          'bulk_half_drift_fraction':r['bulk_velocity_half_drift_fraction'],
                          'regional_reverse_half_drift_fraction':r['regional_reverse_half_drift_fraction'],
                          'max_cell_over_lambda':r['maximum_cell_over_minimum_sampled_lambda']})
    result={'status':'campaign_execution_complete','cases_completed':len(summaries),'cases':summaries,
            'validation':read_json(out/'validation_report.json'),'training_data_approved':False,
            'bird_parity_validated':False}
    pilot.write_json(out/'campaign_report.json',result)
    with open(out/'case_summary.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (out/'CAMPAIGN_COMPLETE').write_text('39 solver cases and reports complete; scientific release remains unapproved.\n')
    pack(out)
    print('SPARTA_CAMPAIGN_COMPLETE CASES=39 TRAINING_DATA_APPROVED=False')


def resume(out):
    """Resubmit only incomplete solver cases after all recorded jobs are terminal."""
    out=Path(out).resolve();meta=read_json(out/'manifest.json')
    history=meta.setdefault('job_history',[])
    all_jobs=set(meta['jobs'].values())|{j for event in history for j in event['jobs'].values()}
    active=subprocess.check_output(['squeue','--me','-h','-o','%i'],text=True).splitlines()
    active_parents={v.strip().split('_')[0] for v in active}
    if active_parents & all_jobs:
        raise ValueError('Campaign still has active/pending jobs; resume would duplicate work')
    missing=[r for r in meta['cases'] if not (out/'cases'/r['id']/'COMPLETED_ATTEMPT').exists()]
    if (out/'CAMPAIGN_COMPLETE').exists():
        raise ValueError('Campaign already completed')
    if ((out/'validation_report.json').exists()
        and read_json(out/'validation_report.json')['status']=='validation_screen_failed'
        and not any(r['phase']=='validation' for r in missing)):
        raise ValueError('All control runs completed but the scientific screen failed; repeating submission cannot fix this')
    lock=out/'RESUME.lock'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    try:
        for row in missing:
            # No recorded campaign job is active; retain logs but remove stale locks.
            (out/'cases'/row['id']/'RUNNING.lock').unlink(missing_ok=True)
        history.append({'status':meta['status'],'jobs':meta['jobs'].copy()})
        meta['jobs']={};meta['status']='resubmitting';pilot.write_json(out/'manifest.json',meta)
        env=dict(os.environ,SPARTA_CAMPAIGN_OUT=str(out),SPARTA_STEP_OUT=str(out),
                 SPARTA_STEP_MPI_MODULE=meta['openmpi_module'],SPARTA_CAMPAIGN_RETRY='1')
        def queue(phase,ranks,memory,wall,indices=None,parallel=1,dependencies=()):
            cmd=['sbatch','--parsable',f"--account={meta['account']}",'--partition=cpu','--nodes=1',
                 '--constraint=x86_64','--export=ALL','--cpus-per-task=1',f'--chdir={out}',
                 f'--job-name=step-camp-{phase}',f'--ntasks={ranks}',f'--mem={memory}G',f'--time={wall}',
                 f'--output={out}/{phase}-%A_%a.out',f'--error={out}/{phase}-%A_%a.err']
            if indices is not None:cmd+=['--array='+','.join(map(str,indices))+f'%{parallel}']
            if dependencies:cmd+=['--dependency=afterok:'+':'.join(meta['jobs'][d] for d in dependencies),'--kill-on-invalid-dep=yes']
            raw=subprocess.check_output(cmd+[str(out/'code'/'campaign_job.sh'),phase],env=env,text=True).strip()
            job=raw.split(';')[0]
            if not job.isdigit():raise ValueError(f'Invalid sbatch response: {raw!r}')
            meta['jobs'][phase]=job;pilot.write_json(out/'manifest.json',meta);print(f'{phase.upper()}_JOB={job}',flush=True)
        build=[]
        if not (out/'PREFLIGHT_PASS').exists():queue('build',8,16,'00:45:00');build=['build']
        controls=[]
        for group,ranks,mem,wall,parallel in [('coarse',16,48,'1-00:00:00',1),('medium',32,96,'3-00:00:00',1),('fine',64,192,'7-00:00:00',3)]:
            indices=[r['index'] for r in missing if r['phase']=='validation' and r['group']==group]
            if indices:queue(group,ranks,mem,wall,indices,parallel,build);controls.append(group)
        gate=[]
        if controls or not (out/'VALIDATION_PASS').exists():
            queue('gate',1,8,'00:30:00',dependencies=controls or build);gate=['gate']
        indices=[r['index'] for r in missing if r['phase']=='geometry']
        if indices:queue('geometry',64,128,'7-00:00:00',indices,4,gate or build)
        queue('collect',1,8,'01:00:00',dependencies=['geometry'] if indices else gate or build)
        meta['status']='submitted';pilot.write_json(out/'manifest.json',meta)
        print(f'CAMPAIGN_RESUBMITTED INCOMPLETE_CASES={len(missing)} OUT={out}')
    except Exception:
        meta['status']='partial_submission';pilot.write_json(out/'manifest.json',meta);raise
    finally:lock.unlink(missing_ok=True)


def status(out):
    out=Path(out); meta=read_json(out/'manifest.json');print(f'OUT={out}\nSUBMISSION_STATUS={meta["status"]}')
    jobs=','.join(dict.fromkeys([j for event in meta.get('job_history',[]) for j in event['jobs'].values()]+list(meta['jobs'].values())))
    if jobs:
        subprocess.run(['sacct','-j',jobs,'--format=JobID%24,JobName%22,State%22,Elapsed,ExitCode,MaxRSS,NodeList%20'],check=False)
    completed=0
    for row in meta['cases']:
        case=out/'cases'/row['id']
        if (case/'COMPLETED_ATTEMPT').exists():
            path,r=completed_attempt(out,row);completed+=1
            print(f"CASE_COMPLETE {row['id']} mass={100*r['mass_imbalance_fraction']:.3f}% bulk_drift={100*r['bulk_velocity_half_drift_fraction']:.3f}%")
        elif (case/'LATEST_ATTEMPT').exists():
            print(f"CASE_INCOMPLETE {row['id']} path={(case/'LATEST_ATTEMPT').read_text().strip()}")
    print(f'COMPLETED_CASES={completed}/{len(meta["cases"])}')
    if (out/'validation_report.json').exists():
        r=read_json(out/'validation_report.json');print('VALIDATION_STATUS='+r['status'])
        for name,passed in r['checks'].items():
            if not passed: print('VALIDATION_CHECK_FAILED='+name)
    for p in sorted(out.glob('*.err')):
        if p.stat().st_size: print(f'LOG={p}\n'+'\n'.join(p.read_text(errors='replace').splitlines()[-8:]))


def pack(out, case_id=None):
    out=Path(out);meta=read_json(out/'manifest.json'); files=[]
    for name in ['manifest.json','run_matrix.csv','validation_report.json','campaign_report.json','case_summary.csv','code.sha256']:
        if (out/name).is_file(): files.append(out/name)
    files += [out/'code'/name for name in CODE_FILES]
    files += list(out.glob('*.out'))+list(out.glob('*.err'))
    if case_id and case_id not in {row['id'] for row in meta['cases']}:
        raise ValueError('Unknown case ID')
    for row in meta['cases']:
        if case_id and row['id']!=case_id: continue
        try: path,_=completed_attempt(out,row)
        except FileNotFoundError: continue
        names=['case.json','report.json','execution.json','comparison_probes.csv',
               'axial_profiles.csv','flux.blocks','boundary.blocks','log.sparta']
        if case_id: names+=['fields.csv.gz','grid.final.gz']
        files += [path/name for name in names if (path/name).is_file()]
    dest=out/(f'review_{case_id}.tar.gz' if case_id else 'campaign_review.tar.gz')
    with tarfile.open(dest,'w:gz') as tf:
        for path in files: tf.add(path,arcname=str(path.relative_to(out)),recursive=False)
    print(f'REVIEW_ARCHIVE={dest} BYTES={dest.stat().st_size}')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('plan')
    s=sub.add_parser('submit');s.add_argument('--root',default=str(Path(__file__).parent));s.add_argument('--base',default=BASE)
    s.add_argument('--ref',required=True);s.add_argument('--account',default='pi_roohie_umass_edu');s.add_argument('--module',default='openmpi/5.0.3')
    s.add_argument('--new-campaign',action='store_true');s.add_argument('--min-free-gib',type=float,default=500)
    for name in ['status','gate','collect','pack','resume']:
        q=sub.add_parser(name);q.add_argument('--out',required=True)
        if name=='pack':q.add_argument('--case-id')
    q=sub.add_parser('run-case');q.add_argument('--out',required=True);q.add_argument('--index',type=int,required=True)
    q.add_argument('--binary',required=True);q.add_argument('--launcher',default='mpirun');q.add_argument('--retry-from')
    args=vars(p.parse_args());action=args.pop('action')
    if action=='plan':print_matrix(matrix());return 0
    if action=='gate':return validation_gate(**args)
    {'submit':submit,'status':status,'collect':collect,'pack':pack,'run-case':run_case,'resume':resume}[action](**args)
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,subprocess.CalledProcessError) as exc:
        print(f'SPARTA_CAMPAIGN_FAILED: {exc}',file=sys.stderr);sys.exit(1)
