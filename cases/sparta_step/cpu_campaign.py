#!/usr/bin/env python3
"""Pilot-budget CPU geometry campaign; execution receipts do not approve training data."""
import argparse
import contextlib
import csv
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

spec = importlib.util.spec_from_file_location('step_bench', Path(__file__).with_name('gpu_benchmark.py'))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
campaign, pilot, checkpoint = bench.campaign, bench.pilot, bench.checkpoint
POINTER = 'LATEST_SPARTA_STEP_CPU_CAMPAIGN'
FILES = ['cpu_campaign.py', 'cpu_campaign_job.sh', 'CPU_CAMPAIGN.md',
         'campaign.py', 'pilot.py', 'gpu_benchmark.py', 'gpu_checkpoint.py']


def matrix():
    rows = []
    for height in campaign.HEIGHTS:
        fluid = 1000 * 200 * (1 - .3 * height / 100)
        ppc = min(20, math.floor((5100000 + .001) / (1.5 * fluid)))
        for seed in campaign.SEEDS[:3 if height == 50 else 2]:
            row = dict(index=len(rows), id=f'cpu_h{height:02d}_s{seed}',
                       phase='cpu_geometry', group='fixed_budget', height_percent=height,
                       seed=seed, nx=1000, ny=200, ppc=ppc, ranks=16,
                       dt_s=campaign.REFERENCE_DT, warmup_steps=80000,
                       sampling_steps=120000, block_steps=10000, sample_every=10)
            row['initial_particle_estimate'] = bench.check_particle_budget(row)
            rows.append(row)
    return rows


def account_jobs(account):
    raw = subprocess.check_output(['squeue', '-r', '-h', '-A', account,
                                   '-o', '%i|%C|%T|%j'], text=True)
    jobs = []
    for line in raw.splitlines():
        jid, cpus, state, name = line.strip().split('|', 3)
        jobs.append(dict(job=jid, cpus=int(cpus), state=state, name=name))
    return jobs


def queue(out, meta, name, ranks, indices=None, dependency=None):
    cmd = ['sbatch', '--parsable', '--account=' + meta['account'], '--partition=cpu',
           '--nodes=1', '--constraint=x86_64', f'--ntasks={ranks}', '--cpus-per-task=1',
           '--mem=16G',
           '--time=1-00:00:00' if name == 'cases' else '--time=01:00:00',
           '--requeue', '--job-name=step-cpu-' + name, '--chdir=' + str(out),
           '--output=' + str(out / (name + '-%A_%a.out')),
           '--error=' + str(out / (name + '-%A_%a.err')),
           '--export=ALL']
    if indices is not None:
        cmd += ['--array=' + ','.join(map(str, indices)) + '%' + str(meta['concurrency'])]
    if dependency:
        cmd += ['--dependency=afterok:' + dependency, '--kill-on-invalid-dep=yes']
    env = dict(os.environ, SPARTA_CPU_OUT=str(out))
    result = subprocess.check_output(cmd + [str(out / 'code/cpu_campaign_job.sh'), name],
                                     env=env, text=True).strip().split(';')[0]
    if not result.isdigit():
        raise ValueError('Unrecognized sbatch response: ' + result)
    meta['jobs'][name] = result
    pilot.write_json(out / 'manifest.json', meta)
    print(f'{name.upper()}_JOB={result}', flush=True)
    return result


def submit(base, source, ref):
    if not re.fullmatch('[0-9a-f]{40}', ref):
        raise ValueError('Full immutable commit required')
    base, source = Path(base).resolve(), Path(source).resolve()
    with open(base / 'cpu-campaign-submit.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / POINTER).exists():
            raise ValueError('CPU campaign already exists; use status/resume, never duplicate')
        rows = matrix()
        jobs = account_jobs('pi_roohie_umass_edu')
        # Conservatively reserve even dependent cylinder jobs separately.
        reserved = sum(j['cpus'] for j in jobs)
        concurrency = min(len(rows), (1000 - reserved - 3) // 16)
        if concurrency < 1:
            raise ValueError('No CPU headroom after reserving other jobs and three cores')
        if shutil.disk_usage(base).free < 100 * 2**30:
            raise ValueError('Need at least 100 GiB filesystem headroom')
        if not (source / 'binary.sha256').is_file():
            raise ValueError('Verified CPU pilot binary receipt missing')
        subprocess.run(['sha256sum', '-c', str(source / 'binary.sha256')], check=True)
        out = Path(tempfile.mkdtemp(prefix=time.strftime('step-cpu-%Y%m%dT%H%M%SZ-', time.gmtime()), dir=base / 'runs'))
        (out / 'code').mkdir()
        for name in FILES:
            shutil.copy2(Path(__file__).with_name(name), out / 'code' / name)
        (out / 'code.sha256').write_text(''.join(f'{pilot.sha(out / "code" / n)}  code/{n}\n' for n in FILES))
        meta = dict(status='submitting', flowmllab_commit=ref, sparta_commit=pilot.SPARTA_COMMIT,
                    source=str(source), binary=str(source / 'sparta-source/src/spa_mpi'),
                    binary_sha256=pilot.sha(source / 'sparta-source/src/spa_mpi'),
                    account='pi_roohie_umass_edu', cases=rows, jobs={}, concurrency=concurrency,
                    resource_budget=dict(account_cpu_limit=1000, reserved_other_jobs=jobs,
                        reserved_cores=reserved, extra_free_cores=3, step_cores=16 * concurrency),
                    training_data_approved=False, mesh_independence_established=False,
                    policy='All pilot-budget geometries explicitly requested; archived refinement campaign not used.')
        pilot.write_json(out / 'manifest.json', meta)
        (base / POINTER).write_text(str(out) + '\n')
        with open(out / 'run_matrix.csv', 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        try:
            first = queue(out, meta, 'preflight', 16)
            array = queue(out, meta, 'cases', 16, list(range(len(rows))), first)
            queue(out, meta, 'collect', 1, dependency=array)
            meta['status'] = 'submitted'
        except Exception:
            meta['status'] = 'partial_submission'
            raise
        finally:
            pilot.write_json(out / 'manifest.json', meta)
        print(f'OUT={out}\nCASES={len(rows)} CONCURRENT={concurrency} STEP_CORES={16*concurrency} OTHER_RESERVED={reserved}', flush=True)


def run_one(root, row, binary, ranks, serial=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / 'execution.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = checkpoint.completed(root)
        if done:
            print('REUSE_COMPLETE=' + str(root), flush=True)
            return done
        attempts = root / 'attempts'; attempts.mkdir(exist_ok=True)
        n = max([int(p.name) for p in attempts.iterdir() if p.name.isdigit()], default=0) + 1
        path = attempts / f'{n:04d}'
        warm = checkpoint.latest_warm(root)
        active_row = row.copy()
        if warm:
            old = bench.load(warm.parent / 'case.json')
            if old['campaign_case_id'] != row['id'] or old['fresh_seed'] != row['seed']:
                raise ValueError('Checkpoint identity mismatch')
            active_row['warmup_steps'] = row['block_steps']
        campaign.generate_case(path, active_row, warm, 'retry' if warm else None,
                               smoke=row.get('smoke', False))
        m = bench.load(path / 'case.json')
        m.update(checkpoint_resume=bool(warm), attempt=n, training_data_approved=False)
        pilot.write_json(path / 'case.json', m)
        checkpoint.checkpoint_deck(path)
        cmd = bench.command(binary, 'cpu', ranks=ranks, serial=serial)
        timing = bench.execute(path, cmd, timeout=23*3600)
        expected = [active_row['warmup_steps'], row['sampling_steps']-row['block_steps'], row['block_steps']]
        if [v['steps'] for v in timing['loops']] != expected:
            raise ValueError('Solver did not complete expected steps')
        if checkpoint.latest_warm(root) != path / 'restart.warm':
            raise ValueError('Atomic warm checkpoint missing')
        timing.update(job=os.environ.get('SLURM_JOB_ID'), resumed=bool(warm))
        pilot.write_json(path / 'timing.json', timing)
        result = bench.validate_case(path)
        checkpoint.commit_result(root, path, timing, result)
        print(f'CPU_CASE_COMPLETE={row["id"]} WALL_SECONDS={timing["wall_seconds"]:.1f}', flush=True)
        return path, timing, result


def preflight(out, binary, serial=False):
    out = Path(out)
    for height in [16, 50, 75]:
        row = bench.smoke_row(height); row['smoke'] = True
        path, _, _ = run_one(out / 'preflight' / f'h{height}', row, binary, 2, serial)
        # Exercise the same restart code used after an interrupted production job.
        retry = out / 'preflight' / f'retry{height}' / 'attempts' / '0001'
        retry.parent.mkdir(parents=True, exist_ok=True)
        if not retry.exists():
            retry.mkdir()
            for name in ['case.json', 'restart.warm', 'WARM_CHECKPOINT.json']:
                shutil.copy2(path / name, retry / name)
        continued, timing, _ = run_one(retry.parent.parent, row, binary, 4, serial)
        if not timing['resumed']:
            raise ValueError('Restart smoke did not resume')
        print(f'CPU_FRESH_AND_RESTART_PASS h={height}', flush=True)
    (out / 'PREFLIGHT_PASS').write_text('Fresh/restart checks passed. Not scientific validation.\n')


def collect(out):
    out = Path(out); meta = bench.load(out / 'manifest.json'); reports = []
    for row in meta['cases']:
        done = checkpoint.completed(out / 'cases' / row['id'])
        if not done:
            raise ValueError('Incomplete case: ' + row['id'])
        path, timing, report = done
        reports.append(dict(id=row['id'], path=str(path), timing=timing, report=report))
    pilot.write_json(out / 'cpu_campaign_report.json', dict(status='execution_complete',
        cases=reports, training_data_approved=False, mesh_independence_established=False))
    (out / 'CPU_CAMPAIGN_COMPLETE').write_text(f'{len(reports)} cases completed; scientific review still required.\n')
    print(f'CPU_CAMPAIGN_COMPLETE={len(reports)} TRAINING_DATA_APPROVED=False')


def resume(out):
    out = Path(out); meta = bench.load(out / 'manifest.json')
    with open(out / 'resume.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = account_jobs(meta['account'])
        known = set(meta['jobs'].values())
        if any(j['job'].split('_')[0] in known for j in jobs):
            raise ValueError('Campaign has active or pending jobs; refusing duplicate')
        missing = [r['index'] for r in meta['cases'] if not checkpoint.completed(out / 'cases' / r['id'])]
        if not missing:
            collect(out); return
        reserved = sum(j['cpus'] for j in jobs)
        meta['concurrency'] = min(len(missing), (1000 - reserved - 3) // 16)
        if meta['concurrency'] < 1:
            raise ValueError('No CPU headroom')
        meta.setdefault('job_history', []).append(meta['jobs'].copy()); meta['jobs'] = {}
        meta['status'] = 'resubmitting'; pilot.write_json(out / 'manifest.json', meta)
        first = None if (out / 'PREFLIGHT_PASS').exists() else queue(out, meta, 'preflight', 16)
        array = queue(out, meta, 'cases', 16, missing, first)
        queue(out, meta, 'collect', 1, dependency=array)
        meta['status'] = 'submitted'; pilot.write_json(out / 'manifest.json', meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['submit', 'plan', 'preflight', 'run', 'collect', 'resume', 'status'])
    p.add_argument('--out'); p.add_argument('--base', default=pilot.DEFAULT_BASE)
    p.add_argument('--source'); p.add_argument('--ref'); p.add_argument('--binary')
    p.add_argument('--serial', action='store_true'); p.add_argument('--index', type=int)
    a = p.parse_args()
    if a.action == 'plan': print(json.dumps(matrix(), indent=2)); return
    if a.action == 'submit': submit(a.base, a.source, a.ref); return
    out = Path(a.out)
    if a.action == 'preflight': preflight(out, a.binary, a.serial); return
    if a.action == 'collect': collect(out); return
    if a.action == 'resume': resume(out); return
    meta = bench.load(out / 'manifest.json')
    if a.action == 'status':
        subprocess.run(['sacct', '-j', ','.join(meta['jobs'].values()), '-X', '-o', 'JobID%24,State%20,Elapsed,ExitCode'])
        print('OUT=' + str(out)); print('JOBS=' + json.dumps(meta['jobs']))
        print('COMPLETED_CASES=' + str(len(list((out / 'cases').glob('*/COMPLETE.json')))) + '/' + str(len(meta['cases'])))
        return
    if not (out / 'PREFLIGHT_PASS').exists(): raise ValueError('Missing preflight pass')
    row = meta['cases'][a.index]
    bench.check_particle_budget(row)
    if int(os.environ.get('SLURM_NTASKS', 0)) != row['ranks']: raise ValueError('Wrong allocated rank count')
    run_one(out / 'cases' / row['id'], row, a.binary, row['ranks'])


if __name__ == '__main__':
    main()
