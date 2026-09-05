#!/usr/bin/env python3
"""Unity gpu-preempt migration and bounded recovery; only recorded jobs are touched."""
import argparse
import contextlib
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import time

spec = importlib.util.spec_from_file_location('gpu', Path(__file__).with_name('gpu_benchmark.py'))
gpu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpu)
write = gpu.checkpoint.atomic_json


def call(cmd, **kwargs):
    return subprocess.check_output(cmd, text=True, **kwargs).strip()


def job_id(raw):
    value = raw.split(';')[0]
    if not value.isdigit(): raise ValueError(f'Invalid sbatch job ID: {raw!r}')
    return value


def job_info(job):
    raw = call(['scontrol', 'show', 'job', '-o', str(job)])
    return dict(re.findall(r'(\w+)=(\S+)', raw))


@contextlib.contextmanager
def lock(path):
    with open(path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def ensure_guard(out, job):
    out = Path(out); meta = gpu.load(out/'manifest.json')
    if job in meta.get('guards', {}): return meta['guards'][job]
    raw = call(['sbatch', '--parsable', '--account=pi_roohie_umass_edu',
        '--partition=cpu', '--nodes=1', '--ntasks=1', '--cpus-per-task=1',
        '--mem=512M', '--time=00:05:00', '--job-name=step-gpu-resume',
        '--no-requeue', '--export=ALL', f'--dependency=afterany:{job}',
        f'--chdir={out}', f'--output={out}/guard-%j.out', f'--error={out}/guard-%j.err',
        str(out/'code/gpu_guard.sh')],
        env=dict(os.environ, SPARTA_GPU_OUT=str(out), SPARTA_WATCH_JOB=job))
    guard = job_id(raw)
    meta.setdefault('guards', {})[job] = guard
    write(out/'manifest.json', meta)
    print(f'RECOVERY_GUARD_JOB={guard} WATCHES_GPU_JOB={job}', flush=True)
    return guard


def launch_again(out):
    out = Path(out); meta = gpu.load(out/'manifest.json')
    if len(meta['allocations']) >= meta['max_allocations']:
        raise RuntimeError('PREEMPT_RETRY_LIMIT_REACHED; checkpoints retained')
    raw = call(['sbatch', '--parsable', '--hold', '--account=pi_roohie_umass_edu',
        '--partition=gpu-preempt', '--nodes=1', '--ntasks=16', '--cpus-per-task=1',
        '--gpus=1', f"--constraint={meta['gpu_type']}", '--mem=48G', '--time=04:00:00',
        '--job-name=step-gpu-bench', '--export=ALL', '--requeue', '--open-mode=append',
        f'--chdir={out}', f'--output={out}/slurm-%j.out', f'--error={out}/slurm-%j.err',
        str(out/'code/gpu_job.sh')], env=dict(os.environ, SPARTA_GPU_OUT=str(out)))
    job = job_id(raw)
    meta['allocations'].append(job); meta['jobs']['benchmark'] = job
    meta['status'] = 'submitted_held'; write(out/'manifest.json', meta)
    (out/'JOB_ID').write_text(job+'\n')
    ensure_guard(out, job)
    call(['scontrol', 'release', job])
    meta = gpu.load(out/'manifest.json'); meta['status'] = 'submitted'
    write(out/'manifest.json', meta)
    print(f'CHECKPOINT_RESUME_SUBMITTED JOB={job} OUT={out}', flush=True)


def guard(out, job):
    out = Path(out)
    with lock(out/'submit.lock'):
        meta = gpu.load(out/'manifest.json')
        if job not in meta['allocations']: raise ValueError('Unrecorded job')
        if meta['jobs']['benchmark'] != job:
            print('NEWER_GPU_JOB_ALREADY_RECORDED'); return
        if (out/'GPU_BENCHMARK_COMPLETE').exists():
            print('BENCHMARK_COMPLETE_NO_RETRY'); return
        state = ''
        # Accounting can lag the afterany dependency. Bounded to one minute.
        for attempt in range(7):
            raw = call(['sacct', '-X', '-n', '-P', '-j', job, '--format=JobIDRaw,State'])
            rows = [line.split('|') for line in raw.splitlines()]
            states = [r[1].split()[0].rstrip('+') for r in rows if len(r)>1 and r[0]==job and r[1]]
            state = states[-1] if states else ''
            if state and state not in ['RUNNING','PENDING','REQUEUED','COMPLETING']: break
            if attempt < 6: time.sleep(10)
        print(f'FINISHED_GPU_JOB={job} STATE={state}', flush=True)
        # User cancellations, solver errors, OOM and invalid physics are not retries.
        if state in ['PREEMPTED','NODE_FAIL','TIMEOUT']:
            launch_again(out)
        else:
            print('NO_AUTOMATIC_RETRY; status/logs and checkpoints retained', flush=True)


def verify_old(base, old):
    previous = Path((base/gpu.POINTER).read_text().strip()).resolve()
    if not previous.is_relative_to((base/'runs').resolve()):
        raise ValueError('Unexpected old run directory')
    meta = gpu.load(previous/'manifest.json')
    if ((previous/'JOB_ID').read_text().strip()!=old or meta['jobs'].get('benchmark')!=old):
        raise ValueError('Old job does not match saved benchmark; no jobs changed')
    info = job_info(old)
    if (info.get('JobName')!='step-gpu-bench' or
            info.get('WorkDir')!=str(previous) or
            not info.get('UserId','').endswith(f'({os.getuid()})')):
        raise ValueError('Job owner/name/workdir mismatch; no jobs changed')
    return info


def migrate(root, base, ref, old=None, gpu_type='a40'):
    base = Path(base).resolve(); base.mkdir(parents=True, exist_ok=True)
    with lock(base/'gpu-migration.lock'):
        record = base/f'GPU_PREEMPT_MIGRATION_{old or "new"}.json'
        if record.exists():
            transaction = gpu.load(record); out = Path(transaction['out'])
            if transaction['status']=='complete':
                print(f'ALREADY_MIGRATED OUT={out}'); gpu.status(out); return
            job = transaction['job']
        else:
            if old:
                info = verify_old(base, old)
                if info['JobState']!='PENDING':
                    print(f'OLD_JOB_STATE={info["JobState"]}; existing job retained; no duplicate submitted')
                    gpu.status(Path((base/gpu.POINTER).read_text().strip())); return
                call(['scontrol','hold',old])
                if job_info(old)['JobState']!='PENDING':
                    call(['scontrol','release',old])
                    raise ValueError('Old job started before hold; left running')
            try:
                out, job = gpu.submit(root, base, ref, gpu_type, new_run=bool(old),
                                      partition='gpu-preempt', hold=True)
            except BaseException:
                if old: call(['scontrol','release',old])
                raise
            transaction = dict(status='new_job_held', old_job=old, job=job, out=str(out))
            write(record, transaction)
        # The replacement is held until its recovery job exists. A repeated
        # command resumes this transaction instead of submitting a duplicate.
        ensure_guard(out, job)
        if old and transaction['status']=='new_job_held':
            call(['scancel','--state=PENDING',old])
            transaction['status']='old_cancelled'; write(record, transaction)
        info = job_info(job)
        if info['JobState']=='PENDING' and info.get('Reason') in ['JobHeldUser','JobHeldAdmin']:
            call(['scontrol','release',job])
        transaction['status']='complete'; write(record, transaction)
        print(f'GPU_PREEMPT_SUBMITTED JOB={job} OUT={out}', flush=True)
        print('GPU=A40' if gpu_type=='a40' else f'GPU={gpu_type}')
        print('PARTITION=gpu-preempt INITIAL_PARTICLES_APPROX=5100000 CHECKPOINTS=enabled')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('submit'); q.add_argument('--root',type=Path,default=Path(__file__).parent)
    q.add_argument('--base',type=Path,default=Path(gpu.pilot.DEFAULT_BASE)); q.add_argument('--ref',required=True)
    q.add_argument('--replace-job'); q.add_argument('--gpu',choices=list(gpu.ARCH),default='a40')
    q=sub.add_parser('guard'); q.add_argument('--out',type=Path,required=True); q.add_argument('--job',required=True)
    a=p.parse_args()
    if a.action=='submit': migrate(a.root,a.base,a.ref,a.replace_job,a.gpu)
    else: guard(a.out,a.job)


if __name__=='__main__': main()
