"""Launch one frozen, checkpointed cavity settling case on allocated scratch.

No scheduling, throttling, automatic lock breaking or duplicate trajectories.
A fresh control has a brief exact rest-start seed; a baseline reuses its
byte-verified native t=10 seed. All later steps use the continuation driver.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--index', type=int, default=None)
    parser.add_argument('--target-time', type=float)
    parser.add_argument('--stop-on-candidate', action='store_true')
    args = parser.parse_args()
    root = args.campaign.resolve().parent
    document = json.loads(args.campaign.read_text(encoding='utf-8'))
    if str(root) != document['campaign_directory'] or not root.is_relative_to(Path(document['workspace'])):
        raise ValueError('Campaign must remain inside its discovered allocated Workspace')
    index = args.index if args.index is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    case = document['cases'][index]
    if case['index'] != index or Path(case['id']).name != case['id']:
        raise ValueError('Case index/path mismatch')
    for relative, expected in document['source_sha256'].items():
        path = (root / 'source' / relative).resolve()
        if not path.is_relative_to(root / 'source') or digest(path) != expected:
            raise ValueError(f'Frozen source hash mismatch: {relative}')
    policy = root / 'policies' / f"re{case['reynolds']}.json"
    if digest(policy) != case['policy_file_sha256']:
        raise ValueError('Frozen stationarity policy bytes changed')
    work = root / 'runs' / case['id']
    work.mkdir(parents=True, exist_ok=True)
    lock = work / '.launch.lock'
    token = uuid.uuid4().hex
    payload = {'token': token, 'hostname': socket.gethostname(), 'pid': os.getpid(),
               'job_id': os.environ.get('SLURM_JOB_ID'), 'array_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
               'array_index': index}
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    active = {'process': None, 'signal': None}
    def stop(signum, _frame):
        active['signal'] = signum
        child = active['process']
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    def execute(command):
        print(json.dumps({'case': case['id'], 'command': command}), flush=True)
        if active['signal'] is not None:
            return 75
        child = subprocess.Popen(command, cwd=root / 'source')
        active['process'] = child
        try:
            return child.wait()
        finally:
            active['process'] = None
    try:
        if case['seed_mode'] == 'existing_native_t10':
            seed = Path(case['seed_path'])
            receipt = Path(case['seed_receipt'])
            for path, sha in ((seed, case['seed_sha256']), (receipt, case['seed_receipt_sha256'])):
                if not path.is_relative_to(Path(document['workspace'])) or digest(path) != sha:
                    raise ValueError('Existing native seed failed Workspace/hash admission')
        elif case['seed_mode'] == 'fresh_rest_start':
            seed_dir = work / 'seed'
            seed = seed_dir / f"re{case['reynolds']}.npz"
            receipt = seed.with_suffix('.json')
            if seed.exists() != receipt.exists():
                raise RuntimeError('Partial seed preserved; inspect before recovery, never overwrite automatically')
            if not seed.exists():
                command = [sys.executable, str(root / 'source/qa/build_cavity_startup.py'), 'solve',
                           '--reynolds', str(case['reynolds']), '--cells', str(case['cells']),
                           '--dt', str(case['dt']), '--end-time', '0.1', '--interval', '0.1',
                           '--method', 'pimple', '--convection', 'central',
                           '--outer-correctors', str(case['outer']), '--pressure-correctors', str(case['pressure']),
                           '--output', str(seed_dir), '--progress']
                code = execute(command)
                if code:
                    return code
            recorded = json.loads(receipt.read_text(encoding='utf-8'))
            config = recorded['config']
            expected = {'cells': case['cells'], 'dt': case['dt'], 'reynolds': case['reynolds'],
                        'outer_correctors': case['outer'], 'pressure_correctors': case['pressure'],
                        'method': 'pimple', 'convection': 'central'}
            if any(config[k] != v for k, v in expected.items()) or recorded['end_time'] != 0.1:
                raise ValueError('Fresh control seed configuration/time mismatch')
        else:
            raise ValueError('Unrecognized seed mode')
        command = [sys.executable, str(root / 'source/qa/continue_cavity_startup.py'), 'run',
                   '--seed', str(seed), '--seed-receipt', str(receipt), '--policy', str(policy),
                   '--output', str(work / 'continuation'),
                   '--target-time', str(args.target_time if args.target_time is not None else case['first_target']), '--progress']
        if args.stop_on_candidate:
            command.append('--stop-on-candidate')
        return execute(command)
    finally:
        if lock.exists() and json.loads(lock.read_text())['token'] == token:
            lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
