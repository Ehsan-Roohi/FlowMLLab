#!/usr/bin/env python3
"""Atomic particle checkpoints and completed-arm receipts; no averaged-tally stitching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.pending')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def seal(case):
    """Called by rank zero only AFTER SPARTA has closed write_restart's file."""
    case = Path(case).resolve()
    pending = case/'restart.warm.pending'
    if not pending.is_file() or pending.stat().st_size == 0:
        raise ValueError('Empty particle checkpoint')
    with open(pending, 'rb') as f:
        os.fsync(f.fileno())
    pending.replace(case/'restart.warm')
    atomic_json(case/'WARM_CHECKPOINT.json', dict(
        restart_sha256=sha(case/'restart.warm'), case_sha256=sha(case/'case.json'),
        step=read(case/'case.json')['warmup_steps'],
        note='Particles saved; RNG and running averages are not restart state.'))
    print('SPARTA_GPU_WARM_CHECKPOINT_COMMITTED', flush=True)


def checkpoint_deck(case):
    case = Path(case).resolve()
    path = case/'in.step'
    text = path.read_text()
    invocation = shlex.join([sys.executable, '-I', str(Path(__file__).resolve()),
                             'seal', '--case', str(case)])
    text = text.replace('write_restart restart.warm\n',
                        'write_restart restart.warm.pending\n' +
                        'shell ' + invocation + '\n')
    path.write_text(text)


def latest_warm(root):
    for receipt in sorted(Path(root).glob('attempts/*/WARM_CHECKPOINT.json'), reverse=True):
        try:
            d = read(receipt); case = receipt.parent
            if (sha(case/'restart.warm') == d['restart_sha256'] and
                    sha(case/'case.json') == d['case_sha256']):
                return case/'restart.warm'
        except (OSError, ValueError, KeyError):
            pass
        print(f'UNCOMMITTED_OR_CORRUPT_CHECKPOINT_SKIPPED={receipt}', flush=True)
    return None


def commit_result(root, case, timing, report):
    """Write receipt last. A killed validator cannot create a completed-arm receipt."""
    root, case = Path(root), Path(case)
    names = ['case.json', 'in.step', 'timing.json', 'report.json', 'grid.final.gz',
             'restart.final', 'solver.stdout', 'log.sparta', 'flux.blocks', 'boundary.blocks']
    names += [p.name for p in case.glob('grid.block.*.gz')]
    names += [p.name for p in case.glob('wall.running.*.gz')]
    atomic_json(root/'COMPLETE.json', dict(attempt=str(case.relative_to(root)),
        hashes={n: sha(case/n) for n in sorted(names)}, timing=timing, report=report))


def completed(root):
    root = Path(root)
    if not (root/'COMPLETE.json').exists():
        return None
    receipt = read(root/'COMPLETE.json')
    case = (root/receipt['attempt']).resolve()
    if not case.is_relative_to(root.resolve()):
        raise ValueError('Completed case path outside run')
    for name, digest in receipt['hashes'].items():
        if sha(case/name) != digest:
            raise ValueError(f'Completed artifact changed: {case/name}')
    return case, receipt['timing'], receipt['report']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['seal']); p.add_argument('--case', required=True)
    seal(p.parse_args().case)
