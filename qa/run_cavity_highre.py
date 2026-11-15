"""Run new high-Re CFD roots; never read Ghia samples during solution.

Example (CPU): python qa/run_cavity_highre.py --cells 128 --seed FIELD.npz
    --output SCRATCH_DIRECTORY
One atomic configuration-matched checkpoint is kept per grid. A failed Newton
trial reduces only the continuation step in Re, not the PDE or acceptance gate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'flowmllab'))
from cavity_steady import SteadyCavity

TARGETS = [3200, 5000, 7500, 10000]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.pending.npz')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(tmp, allow_pickle=False) as check:
        for key in arrays:
            if not np.array_equal(check[key], arrays[key]):
                raise IOError(f'Checkpoint round trip mismatch: {key}')
    os.replace(tmp, path)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, required=True)
    parser.add_argument('--seed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output / f'n{args.cells}'
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                     for name in ['flowmllab/cavity_steady.py', 'qa/run_cavity_highre.py']}
    config = dict(cells=args.cells, method='steady-newton-central-thom', lid=1.0,
                  length=1.0, targets=TARGETS, nonlinear_tolerance=1e-8,
                  acceptance_steady_linf=1e-6, source_sha256_lf=source_hashes,
                  seed_sha256=digest(args.seed), start_reynolds=1000,
                  max_continuation_increment=500, min_continuation_increment=15.625,
                  pressure='not reconstructed; velocity/vorticity validation only')
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    solver = SteadyCavity(args.cells)
    checkpoint = out/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            if meta['signature'] != signature:
                raise ValueError('Checkpoint configuration/source mismatch; refusing reuse')
            state = z['state'].copy()
        completed = meta['completed_reynolds']
        history = meta['history']
        seconds_before = meta['compute_seconds']
    else:
        with np.load(args.seed, allow_pickle=False) as seed:
            state = solver.interpolate_seed(seed)
        completed, history, seconds_before = 0, [], 0.0
    clock_start = time.perf_counter()
    attempt, increment = 0, 500.0
    # Recover the last complete root, never a partly advanced continuation trial.
    def save_root(root, reynolds):
        metadata = dict(signature=signature, config=config, completed_reynolds=reynolds,
                        history=history, compute_seconds=seconds_before+time.perf_counter()-clock_start)
        atomic_npz(checkpoint, state=root, metadata=np.array(json.dumps(metadata, allow_nan=False)))

    def root_metadata(root, reynolds, iterations):
        f = solver.fields(root)
        residual = solver.residual(root, reynolds)
        h = solver.h
        div = (f['u'][1:-1,2:]-f['u'][1:-1,:-2]+f['v'][2:,1:-1]-f['v'][:-2,1:-1])/(2*h)
        j, i = np.unravel_index(np.argmin(f['psi']), f['psi'].shape)
        data_path = out/f're{int(reynolds)}.npz'
        atomic_npz(data_path, **f)
        return dict(config=config, signature=signature, reynolds=int(reynolds),
                    cells=args.cells, converged=bool(np.max(np.abs(residual))<1e-6),
                    poisson_linf=float(np.max(np.abs(residual[:solver.n]))),
                    steady_vorticity_linf=float(np.max(np.abs(residual[solver.n:]))),
                    continuity_linf=float(np.max(np.abs(div))),
                    vortex_x=float(f['x'][i]), vortex_y=float(f['y'][j]), psi_min=float(f['psi'][j,i]),
                    branch='2-D steady Navier-Stokes root; temporal stability not inferred',
                    field_sha256=digest(data_path), newton_history=iterations,
                    continuation_history=history,
                    cumulative_compute_seconds=seconds_before+time.perf_counter()-clock_start,
                    host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    utc=datetime.now(timezone.utc).isoformat(),
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                    ghia_used_during_solution=False)

    while completed < TARGETS[-1]:
        if completed == 0:
            proposed = 1000.0
        else:
            next_target = min(t for t in TARGETS if t > completed)
            proposed = min(completed+increment, next_target)
        attempt += 1
        print(json.dumps(dict(event='start_root', cells=args.cells, reynolds=proposed,
                              previous_reynolds=completed, attempt=attempt)), flush=True)
        local_history = []
        def callback(z, record):
            local_history.append(dict(record))
            print(json.dumps(dict(cells=args.cells, reynolds=proposed, **record)), flush=True)
        started = time.perf_counter()
        try:
            candidate, records = solver.solve(state, proposed, callback=callback)
        except (ArithmeticError, FloatingPointError) as exc:
            history.append(dict(reynolds=proposed, success=False, message=str(exc),
                                seconds=time.perf_counter()-started, last_residual=local_history[-1] if local_history else None))
            atomic_json(out/'status.json', dict(config=config, completed_reynolds=completed,
                        trial_reynolds=proposed, failure=str(exc), history=history))
            if completed == 0 or increment/2 < config['min_continuation_increment']:
                raise
            increment /= 2
            continue
        history.append(dict(reynolds=proposed, success=True, iterations=len(records)-1,
                            seconds=time.perf_counter()-started, final=records[-1]))
        state, completed = candidate, proposed
        if completed in TARGETS:
            meta = root_metadata(state, completed, records)
            atomic_json(out/f're{int(completed)}.json', meta)
        save_root(state, completed)
        atomic_json(out/'status.json', dict(config=config, completed_reynolds=completed,
                    status='complete' if completed == TARGETS[-1] else 'continuing', history=history))
        if len(records) <= 8:
            increment = min(500.0, increment*2)
    print(json.dumps(dict(event='complete', cells=args.cells, targets=TARGETS)), flush=True)


if __name__ == '__main__':
    main()
