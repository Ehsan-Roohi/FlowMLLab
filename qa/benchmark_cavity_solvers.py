"""One isolated matched cavity-solver benchmark trial (CPU/Linux).

Never changes cavity_steady.py, reference fields, trained models, or tolerances.
Timings include Jacobian and preconditioner construction. Timed cold starts are
not claimed: all scenarios declare the existing field used as their initial state.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
import platform
try:
    import resource
except ImportError:
    resource = None  # Small admission tests also run on Windows.
import socket
import sys
from time import perf_counter, process_time

import numpy as np
import scipy
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'flowmllab'))
import cavity_steady as base
from cavity_krylov import LinearSolver, KrylovConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def atomic_npz(path, **values):
    tmp = Path(path).with_suffix('.pending.npz')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
        f.flush(); os.fsync(f.fileno())
    with np.load(tmp, allow_pickle=False) as z:
        for k, v in values.items():
            if not np.array_equal(z[k], v):
                raise IOError('Checkpoint read-back mismatch: '+k)
    os.replace(tmp, path)


class TimedCavity(base.SteadyCavity):
    def __init__(self, cells):
        self.jacobian_seconds = 0.
        self.residual_seconds = 0.
        self.jacobian_calls = 0
        self.residual_calls = 0
        super().__init__(cells)

    def jacobian(self, *args):
        t = perf_counter(); a = super().jacobian(*args)
        self.jacobian_seconds += perf_counter()-t; self.jacobian_calls += 1
        return a

    def residual(self, *args):
        t = perf_counter(); a = super().residual(*args)
        self.residual_seconds += perf_counter()-t; self.residual_calls += 1
        return a


@contextmanager
def injected_linear_solver(linear):
    original = base.spsolve
    base.spsolve = linear
    try:
        yield
    finally:
        base.spsolve = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, required=True)
    parser.add_argument('--method', choices=['direct','gmres_mg','gmres_ilu'], required=True)
    parser.add_argument('--scenario', choices=['warm7500','warm10000','continuation'], required=True)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--linear-cycles', type=int, default=12)
    parser.add_argument('--fill-factor', type=float, default=4.)
    parser.add_argument('--drop-tol', type=float, default=1e-3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output/'result.json'
    if result_file.exists():
        raise FileExistsError('Completed/failed trial already recorded; refusing overwrite')
    cfg = KrylovConfig(gmres_cycles=args.linear_cycles, ilu_fill_factor=args.fill_factor,
                        ilu_drop_tol=args.drop_tol)
    if args.scenario == 'continuation':
        start_re, end_re = 1000, 10000
        seed_path = args.data_root/'n128/re1000.npz'
        schedule = [1000.]
        done = 1000.
        while done < 10000:
            target = next(t for t in [3200,5000,7500,10000] if t > done)
            done = min(done+500, target); schedule.append(float(done))
    else:
        start_re, end_re = (6500,7500) if args.scenario == 'warm7500' else (9000,10000)
        seed_path = args.data_root/f'n{args.cells}/re{start_re}.npz'
        schedule = [float(end_re)]
    reference_path = args.data_root/f'n{args.cells}/re{end_re}.npz'
    record = dict(method=args.method, cells=args.cells, nodes=args.cells+1,
        scenario=args.scenario, repeat=args.repeat, start_re=start_re, target_re=end_re,
        schedule=schedule, nonlinear_tolerance=1e-8, original_base_linear_gate=1e-5,
        candidate_config=asdict(cfg), seed_path=str(seed_path), seed_sha256=digest(seed_path),
        reference_path=str(reference_path), reference_sha256=digest(reference_path),
        source_sha256_lf={s:hashlib.sha256((ROOT/s).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
          for s in ['flowmllab/cavity_steady.py','flowmllab/cavity_krylov.py','qa/benchmark_cavity_solvers.py']},
        host=socket.gethostname(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},
        timing_scope='Same-seed steady CFD solve only; includes Jacobian and preconditioner setup; excludes queue, imports, input I/O, field verification, checkpoint I/O and original seed creation.',
        success=False, history=[])
    try:
        record['cpu_model'] = next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
    except (FileNotFoundError, StopIteration):
        record['cpu_model'] = platform.processor()
    with np.load(seed_path, allow_pickle=False) as z:
        seed = {k:z[k].copy() for k in ['x','y','psi','omega','u','v']}
    t = perf_counter(); solver = TimedCavity(args.cells)
    record['solver_construction_seconds'] = perf_counter()-t
    state = solver.interpolate_seed(seed)
    record['packed_initial_state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
    linear = LinearSolver(args.method, args.cells, cfg)
    try:
        with injected_linear_solver(linear):
            for re in schedule:
                start = perf_counter(); cpu_start = process_time()
                try:
                    state, history = solver.solve(state, re, tol=1e-8, max_newton=35)
                except Exception:
                    record['failed_root_wall_seconds'] = perf_counter()-start
                    record['failed_root_cpu_seconds'] = process_time()-cpu_start
                    raise
                wall, cpu = perf_counter()-start, process_time()-cpu_start
                record['history'].append(dict(reynolds=re,wall_seconds=wall,cpu_seconds=cpu,
                    newton_iterations=len(history)-1,iterations=history))
                record['linear_solves'] = linear.records
                atomic_npz(args.output/'recovery.npz', state=state,
                    metadata=np.array(json.dumps(record,allow_nan=False)))
                print(json.dumps(dict(event='root',method=args.method,cells=args.cells,
                    re=re,wall_seconds=wall,iterations=len(history)-1)),flush=True)
        record['solve_wall_seconds'] = sum(v['wall_seconds'] for v in record['history'])
        record['solve_cpu_seconds'] = sum(v['cpu_seconds'] for v in record['history'])
        residual = solver.residual(state,end_re)
        fields = solver.fields(state)
        with np.load(reference_path, allow_pickle=False) as ref:
            metrics={}
            for field in ['psi','omega','u','v']:
                difference = fields[field]-ref[field]
                metrics[field] = dict(relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(ref[field]),1e-30)),
                    linf=float(abs(difference).max()))
            vel_difference = np.r_[(fields['u']-ref['u']).ravel(),(fields['v']-ref['v']).ravel()]
            vel_reference = np.r_[ref['u'].ravel(),ref['v'].ravel()]
            metrics['velocity_relative_l2'] = float(np.linalg.norm(vel_difference)/np.linalg.norm(vel_reference))
        record['reference_discrepancies'] = metrics
        record['final_residual_linf'] = float(abs(residual).max())
        record['solution_gate'] = bool(record['final_residual_linf']<1e-8 and
            metrics['velocity_relative_l2']<1e-8 and metrics['omega']['relative_l2']<1e-7)
        record['success'] = record['solution_gate']
        # One representative physical field; repeat metrics remain compact.
        if args.repeat == 0:
            atomic_npz(args.output/'final_fields.npz', **fields)
            record['final_fields_sha256'] = digest(args.output/'final_fields.npz')
        record['final_state_sha256'] = hashlib.sha256(state.tobytes()).hexdigest()
        if not record['solution_gate']:
            record['error'] = 'PDE or retained-reference field agreement gate failed'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['linear_solves'] = linear.records
    record['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    record['profiling'] = dict(jacobian_seconds=solver.jacobian_seconds,
        jacobian_calls=solver.jacobian_calls,residual_seconds=solver.residual_seconds,
        residual_calls=solver.residual_calls,linear_total_seconds=sum(v['total_seconds'] for v in linear.records),
        preconditioner_setup_seconds=sum(v.get('setup_seconds') or 0 for v in linear.records),
        krylov_iterations=sum(v.get('inner_iterations',0) for v in linear.records))
    atomic_json(result_file,record)
    print(json.dumps({k:record[k] for k in ['method','cells','scenario','repeat','success','peak_rss_KiB']}
        | {k:record[k] for k in ['solve_wall_seconds','error','final_residual_linf'] if k in record}),flush=True)
    return 0 if record['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
