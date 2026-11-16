#!/usr/bin/env python3
"""Run one arm of the controlled D/W=5 optimizer comparison; never submits jobs."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('d5_training', ROOT/'flowmllab/deep_cavity_training.py')
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)
STOP = False


def stop_requested(_signum, _frame):
    global STOP
    STOP = True


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def dependency_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_entry(device, phase):
    return {'utc': utcnow(), 'hostname': socket.gethostname(), 'phase_on_entry': phase,
            'slurm_job_id': os.getenv('SLURM_JOB_ID'),
            'slurm_restart_count': os.getenv('SLURM_RESTART_COUNT', '0'),
            'python': platform.python_version(), 'torch': torch.__version__,
            'torch_cuda': torch.version.cuda, 'numpy': np.__version__,
            'pytorch_optimizer': dependency_version('pytorch-optimizer'),
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'gpu_capability': list(torch.cuda.get_device_capability(device)) if str(device).startswith('cuda') else None,
            'threads': torch.get_num_threads()}


def write_history(history, path):
    if not history:
        return
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['adam_lbfgs', 'soap_ssb'], required=True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--points', type=int, default=8192)
    parser.add_argument('--warm-evaluations', type=int, default=50)
    parser.add_argument('--qn-evaluations', type=int, default=10)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--audit-points', type=int, default=1024)
    parser.add_argument('--grid-nx', type=int, default=121)
    parser.add_argument('--grid-nh', type=int, default=605)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--max-seconds', type=float, default=0.)
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Pause at an accepted boundary without altering the frozen total budget')
    parser.add_argument('--purpose', choices=['cost_smoke', 'controlled_pilot'], default='cost_smoke')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if min(args.points, args.warm_evaluations, args.qn_evaluations, args.chunk_size,
           args.audit_points, args.checkpoint_every, args.threads) < 1:
        parser.error('Counts must be positive')
    torch.set_num_threads(args.threads)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; refusing silent CPU fallback')
    device = torch.device(args.device)
    source = d5.load_author_source(args.source_dir)
    source_hashes = {name: d5.sha256(args.source_dir/name) for name in d5.AUTHOR_SHA256}
    config = {'schema': d5.SCHEMA, 'purpose': args.purpose, 'method': args.method,
              'seed': args.seed, 'Re': 1000., 'W': 1., 'D': 5., 'width': 64, 'depth': 6,
              'points': args.points, 'chunk_size': args.chunk_size,
              'warm_evaluations': args.warm_evaluations, 'qn_evaluations': args.qn_evaluations,
              'audit_points_per_region': args.audit_points, 'audit_seed': 20264,
              'grid_nx': args.grid_nx, 'grid_nh': args.grid_nh,
              'dtype': 'float64', 'adaptive_sampling': False,
              'sampling': 'authored_balanced_mixture_frozen_both_phases',
              'sample_seed': args.seed*1000003+500009,
              'initialization': 'fresh_Xavier_normal_zero_bias_no_transfer',
              'source_sha256': source_hashes,
              'adapter_sha256': d5.sha256(ROOT/'flowmllab/deep_cavity_training.py'),
              'runner_sha256': d5.sha256(__file__),
              'torch': torch.__version__, 'numpy': np.__version__,
              'SOAP_version_new_runtime': d5.SOAP_VERSION,
              'historical_SOAP_version': 'not_recovered',
              'first_order': {'lr': .002, 'betas': [.99,.999], 'weight_decay': 0.,
                              'SOAP_precondition_frequency': 1},
              'quasi_Newton': {'line_search': 'strong_wolfe', 'c1': 1e-4, 'c2': .9,
                               'LBFGS_history_size': 100, 'SSB_capacity': args.qn_evaluations,
                               'SSB_generators_dropped_allowed': 0}}
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_stream = (out/'writer.lock').open('a+')
    if os.name == 'posix':
        import fcntl
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    marker = out/'contract.json'
    checkpoint = out/'checkpoint.pt'
    if marker.exists():
        if not args.resume:
            raise ValueError('Existing run requires --resume; no duplicate trajectory')
        if json.loads(marker.read_text()) != config:
            raise ValueError('Resume contract changed: source, objective, optimizer, or budget')
        if not checkpoint.exists():
            raise ValueError('No complete recovery available; refusing optimizer reset')
    elif args.resume:
        raise ValueError('Cannot resume without a matching contract and complete recovery')
    elif any(p.name != 'writer.lock' for p in out.iterdir()):
        raise ValueError('Refusing to overwrite a foreign or partial run')
    else:
        d5.atomic_json(config, marker)
    net, geom = d5.make_problem(source, args.seed, device)
    history, runtime, completed = [], [], False
    counters = {'function_evaluations':0, 'gradient_evaluations':0,
                'warm_evaluations':0, 'qn_evaluations':0,
                'warm_steps':0, 'qn_steps':0, 'aborted_budget_steps':0}
    phase, active_seconds, training_seconds = 'warm', 0., 0.
    points = d5.frozen_points(source, args.points, config['sample_seed'])
    audits = d5.audit_points(config['audit_seed'], args.audit_points)
    initial_model = d5.cpu_copy(net.state_dict())
    initial_hash = d5.model_digest(net)
    initial_audit = None
    freeze = None
    if args.resume:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck.get('schema') != d5.SCHEMA or ck['contract'] != config:
            raise ValueError('Checkpoint contract mismatch')
        if d5.state_digest(ck['model']) != ck['model_sha256']:
            raise ValueError('Checkpoint model digest mismatch')
        if d5.state_digest(ck['initial_model']) != ck['initial_model_sha256']:
            raise ValueError('Initial state digest mismatch')
        net.load_state_dict(ck['model'])
        points, audits = ck['points'], ck['audit_points']
        if d5.array_digest(points.numpy()) != ck['sample_sha256']:
            raise ValueError('Frozen sample corruption')
        if {name:d5.array_digest(value.numpy()) for name,value in audits.items()} != ck['audit_sample_sha256']:
            raise ValueError('Independent audit samples changed')
        counters, history, runtime = ck['counters'], ck['history'], ck['runtime']
        phase, completed = ck['phase'], ck['completed']
        active_seconds, training_seconds = ck['active_seconds'], ck['training_seconds']
        initial_model, initial_hash = ck['initial_model'], ck['initial_model_sha256']
        initial_audit = ck['initial_audit']
        freeze = ck.get('freeze')
        if completed:
            if not freeze or freeze['model_sha256'] != d5.model_digest(net):
                raise ValueError('Completed checkpoint lacks valid immutable freeze metadata')
            previous = out/'manifest.json'
            if previous.exists():
                frozen = json.loads(previous.read_text())
                if frozen.get('status') == 'frozen':
                    checks = [('checkpoint_sha256', checkpoint),
                              ('prediction_sha256', out/'prediction.npz'),
                              ('final_audit_sha256', out/'final_audit.json'),
                              ('history_sha256', out/'history.csv')]
                    if frozen['configuration'] != config or frozen['frozen_at_utc'] != freeze['utc'] or frozen['frozen_model_sha256'] != freeze['model_sha256']:
                        raise ValueError('Frozen bundle metadata mismatch')
                    if any(d5.sha256(path) != frozen[key] for key,path in checks):
                        raise ValueError('Frozen bundle byte verification failed')
                    print('ALREADY_FROZEN_AND_HASH_VERIFIED', flush=True)
                    return 0
    opt = d5.make_optimizer(source, net, args.method, phase, args.qn_evaluations)
    if args.resume:
        d5.unpack_optimizer(opt, ck['optimizer'])
        d5.restore_rng(ck['rng'])
    if not completed:
        runtime.append(runtime_entry(device, phase))
    points_device = points.to(device)
    started, steps_this_entry = time.perf_counter(), 0
    sample_hash = d5.array_digest(points.numpy())
    def active_clock():
        return active_seconds + time.perf_counter()-started
    def save():
        d5.atomic_torch_save({'schema':d5.SCHEMA, 'contract':config,
            'model':d5.cpu_copy(net.state_dict()), 'model_sha256':d5.model_digest(net), 'optimizer':d5.pack_optimizer(opt),
            'rng':d5.rng_state(), 'points':points, 'audit_points':audits,
            'sample_sha256':sample_hash, 'audit_sample_sha256':{name:d5.array_digest(value.numpy()) for name,value in audits.items()}, 'initial_model':initial_model,
            'initial_model_sha256':initial_hash, 'initial_audit':initial_audit,
            'phase':phase, 'completed':completed, 'freeze':freeze, 'counters':counters,
            'history':history, 'runtime':runtime, 'training_seconds':training_seconds,
            'active_seconds':active_clock(), 'adaptive_state':None}, checkpoint)
        write_history(history, out/'history.csv')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop_requested)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, stop_requested)
    if initial_audit is None:
        initial_audit = {'regions':d5.independent_audit(source,net,geom,audits),
                         'walls':d5.wall_audit(source,net,geom)}
        d5.atomic_json(initial_audit, out/'initial_audit.json')
    if not completed:
        save()
    failure = None
    while not completed:
        budget = args.warm_evaluations if phase=='warm' else args.qn_evaluations
        if counters[phase+'_evaluations'] >= budget:
            if phase=='warm':
                phase = 'qn'
                opt = d5.make_optimizer(source, net, args.method, phase, args.qn_evaluations)
                runtime.append(runtime_entry(device, phase))
                save()
                continue
            completed = True
            break
        if STOP or (args.max_seconds and time.perf_counter()-started >= args.max_seconds) or (args.max_steps and steps_this_entry>=args.max_steps):
            break
        closure = d5.CountedClosure(source,net,points_device,geom,args.chunk_size,opt,
                                    budget,counters,history,phase,counters[phase+'_steps'],active_clock)
        synchronize(device)
        tick = time.perf_counter()
        try:
            d5.controlled_step(closure)
            counters[phase+'_steps'] += 1
            steps_this_entry += 1
        except d5.EvaluationBudgetReached:
            counters['aborted_budget_steps'] += 1
        except FloatingPointError as exc:
            failure = str(exc)
        synchronize(device)
        training_seconds += time.perf_counter()-tick
        if failure or steps_this_entry % args.checkpoint_every == 0:
            save()
        if failure:
            break
    if not completed or freeze is None:
        if completed:
            freeze = {'utc':utcnow(), 'model_sha256':d5.model_digest(net)}
        save()
    hardware = sorted({(r['gpu_name'] or 'CPU', str(r['gpu_capability']),r['torch']) for r in runtime})
    report = {'schema':d5.SCHEMA, 'status':'failed' if failure else 'frozen' if completed else 'paused',
              'failure':failure, 'purpose':args.purpose, 'method':args.method,'seed':args.seed,
              'configuration':config, 'initial_model_sha256':initial_hash,
              'sample_sha256':sample_hash, 'frozen_model_sha256':d5.model_digest(net),
              'counters':counters, 'training_seconds':training_seconds,
              'active_seconds':active_clock(), 'runtime':runtime,
              'timing_definition':{'elapsed_seconds':'active wall time including audits/checkpoint I/O, excludes downtime', 'training_seconds':'synchronized optimizer-step time including closure, line search and transactional state handling'},
              'history_loss_definition':'warm rows are pre-update closure losses; QN rows are evaluated trial losses; accepted marks an evaluated state equal to a returned accepted endpoint',
              'wall_time_comparison_hardware_unchanged':len(hardware)==1,
              'physics':'author_Gaussian_lid_full_envelope_unchanged',
              'boundary_validation':'exact_hard_velocity_constraints',
              'CFD_used_for_training_or_selection':False,
              'convergence_claim':False, 'frozen_at_utc':freeze['utc'] if completed else None}
    if completed:
        # Freeze the final accepted state before generating assessment outputs.
        write_history(history, out/'history.csv')
        report['history_sha256'] = d5.sha256(out/'history.csv')
        report['checkpoint_sha256'] = d5.sha256(checkpoint)
        final_audit = {'regions':d5.independent_audit(source,net,geom,audits),
                       'walls':d5.wall_audit(source,net,geom)}
        d5.atomic_json(final_audit, out/'final_audit.json')
        report['final_audit_sha256'] = d5.sha256(out/'final_audit.json')
        values = d5.prediction(source,net,geom,args.grid_nx,args.grid_nh,min(args.chunk_size,2048))
        np.savez_compressed(out/'prediction.npz', **values)
        report['prediction_sha256'] = d5.sha256(out/'prediction.npz')
        report['prediction_coordinates'] = {'x':'x/W','h':'height_above_floor/W','source_y':'h-4',
                                             'layout':'h,x','pressure':'raw_single_gauge_pending_CFD_alignment',
                                             'omega':'native_automatic_differentiation_vx_minus_uy'}
        report['final_audit'] = final_audit
    d5.atomic_json(report, out/'manifest.json')
    print(json.dumps({k:report[k] for k in ('status','purpose','method','seed','counters','training_seconds','wall_time_comparison_hardware_unchanged')},indent=2),flush=True)
    if failure:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

