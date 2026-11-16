#!/usr/bin/env python3
"""Run one predeclared area-stratified PINN against its frozen baseline work cap.

No CFD/Ghia values are read. This script only reads verified baseline training
metadata. All 12 fresh predictions must freeze before independent scoring.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flowmllab'))
import numpy as np
import torch
from cavity_pinn_stratified import (Config, Trainer, BASE_SOURCE_SHA256, numerical_admission,
    final_audit, predict, source_hash, tensor_hash, file_sha256, state_hash, read_baseline_contract)

STOP_REQUESTED = False


def stop_requested(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def runner_hash():
    import hashlib
    return hashlib.sha256(Path(__file__).read_text(encoding='utf-8-sig').encode()).hexdigest()


def acquire_lock(path):
    stream = path.open('a+b')
    if os.name == 'nt':
        import msvcrt
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            stream.close(); raise RuntimeError('Another process owns this trajectory')
    else:
        import fcntl
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close(); raise RuntimeError('Another process owns this trajectory')
    return stream


def verify_frozen(trainer, output):
    manifest = json.loads((output / 'frozen_manifest.json').read_text(encoding='utf-8'))
    if manifest['config'] != asdict(trainer.config) or manifest['frozen_model'] != trainer.frozen:
        raise ValueError('Frozen configuration/model differs')
    if (manifest['source_sha256'] != source_hash() or manifest['runner_sha256'] != runner_hash()
            or manifest['source_raw_sha256'] != file_sha256(Path(__file__).resolve().parents[1]/'flowmllab/cavity_pinn_stratified.py')
            or manifest['runner_raw_sha256'] != file_sha256(__file__)):
        raise ValueError('Frozen implementation differs')
    if tensor_hash(trainer.model.state_dict()) != trainer.frozen['model_sha256']:
        raise ValueError('Model changed after freeze')
    for name, digest in manifest['artifact_sha256'].items():
        if Path(name).name != name or file_sha256(output / name) != digest:
            raise ValueError(f'Frozen artifact hash failed: {name}')
    if manifest['checkpoint_sha256'] != manifest['artifact_sha256']['checkpoint.pt']:
        raise ValueError('Frozen checkpoint binding differs')
    return manifest


def export_frozen(trainer, output):
    if not trainer.finished or trainer.frozen is None:
        raise RuntimeError('A complete, frozen endpoint is required')
    if (output / 'frozen_manifest.json').exists():
        return verify_frozen(trainer, output)
    if trainer.endpoint_audit is None:
        raise RuntimeError('Record the accepted-endpoint audit before saving the completed checkpoint')
    # Rebuild only an interrupted export, from the preserved frozen checkpoint.
    write_json(output / 'config.json', asdict(trainer.config))
    write_json(output / 'budget_contract.json', trainer.baseline_contract)
    write_json(output / 'history.json', trainer.history)
    write_json(output / 'closure_history.json', trainer.evaluation_history)
    write_json(output / 'sampling_coverage.json', trainer.coverage())
    n = trainer.config.export_grid
    axis = np.linspace(0, 1, n); xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    singular = (xy[:, 1] == 1) & ((xy[:, 0] == 0) | (xy[:, 0] == 1))
    values = predict(trainer.model, xy[~singular], str(trainer.device), trainer.config.chunk_size)
    arrays = dict(x=axis, y=axis, reynolds=np.array(trainer.config.reynolds),
                  singular_corner_mask=singular.reshape(n, n))
    for key, value in values.items():
        full = np.full(n * n, np.nan); full[~singular] = value; arrays[key] = full.reshape(n, n)
    profile = np.linspace(0, 1, 513)
    vertical = predict(trainer.model, np.column_stack((np.full_like(profile, .5), profile)), str(trainer.device))
    horizontal = predict(trainer.model, np.column_stack((profile, np.full_like(profile, .5))), str(trainer.device))
    arrays.update(centerline_y=profile, centerline_u=vertical['u'], centerline_x=profile, centerline_v=horizontal['v'])
    destination = output / 'frozen_predictions.npz'
    temporary = destination.with_name(destination.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, destination)
    audit = final_audit(trainer.model, trainer.audit_points, str(trainer.device), trainer.config.chunk_size)
    audit.update(weighted_accepted_endpoint=trainer.endpoint_audit, stratified_coverage=trainer.coverage(),
                 training_work_excludes_audits=True)
    write_json(output / 'final_physics_audit.json', audit)
    artifacts = ('checkpoint.pt', 'config.json', 'budget_contract.json', 'history.json',
                 'closure_history.json', 'sampling_coverage.json', 'frozen_predictions.npz', 'final_physics_audit.json')
    hashes = {name: file_sha256(output / name) for name in artifacts}
    manifest = dict(schema=1, experiment='constant-lid area-stratified matched-work PINN',
        config=asdict(trainer.config), frozen_model=trainer.frozen,
        source_sha256=source_hash(), baseline_source_sha256=BASE_SOURCE_SHA256, runner_sha256=runner_hash(),
        source_raw_sha256=file_sha256(Path(__file__).resolve().parents[1]/'flowmllab/cavity_pinn_stratified.py'),
        runner_raw_sha256=file_sha256(__file__),
        checkpoint_sha256=hashes['checkpoint.pt'], artifact_sha256=hashes,
        prediction_file=destination.name, prediction_sha256=hashes[destination.name],
        physics_audit_file='final_physics_audit.json', physics_audit_sha256=hashes['final_physics_audit.json'],
        frozen_utc=trainer.frozen['frozen_utc'], prediction_grid_n=n,
        reference_labels_used=False, reference_evaluation_performed=False,
        baseline_contract_sha256=state_hash(trainer.baseline_contract),
        accepted_endpoint_audit=trainer.endpoint_audit, closure_budget=trainer.config.closure_budget,
        closure_calls=trainer.closure_calls, residual_point_evaluations=trainer.residual_point_evaluations,
        array_layout='(y,x); exact top corners are NaN and explicitly masked', pressure_gauge='p(0.5,0.5)=0',
        limits='one sampling-only intervention; finite matched work; no accuracy or convergence acceptance implied')
    write_json(output / 'frozen_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-metadata', type=Path,
                        help='Verified archived config/receipt/freeze metadata and baseline_provenance.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--admit-only', action='store_true')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--max-wall-seconds', type=float, default=0)
    parser.add_argument('--checkpoint-seconds', type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    if args.max_steps < 0 or args.max_wall_seconds < 0 or args.checkpoint_seconds <= 0:
        parser.error('Invalid process/checkpoint bound')
    if args.admit_only:
        admission = numerical_admission(args.device); print(json.dumps(admission), flush=True)
        if not admission['passed']: raise RuntimeError('Original physical admission failed')
        return
    if args.output is None: parser.error('--output is required')
    if args.baseline_metadata is None and not args.resume: parser.error('--baseline-metadata is required')
    contract = read_baseline_contract(args.baseline_metadata) if args.baseline_metadata else None
    if contract is not None:
        contract['runner_source_sha256'] = runner_hash()
        contract['runner_raw_sha256'] = file_sha256(__file__)
    config = Config(**contract['baseline_config'], closure_budget=contract['baseline_closure_calls']) if contract else None
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(args.output / 'run.lock')
    try:
        checkpoint = args.output / 'checkpoint.pt'
        if args.resume:
            trainer = Trainer.load(checkpoint, args.device, config=config)
            if contract is not None and contract != trainer.baseline_contract:
                raise ValueError('Paired baseline evidence changed')
            if (trainer.baseline_contract.get('runner_source_sha256') != runner_hash()
                    or trainer.baseline_contract.get('runner_raw_sha256') != file_sha256(__file__)):
                raise ValueError('Restart runner differs from the declared implementation')
            if trainer.finished and (args.output / 'frozen_manifest.json').exists():
                manifest = verify_frozen(trainer, args.output)
                print(json.dumps(dict(status='verified_frozen_noop', manifest=manifest)), flush=True)
                return
        else:
            if checkpoint.exists() or (args.output / 'frozen_manifest.json').exists():
                raise RuntimeError('Existing trajectory: use identical --resume')
            trainer = Trainer(config, args.device, contract)
        admission = numerical_admission(args.device)
        if not admission['passed']: raise RuntimeError('Original physical admission failed')
        # Admission uses independent temporary tensors but changes global RNG. Preserve
        # the training RNG exactly across both initial entry and resumed admission.
        # Reloading the first recovery below restores the complete saved RNG.
        if not checkpoint.exists():
            trainer = Trainer(config, args.device, contract)
            trainer.save(checkpoint)
        else:
            trainer = Trainer.load(checkpoint, args.device, config=trainer.config)
        write_json(args.output / 'admission.json', admission)
        write_json(args.output / 'config.json', asdict(trainer.config))
        write_json(args.output / 'budget_contract.json', trainer.baseline_contract)
        for signame in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
            if hasattr(signal, signame): signal.signal(getattr(signal, signame), stop_requested)
        started = time.perf_counter(); saved = started; executed = 0
        while not trainer.finished and not STOP_REQUESTED:
            if args.max_steps and executed >= args.max_steps: break
            if args.max_wall_seconds and time.perf_counter() - started >= args.max_wall_seconds: break
            row = trainer.step(); executed += 1
            if executed == 1 or executed % 10 == 0: print(json.dumps(row), flush=True)
            if time.perf_counter() - saved >= args.checkpoint_seconds:
                trainer.save(checkpoint); saved = time.perf_counter()
        if trainer.finished: trainer.audit_endpoint()
        digest = trainer.save(checkpoint)
        write_json(args.output / 'history.json', trainer.history)
        write_json(args.output / 'closure_history.json', trainer.evaluation_history)
        write_json(args.output / 'sampling_coverage.json', trainer.coverage())
        receipt = dict(finished=trainer.finished, adam_completed=trainer.adam_completed,
            lbfgs_completed=trainer.lbfgs_completed, lbfgs_aborted_steps=trainer.lbfgs_aborted_steps,
            closure_budget=trainer.config.closure_budget, closure_calls=trainer.closure_calls,
            residual_point_evaluations=trainer.residual_point_evaluations,
            audit_closure_calls=trainer.audit_closure_calls, audit_residual_point_evaluations=trainer.audit_residual_point_evaluations,
            checkpoint_sha256=digest, runtime_history=trainer.runtime_history,
            elapsed_training_seconds=trainer.elapsed_seconds, process_seconds=time.perf_counter() - started,
            source_sha256=source_hash(), baseline_source_sha256=BASE_SOURCE_SHA256,
            budget_contract_sha256=state_hash(trainer.baseline_contract), stop_requested=STOP_REQUESTED,
            initial_model_sha256=trainer.initial_model_sha256, objective_sha256=trainer.objective_hash)
        # Persist completed recovery identity before creating the final manifest.
        # An interrupted export can therefore always be resumed and independently scored.
        write_json(args.output / 'current_receipt.json', receipt)
        if trainer.finished:
            receipt['frozen_manifest'] = export_frozen(trainer, args.output)
            write_json(args.output / 'current_receipt.json', receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        lock.close()


if __name__ == '__main__': main()
