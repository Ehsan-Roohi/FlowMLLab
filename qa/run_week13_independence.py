"""Bounded, restartable Re500 h/p/dt/corner-resolution study on compute nodes.

Never changes source data or existing campaigns. Every candidate passes its own
mapping, split-restart, clock, boundary and CFL tests before production.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path

from run_week13_nektar_refinement import (accepted_source, preserve_time,
    smoke_boundary_check, local_mapping_check)
from run_week13_nektar_reynolds import run_process, sha, field_time, validate_ascii_vtu
from check_week13_nektar_vtu import compare
from week13_independence import CASES, session, audit, temporal_gate, atomic_json


HEAVY = {'cavity.fld', 'cavity.vtu', 'cavity-vorticity.fld', 'cavity-vorticity.vtu'}


def prune(root, records):
    """Prune only this campaign's accepted intermediate artifacts, never sources.

    Keep two most recent quarter-time checkpoints and all ten-time-unit snapshots.
    Logs, sessions, acceptance hashes, diagnostics, and incomplete attempts remain.
    """
    root = Path(root).resolve()
    for record_path in records[:-2]:
        data = json.loads(record_path.read_text())
        if abs(data['time']/10-round(data['time']/10)) < 1e-8 or data.get('pruned'):
            continue
        folder = (record_path.parent/data['attempt']).resolve()
        if folder.parent != record_path.parent or root not in folder.parents:
            raise ValueError('Unsafe pruning target')
        deleted = []
        for path in folder.iterdir():
            if path.name in HEAVY or re.fullmatch(r'cavity_\d+\.chk', path.name):
                if path.is_symlink() or not path.is_file() or path.resolve().parent != folder:
                    raise ValueError('Unexpected artifact type; no deletion')
                path.unlink()
                deleted.append(path.name)
        data['pruned'] = deleted
        data['retention'] = 'Recent two checkpoints and every ten-time-unit snapshot retained'
        atomic_json(record_path, data)


def run(args):
    case = CASES[args.case]
    work = args.output.resolve()/case['label']
    work.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (work/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.monotonic()+args.seconds
    source, marker, source_spec = accepted_source(args.source_chunk, 500)
    if abs(marker['time']-220) > 1e-7:
        raise ValueError('This study is explicitly initialized from Re500 t220')
    spec = dict(case=case, re=500, depth_over_width=5, start_time=220, end_time=280,
                checkpoint_interval=.25, audit_interval=10,
                corner_convention='stationary_endpoints', source=str(source),
                source_sha256=marker['field_sha256'], image_sha256=sha(args.image),
                code_commit=args.commit, max_requeues=24,
                retention='two recent accepted checkpoints, ten-time-unit fields, all logs and metrics',
                scientific_status='verification_in_progress_not_benchmark_certified')
    manifest = work/'study.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError('Immutable study configuration changed')
    if not manifest.exists():
        atomic_json(manifest, spec)
    base = ['apptainer', 'exec', '--cleanenv', '--bind', '/project', args.image]

    def command(argv, cwd, log):
        status = run_process(base+argv, cwd, Path(cwd)/log, deadline, reserve=120)
        if status:
            raise subprocess.CalledProcessError(status, argv)

    def export(directory, field='cavity.fld', name='cavity.vtu'):
        command(['FieldConvert', 'cavity.xml', field, name+':vtu:uncompress'], directory, name+'.log')
        validate_ascii_vtu(directory/name)

    def solve(directory, initial, interval):
        session(directory, case, restart=str(initial), interval=interval)
        command(['IncNavierStokesSolver', 'cavity.xml'], directory, 'solver.log')
        actual = field_time(directory/'cavity.fld')
        if abs(actual-field_time(initial)-interval) > 1e-7:
            raise ValueError('Solver restart clock mismatch')
        cfl = [float(v) for v in re.findall(r'CFL\s*(?:=|:)\s*([0-9.eE+-]+)',
                                             (directory/'solver.log').read_text())]
        if not cfl or not all(math.isfinite(v) and v < .8 for v in cfl):
            raise ValueError(f'Missing/excessive CFL: {cfl}')
        return actual, max(cfl)

    ready = work/'initial-state.json'
    if not ready.exists():
        attempts = list(work.glob('prepare-*'))
        if len(attempts) >= 4:
            raise ValueError('Four incomplete preparation attempts; review required')
        dest = work/f'prepare-{len(attempts):02d}'
        session(dest, case, interval=.5)
        mapped = dest/'mapped.fld'
        if case['nx'] == 16 and case['ny'] == 80 and case['order'] == 6:
            shutil.copyfile(source/'cavity.fld', mapped)
        else:
            command(['FieldConvert', '-m', f'interpfield:fromxml={source}/cavity.xml:fromfld={source}/cavity.fld',
                     'cavity.xml', 'mapped.fld'], dest, 'mapping.log')
        preserve_time(mapped, marker['time'])
        command(['FieldConvert', '-m', f'interpfield:fromxml={dest}/cavity.xml:fromfld={mapped}',
                 str(source/'cavity.xml'), 'roundtrip.fld'], dest, 'roundtrip.log')
        command(['FieldConvert', str(source/'cavity.xml'), 'roundtrip.fld',
                 'roundtrip.vtu:vtu:uncompress'], dest, 'roundtrip-export.log')
        mapping = compare(source/'cavity.vtu', dest/'roundtrip.vtu', atol=1e-8, rtol=1e-6)
        atomic_json(dest/'mapping-check.json', mapping)
        if not mapping['numeric_comparison_passed']:
            raise ValueError('Mapping roundtrip failed')
        weak = local_mapping_check(source/'cavity.vtu', dest/'roundtrip.vtu')
        whole, half, split = dest/'whole', dest/'half', dest/'split'
        # Compare at the actual production restart cadence. IMEX2 startup loses
        # multistep history; a two-by-.01 test exposed startup differences and
        # remains retained as failed evidence, not silently accepted or erased.
        actual, cfl = solve(whole, mapped, .5)
        solve(half, mapped, .25)
        split_actual, split_cfl = solve(split, half/'cavity.fld', .25)
        export(whole)
        export(split)
        restart = compare(whole/'cavity.vtu', split/'cavity.vtu', atol=1e-8, rtol=1e-5)
        atomic_json(dest/'restart-check.json', restart)
        if abs(actual-split_actual) > 1e-7 or not restart['numeric_comparison_passed']:
            raise ValueError('Matched-time split restart failed')
        weak_restart = local_mapping_check(whole/'cavity.vtu', split/'cavity.vtu')
        boundaries = smoke_boundary_check(whole/'cavity.vtu')
        # Production starts at the mapped physical time, not the smoke end time.
        baseline = audit(source/'cavity.vtu', CASES[0])
        baseline['time'] = 220.
        atomic_json(work/'audit-0880.json', baseline)
        atomic_json(ready, dict(field=str(mapped), field_sha256=sha(mapped), time=marker['time'],
                    mapping=mapping, weak_vortices=weak, restart=restart,
                    restart_interval=.25, restart_comparison_horizon=.5,
                    weak_restart=weak_restart,
                    boundaries=boundaries, smoke_max_cfl=max(cfl, split_cfl),
                    operability_passed=True, benchmark_certified=False))
        print('INITIAL_STATE_VERIFIED', case['label'], flush=True)
    initial = json.loads(ready.read_text())
    if sha(initial['field']) != initial['field_sha256']:
        raise ValueError('Mapped field hash changed')
    records = sorted(work.glob('step-*/accepted.json'))
    if (work/'stopped.json').exists():
        print('ALREADY_STOPPED', case['label'], flush=True)
        return 0
    if records:
        last = json.loads(records[-1].read_text())
        prior = records[-1].parent/last['attempt']/'cavity.fld'
        if last.get('clean_exit') is not True or sha(prior) != last['field_sha256']:
            raise ValueError('Latest accepted restart is invalid')
        if abs(field_time(prior)-last['time']) > 1e-7:
            raise ValueError('Latest accepted restart time is invalid')
        current = last['time']
    else:
        prior, current = Path(initial['field']), initial['time']
    for index in range(round(current*4), 280*4):
        if time.monotonic() > deadline-300:
            return 75
        chunk = work/f'step-{index+1:04d}'
        chunk.mkdir(exist_ok=True)
        number = len(list(chunk.glob('attempt-*')))
        if number >= 4:
            raise ValueError('Four incomplete attempts at same step; review required')
        attempt = chunk/f'attempt-{number:02d}'
        actual, cfl = solve(attempt, prior, .25)
        if abs(actual-(index+1)/4) > 1e-7:
            raise ValueError('Quarter-time sequence mismatch')
        is_audit = (index+1) % 40 == 0
        extra = {}
        export(attempt)  # Every accepted checkpoint has a finite primitive field.
        if is_audit:
            command(['FieldConvert', '-m', 'vorticity', 'cavity.xml', 'cavity.fld',
                     'cavity-vorticity.fld'], attempt, 'vorticity.log')
            command(['FieldConvert', 'cavity.xml', 'cavity-vorticity.fld',
                     'cavity-vorticity.vtu:vtu:uncompress'], attempt, 'vorticity-export.log')
            validate_ascii_vtu(attempt/'cavity-vorticity.vtu', require_vorticity=True)
            result = audit(attempt/'cavity.vtu', case)
            result.update(time=actual, boundaries=smoke_boundary_check(attempt/'cavity.vtu'))
            atomic_json(work/f'audit-{index+1:04d}.json', result)
            extra = {name:sha(attempt/name) for name in HEAVY if (attempt/name).exists()}
        record = dict(attempt=attempt.name, time=actual, field_sha256=sha(attempt/'cavity.fld'),
                      clean_exit=True, max_cfl=cfl, audit_artifact_hashes=extra,
                      acceptance_scope='clean solver exit and clock; not scientific convergence')
        atomic_json(chunk/'accepted.json', record)
        prior, current = attempt/'cavity.fld', actual
        records.append(chunk/'accepted.json')
        prune(work, records)
        atomic_json(work/'progress.json', dict(time=actual, max_cfl=cfl,
                    checkpoint=str(prior), restart_hash=record['field_sha256']))
        print('ACCEPTED', case['label'], format(actual,'.6f'), flush=True)
        if is_audit:
            audits = [json.loads(p.read_text()) for p in sorted(work.glob('audit-*.json'))]
            gate = temporal_gate(audits)
            atomic_json(work/'temporal-gate.json', gate)
            if gate['passed']:
                atomic_json(work/'stopped.json', dict(reason='two_window_temporal_gate',
                            time=actual, benchmark_certified=False))
                print('TEMPORALLY_SETTLED', case['label'], actual, flush=True)
                return 0
    atomic_json(work/'stopped.json', dict(reason='bounded_end_time_requires_review',
                                        time=current, benchmark_certified=False))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=int, choices=range(len(CASES)), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-chunk', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--seconds', type=int, default=6600)
    args = p.parse_args()
    try:
        raise SystemExit(run(args))
    except subprocess.TimeoutExpired:
        print('WALL_BUDGET_REQUEUE_FROM_LAST_ACCEPTED', flush=True)
        raise SystemExit(75)
