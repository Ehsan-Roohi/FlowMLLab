"""Bounded transient pilot; accepted chunks are immutable, not steady evidence.

Only successfully exited solver + validated export chunks can become restart
sources. Interrupted attempts are retained, never overwritten or resumed.
Restart equivalence is a prerequisite checked outside this runner.
"""
import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

from prepare_week13_nektar import make_case


def sha(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
        return digest.hexdigest()


def validate_vtu(path):
    root = ET.parse(path).getroot()
    required = {'u', 'v', 'p'}
    pieces = root.findall('.//Piece')
    if not pieces:
        raise ValueError('No VTU pieces')
    for piece in pieces:
        count = int(piece.get('NumberOfPoints', '0'))
        found = set()
        arrays = piece.findall('./PointData/DataArray') + piece.findall('./Points/DataArray')
        coordinates = False
        for array in arrays:
            name = array.get('Name', '')
            is_coordinates = array in piece.findall('./Points/DataArray')
            if name not in required and not is_coordinates:
                continue
            if array.get('format', 'ascii').lower() != 'ascii':
                raise ValueError('Finite-field gate requires ASCII VTU')
            values = [float(v) for v in (array.text or '').split()]
            expected = count * (3 if is_coordinates else 1)
            if count < 1 or len(values) != expected or not all(map(math.isfinite, values)):
                raise ValueError(f'Wrong count/nonfinite field {name}')
            if is_coordinates:
                coordinates = True
            else:
                found.add(name)
        if found != required or not coordinates:
            raise ValueError('Missing required scalar fields or coordinates')


def field_time(path):
    root = ET.parse(path).getroot()
    for element in root.iter():
        if element.tag.lower() == 'time' and element.text:
            return float(element.text)
    raise ValueError('No Time metadata in final field; fail closed')


def run(args):
    if not math.isfinite(args.dt) or args.dt <= 0 or args.seconds <= 120:
        raise ValueError('Positive finite dt and wall budget above 120 seconds required')
    case = Path(args.output).resolve()
    case.mkdir(parents=True, exist_ok=True)
    config = dict(re=100, order=args.order, dt=args.dt, chunk_time=1.0,
                  end_time=args.end_time, nx=8, ny=40,
                  image_sha256=sha(Path(args.image)),
                  code_commit=args.commit, status='transient_pilot_not_steady_evidence')
    manifest = case/'campaign.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError('Campaign specification changed; use a new output directory')
    else:
        manifest.write_text(json.dumps(config, indent=2)+'\n')
    deadline = time.monotonic() + args.seconds
    prior = None
    current = 0.0
    chunk_count = round(args.end_time)
    if chunk_count != args.end_time or chunk_count < 1:
        raise ValueError('end-time must be a positive integer')
    base = ['apptainer', 'exec', '--cleanenv', '--bind', '/project', args.image]
    for index in range(chunk_count):
        chunk = case/f'chunk-{index:04d}'
        chunk.mkdir(exist_ok=True)
        marker = chunk/'accepted.json'
        if marker.exists():
            accepted = json.loads(marker.read_text())
            prior = chunk/accepted['attempt']/'cavity.fld'
            if sha(prior) != accepted['field_sha256']:
                raise ValueError('Accepted restart hash changed')
            current = accepted['time']
            if abs(current-(index+1)) > 1e-7:
                raise ValueError('Accepted time sequence mismatch')
            continue
        if time.monotonic() >= deadline-120:
            return 75
        attempt_number = len(list(chunk.glob('attempt-*')))
        if attempt_number >= 4:
            raise RuntimeError('Four incomplete attempts: stop for human review')
        attempt = chunk/f'attempt-{attempt_number:02d}'
        attempt.mkdir()
        steps = round(1.0/args.dt)
        if abs(steps*args.dt-1) > 1e-12:
            raise ValueError('dt must divide chunk duration exactly')
        make_case(attempt, re=100, order=args.order, restart=str(prior) if prior else None,
                  dt=args.dt, steps=steps, check_steps=steps,
                  purpose='bounded_transient_pilot_not_steady_evidence')
        try:
            with (attempt/'solver.log').open('w') as log:
                process = subprocess.Popen(base+['IncNavierStokesSolver', 'cavity.xml'],
                                           cwd=attempt, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                try:
                    status = process.wait(timeout=max(1, deadline-time.monotonic()-60))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
                if status:
                    raise subprocess.CalledProcessError(status, process.args)
        except subprocess.TimeoutExpired:
            (attempt/'interrupted-budget.txt').write_text('Unaccepted attempt; never used as restart.\n')
            return 75
        final = attempt/'cavity.fld'
        actual = field_time(final)
        if not math.isfinite(actual) or abs(actual-(current+1.0)) > 1e-7:
            raise ValueError(f'Unexpected final time {actual}, expected {current+1}')
        with (attempt/'convert.log').open('w') as log:
            subprocess.run(base+['FieldConvert', 'cavity.xml', 'cavity.fld',
                                'cavity.vtu:vtu:uncompress'], cwd=attempt,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        validate_vtu(attempt/'cavity.vtu')
        record = dict(attempt=attempt.name, time=actual, field_sha256=sha(final),
                      vtu_sha256=sha(attempt/'cavity.vtu'), clean_exit=True,
                      numerical_accuracy='not_yet_validated')
        temporary = chunk/'accepted.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        os.replace(temporary, marker)
        prior, current = final, actual
    (case/'pilot-complete-needs-scientific-review').write_text(
        'Reached bounded transient end time. NOT a steady-state or vortex-accuracy certificate.\n')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--order', type=int, choices=[4, 6, 8], required=True)
    p.add_argument('--dt', type=float, default=0.0005)
    p.add_argument('--end-time', type=float, default=20)
    p.add_argument('--seconds', type=float, default=6000)
    raise SystemExit(run(p.parse_args()))
