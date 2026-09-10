"""Map accepted cavity checkpoints to nested meshes, verify, then continue.

All interpolation and solver execution belong on a compute node. Source data
are read-only. A successful mapping is initialization, not benchmark validation.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from audit_week13_nektar import structured, integrate_paths
from audit_week13_cfd import extrema
from check_week13_nektar_vtu import compare, read_vtu
from prepare_week13_nektar import make_case
from run_week13_nektar_reynolds import run, run_process, sha, field_time, validate_ascii_vtu


def accepted_source(chunk, reynolds):
    chunk = Path(chunk).resolve()
    marker = json.loads((chunk / "accepted.json").read_text())
    spec = json.loads((chunk.parent / "campaign.json").read_text())
    if (marker.get("clean_exit") is not True or spec["re"] != reynolds
            or spec["order"] != 6 or spec["corner_convention"] != "stationary_endpoints"
            or spec["depth_over_width"] != 5):
        raise ValueError("Source case does not match the declared physics")
    attempt = (chunk / marker["attempt"]).resolve()
    if attempt.parent != chunk:
        raise ValueError("Source attempt is outside the accepted chunk")
    for key, name in (("field_sha256", "cavity.fld"),
                      ("primitive_vtu_sha256", "cavity.vtu"),
                      ("vorticity_vtu_sha256", "cavity-vorticity.vtu")):
        if sha(attempt / name) != marker[key]:
            raise ValueError(f"Source hash mismatch: {name}")
    if abs(field_time(attempt / "cavity.fld") - marker["time"]) > 1e-7:
        raise ValueError("Source physical time mismatch")
    return attempt, marker, spec


def preserve_time(path, physical_time):
    """Set only the mapped file's restart clock; never touch source coefficients."""
    if not math.isfinite(physical_time) or physical_time < 0:
        raise ValueError('Restart time must be finite and nonnegative')
    tree = ET.parse(path)
    nodes = [el for el in tree.getroot().iter() if el.tag.lower() == "time"]
    if not nodes:
        metadata = tree.getroot().find("Metadata")
        if metadata is None:
            metadata = ET.SubElement(tree.getroot(), "Metadata")
        nodes = [ET.SubElement(metadata, "Time")]
    for node in nodes:
        node.text = format(physical_time, ".17g")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    assert abs(field_time(path) - physical_time) < 1e-7


def smoke_boundary_check(path):
    """Enforce no-slip walls and central lid; report finite-p corner projection."""
    parts = read_vtu(path)
    xyz = np.concatenate([part[0] for part in parts])
    u = np.concatenate([part[1]['u'] for part in parts])
    v = np.concatenate([part[1]['v'] for part in parts])
    x, y = xyz[:, 0], xyz[:, 1]
    walls = np.isclose(x, 0, atol=1e-12, rtol=0) | np.isclose(x, 1, atol=1e-12, rtol=0) | np.isclose(y, 0, atol=1e-12, rtol=0)
    lid = np.isclose(y, 5, atol=1e-12, rtol=0) & ~walls
    central = lid & (x >= .125) & (x <= .875)
    if not walls.any() or not central.any() or not lid.any():
        raise ValueError('Smoke export lacks boundary samples')
    report = dict(wall_max_speed=float(np.max(np.hypot(u[walls], v[walls]))),
                  central_lid_max_error=float(np.max(np.hypot(u[central]-1, v[central]))),
                  full_lid_max_error=float(np.max(np.hypot(u[lid]-1, v[lid]))),
                  tolerance=1e-7,
                  scope='operability only; full-lid projection reported, not benchmark-certified')
    if report['wall_max_speed'] > 1e-7 or report['central_lid_max_error'] > 1e-7:
        raise ValueError(f'Mapped restart boundary check failed: {report}')
    return report


def local_mapping_check(source_vtu, roundtrip_vtu):
    x, y, a, _ = structured(read_vtu(source_vtu))
    xb, yb, b, _ = structured(read_vtu(roundtrip_vtu))
    if not np.array_equal(x, xb) or not np.array_equal(y, yb):
        raise ValueError("Roundtrip grid mismatch")
    pa, _ = integrate_paths(x, y, a['u'], a['v'])
    pb, _ = integrate_paths(x, y, b['u'], b['v'])
    original, mapped = extrema(pa, x, y), extrema(pb, x, y)
    records = []
    # Inspect all detected main vortices; no paper values used to accept mapping.
    for q in original:
        candidates = [v for v in mapped if v['psi'] * q['psi'] > 0
                      and abs(v['y'] - q['y']) < .1]
        if not candidates:
            raise ValueError("Mapping lost a main vortex")
        v = min(candidates, key=lambda v: math.hypot(v['x']-q['x'], v['y']-q['y']))
        change = abs(v['psi']/q['psi'] - 1)
        shift = math.hypot(v['x']-q['x'], v['y']-q['y'])
        records.append(dict(source=q, relative_strength_change=change, centre_shift_W=shift))
        if change >= .001 or shift >= .001:
            raise ValueError(f"Mapping changes a weak vortex: {records[-1]}")
    if len(records) < 4:
        raise ValueError("Expected at least four main source vortices")
    return records


def prepare(args, source, marker, source_spec, work, deadline):
    ready = work / 'initial-state.json'
    if ready.exists():
        state = json.loads(ready.read_text())
        if state['source_field_sha256'] != marker['field_sha256']:
            raise ValueError('Source changed on resume')
        return ready
    attempts = sorted(work.glob('map-attempt-*'))
    if len(attempts) >= 4:
        raise ValueError('Mapping attempt budget exhausted; review required')
    dest = work / f'map-attempt-{len(attempts):02d}'
    dest.mkdir()
    base = ['apptainer', 'exec', '--cleanenv', '--bind', '/project', args.image]

    def command(argv, log, directory=dest):
        rc = run_process(base + argv, directory, dest / log, deadline)
        if rc:
            raise subprocess.CalledProcessError(rc, argv)

    make_case(dest, re=args.re, order=6, nx=args.nx, ny=args.ny,
              dt=.00025, steps=80, check_steps=80, corner_convention='stationary_endpoints')
    mapped = dest / 'mapped.fld'
    same_mesh = (args.nx, args.ny) == (source_spec['nx'], source_spec['ny'])
    if same_mesh:
        shutil.copyfile(source / 'cavity.fld', mapped)
    else:
        if (args.nx, args.ny) != (2*source_spec['nx'], 2*source_spec['ny']):
            raise ValueError('Only factor-two nested h-refinement is qualified here')
        command(['FieldConvert', '-m', f'interpfield:fromxml={source}/cavity.xml:fromfld={source}/cavity.fld',
                 'cavity.xml', 'mapped.fld'], 'map.log')
    preserve_time(mapped, marker['time'])
    command(['FieldConvert', 'cavity.xml', 'mapped.fld', 'mapped.vtu:vtu:uncompress'], 'map-export.log')
    validate_ascii_vtu(dest / 'mapped.vtu')
    # Return to the source grid; verify field and tiny-vortex preservation.
    command(['FieldConvert', '-m', f'interpfield:fromxml={dest}/cavity.xml:fromfld={mapped}',
             str(source / 'cavity.xml'), 'roundtrip.fld'], 'roundtrip.log')
    command(['FieldConvert', str(source / 'cavity.xml'), 'roundtrip.fld',
             'roundtrip.vtu:vtu:uncompress'], 'roundtrip-export.log')
    comparison = compare(source / 'cavity.vtu', dest / 'roundtrip.vtu', atol=1e-8, rtol=1e-6)
    if not all(item['passed'] for item in comparison['fields'].values()):
        (dest / 'failed-roundtrip.json').write_text(json.dumps(comparison, indent=2))
        raise ValueError('Mapping roundtrip field tolerance failed')
    vortices = local_mapping_check(source / 'cavity.vtu', dest / 'roundtrip.vtu')
    # Independent short solve from mapped coefficients checks clock, BCs and CFL.
    smoke = dest / 'smoke'
    make_case(smoke, re=args.re, order=6, nx=args.nx, ny=args.ny,
              restart=str(mapped), dt=.00025, steps=80, check_steps=80,
              corner_convention='stationary_endpoints', purpose='mapped_restart_operability')
    command(['IncNavierStokesSolver', 'cavity.xml'], 'smoke.log', smoke)
    actual = field_time(smoke / 'cavity.fld')
    if abs(actual - marker['time'] - .02) > 1e-7:
        raise ValueError(f'Mapped restart clock failed: {actual}')
    command(['FieldConvert', 'cavity.xml', 'cavity.fld', 'smoke.vtu:vtu:uncompress'], 'smoke-export.log', smoke)
    validate_ascii_vtu(smoke / 'smoke.vtu')
    boundaries = smoke_boundary_check(smoke / 'smoke.vtu')
    log = (dest / 'smoke.log').read_text()
    cfl = [float(v) for v in re.findall(r'CFL\s*(?:=|:)\s*([0-9.eE+-]+)', log)]
    if not cfl or not all(math.isfinite(v) and v < .8 for v in cfl):
        raise ValueError(f'Missing/excessive smoke CFL: {cfl}')
    state = dict(re=args.re, nx=args.nx, ny=args.ny, order=6, dt=.00025,
                 corner_convention='stationary_endpoints', time=marker['time'],
                 field=str(mapped), field_sha256=sha(mapped),
                 source=str(source), source_field_sha256=marker['field_sha256'],
                 mapping='same-mesh copy' if same_mesh else 'FieldConvert interpfield; nested h refinement',
                 time_metadata='explicitly preserved from accepted source; no elapsed evolution during mapping',
                 roundtrip=comparison, weak_vortices=vortices,
                 smoke_end_time=actual, smoke_max_cfl=max(cfl),
                 smoke_boundaries=boundaries,
                 operability_passed=True, benchmark_validated=False)
    temporary = work / 'initial-state.tmp'
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(ready)
    print(f'INITIAL_STATE_VERIFIED {ready}', flush=True)
    return ready


def main():
    import fcntl  # Compute-node lock; permit read-only unit tests on Windows.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-chunk', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--re', type=int, choices=[500, 1000], required=True)
    p.add_argument('--nx', type=int, choices=[16, 32], required=True)
    p.add_argument('--ny', type=int, choices=[80, 160], required=True)
    p.add_argument('--end-time', type=int, required=True)
    p.add_argument('--seconds', type=int, default=6600)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, marker, spec = accepted_source(args.source_chunk, args.re)
        start = time.monotonic()
        try:
            state = prepare(args, source, marker, spec, args.output, start+args.seconds)
        except subprocess.TimeoutExpired:
            return 75
        remaining = args.seconds - (time.monotonic()-start)
        if remaining < 240:
            return 75
        return run(SimpleNamespace(output=str(args.output / 'run'), image=args.image,
                    gate=args.gate, commit=args.commit, re=args.re, order=6,
                    corner_convention='stationary_endpoints', dt=.00025,
                    nx=args.nx, ny=args.ny, end_time=args.end_time,
                    seconds=int(remaining), initial_state=str(state)))


if __name__ == '__main__':
    raise SystemExit(main())
