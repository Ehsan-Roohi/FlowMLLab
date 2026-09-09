"""Predeclared short-time restart gate. Does not certify CFD convergence."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from check_week13_nektar_vtu import compare


def validate_spec(spec, directory, name, dt, order, convention):
    expected = dict(Re=100, depth=5, width=1, nx=8, ny=40, order=order,
                    corner_convention=convention)
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'{directory}/{name}: mismatched {key}')
    if not math.isclose(spec.get('dt', float('nan')), dt, rel_tol=0, abs_tol=1e-15):
        raise ValueError(f'{directory}/{name}: mismatched dt')
    duration = .1 if name == 'continuous' else .05
    if spec.get('steps') != round(duration/dt) or abs(round(duration/dt)*dt-duration) > 1e-12:
        raise ValueError(f'{directory}/{name}: mismatched duration/steps')
    restart = spec.get('restart')
    if name == 'restarted':
        if not restart or Path(restart).resolve() != (directory/'first/cavity.fld').resolve():
            raise ValueError('Restart input is not the first segment field')
    elif restart is not None:
        raise ValueError('Continuous/first segment unexpectedly restarts')
    expression = '(x>0)*(x<1)' if convention == 'stationary_endpoints' else '1'
    if spec.get('lid_expression') != expression:
        raise ValueError('Lid expression does not match corner convention')


def boundary_passed(observations):
    values = [observations.get('wall_max_speed'),
              observations.get('central_lid_x_0_125_to_0_875', {}).get('max_velocity_error')]
    return all(value is not None and math.isfinite(value) and value <= 1e-10 for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--array', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dt-base', type=float, required=True)
    parser.add_argument('--corner-convention', required=True,
                        choices=['legacy_conflicting', 'stationary_endpoints'])
    args = parser.parse_args()
    if not math.isfinite(args.dt_base) or args.dt_base <= 0:
        parser.error('--dt-base must be positive and finite')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for task in range(6):
        directory = Path(args.root)/f'restart-check-{args.array}-{task}'
        dt = args.dt_base if task < 3 else args.dt_base/2
        order = [4, 6, 8][task % 3]
        times, segment_boundaries = {}, {}
        for name in ('continuous', 'first', 'restarted'):
            spec = json.loads((directory/name/'spec.json').read_text())
            validate_spec(spec, directory, name, dt, order, args.corner_convention)
            times[name] = float(ET.parse(directory/name/'cavity.fld').findtext('./Metadata/Time'))
            vtu = directory/name/'cavity.vtu'
            segment_boundaries[name] = compare(vtu, vtu, atol=1e-8, rtol=1e-5)['candidate_boundary_observations']
        result = compare(directory/'continuous/cavity.vtu', directory/'restarted/cavity.vtu',
                         atol=1e-8, rtol=1e-5)
        result['time_match_verified'] = all(math.isfinite(t) and abs(t-(.05 if name == 'first' else .1)) < 1e-10
                                            for name, t in times.items())
        result['times'] = times
        result['task'] = task
        result['dt'] = dt
        result['order'] = order
        result['segment_boundaries'] = segment_boundaries
        result['boundary_operability_passed'] = all(boundary_passed(b) for b in segment_boundaries.values())
        cases.append(result)
    gate = dict(gate_passed=all(c['numeric_comparison_passed'] and c['time_match_verified'] and c['boundary_operability_passed'] for c in cases),
                time_match_verified=all(c['time_match_verified'] for c in cases),
                boundary_operability_passed=all(c['boundary_operability_passed'] for c in cases),
                corner_convention=args.corner_convention, boundary_convention=args.corner_convention,
                boundary_tolerance=1e-10, atol=1e-8, rtol=1e-5,
                Re=100, orders=[4,6,8], dt=[args.dt_base,args.dt_base/2],
                scope='short_time_restart_operability_and_agreement_not_steady_accuracy',
                boundary_scope='stationary walls and central lid only; full and nearcorner lid errors retained in all cases',
                full_lid_accuracy_certified=False, cases=cases)
    with output.open('x') as stream:
        stream.write(json.dumps(gate, indent=2, allow_nan=False)+'\n')
    print(json.dumps(gate, indent=2, allow_nan=False))
    return 0 if gate['gate_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
