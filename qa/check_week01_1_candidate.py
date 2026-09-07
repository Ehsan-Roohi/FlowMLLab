"""Check a student's boundary flux implementation; never substitute a solution."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def assess(flux):
    from flowmllab.scientific_software import load_cavity_case, validate_week01_1_evidence

    checks = []

    def check(level, name, operation):
        try:
            operation()
            checks.append(dict(level=level, name=name, passed=True))
        except Exception as exc:
            checks.append(dict(level=level, name=name, passed=False, error=str(exc)))

    def expect(x, y, u, v, reference):
        value = flux(x, y, u, v)
        if np.ndim(value) != 0 or not np.isfinite(value) or abs(float(value)-reference) > 1e-12:
            raise AssertionError(f'expected {reference}, observed {value}')

    x = np.linspace(0, 1, 17)
    y = np.linspace(0, 1, 23)
    xx, yy = np.meshgrid(x, y)
    zero = np.zeros_like(xx)

    def invalid_inputs():
        bad = zero.copy(); bad[0, 0] = np.nan
        for args in [(x, y, zero[:-1], zero), (x[::-1], y, zero, zero),
                     (x*0, y, zero, zero), (x, y, bad, zero),
                     (x[:1], y, zero[:, :1], zero[:, :1])]:
            try:
                flux(*args)
            except ValueError:
                continue
            raise AssertionError('invalid input did not raise ValueError')

    check('unit', 'invalid inputs', invalid_inputs)
    check('physical invariant', 'constant velocity', lambda: expect(x, y, xx*0+2, yy*0-3, 0))
    check('baseline/reference', 'linear expansion', lambda: expect(x, y, xx, yy, 2))
    check('baseline/reference', 'quadratic velocity', lambda: expect(x, y, xx**2, zero, 1))
    # Rectangle area is 6; div(x,y)=2, so exact outward flux is 12.
    xr = np.array([0., .2, .8, 2.]); yr = np.array([0., .1, .7, 1.6, 3.])
    rx, ry = np.meshgrid(xr, yr)
    check('baseline/reference', 'nonuniform rectangle', lambda: expect(xr, yr, rx, ry, 12))

    def cavity_checks():
        case, _ = load_cavity_case(ROOT)
        expect(case['x'], case['y'], case['u'], case['v'], 0)
        altered = case['u'].copy(); altered[:, -1] += .1
        expect(case['x'], case['y'], altered, case['v'], .1)

    check('physical invariant', 'cavity and deliberate boundary leak', cavity_checks)
    check('numerical regression', 'retained derivative evidence', lambda: validate_week01_1_evidence(ROOT))
    return {'decision': 'accept' if all(c['passed'] for c in checks) else 'reject', 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path, help='Student Python file to import and execute')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('student_flux_candidate', args.candidate.resolve())
    if spec is None or spec.loader is None:
        parser.error('candidate must be a Python module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = assess(module.net_volume_flux)
    print(json.dumps(report, indent=2))
    return 0 if report['decision'] == 'accept' else 1


if __name__ == '__main__':
    raise SystemExit(main())
