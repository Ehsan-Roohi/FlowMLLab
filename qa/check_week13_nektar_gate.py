"""Predeclared short-time restart gate. Does not certify CFD convergence."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from check_week13_nektar_vtu import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--array', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cases = []
    for task in range(6):
        directory = Path(args.root)/f'restart-check-{args.array}-{task}'
        times = [float(ET.parse(directory/name/'cavity.fld').findtext('./Metadata/Time'))
                 for name in ('continuous', 'restarted')]
        result = compare(directory/'continuous/cavity.vtu', directory/'restarted/cavity.vtu',
                         atol=1e-8, rtol=1e-5)
        result['time_match_verified'] = all(abs(t-0.1) < 1e-10 for t in times)
        result['times'] = times
        result['task'] = task
        cases.append(result)
    gate = dict(gate_passed=all(c['numeric_comparison_passed'] and c['time_match_verified'] for c in cases),
                time_match_verified=all(c['time_match_verified'] for c in cases),
                Re=100, orders=[4,6,8], dt=[0.0005,0.00025],
                scope='short_time_restart_operability_and_agreement_not_steady_accuracy', cases=cases)
    output.write_text(json.dumps(gate, indent=2, allow_nan=False)+'\n')
    print(json.dumps(gate, indent=2, allow_nan=False))
    return 0 if gate['gate_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
