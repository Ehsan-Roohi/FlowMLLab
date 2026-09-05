"""Run the inexpensive follow-up checks in scratch, never overwrite evidence."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def retained_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('data', 'results', 'notebooks', 'lectures')
            for p in (ROOT / folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if any(out.is_relative_to(ROOT / folder) for folder in
           ('data', 'results', 'notebooks', 'lectures')):
        parser.error('Output must be scratch, outside retained evidence directories')
    out.mkdir(parents=True, exist_ok=False)
    before = retained_hashes()
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    for name, suffix in [('JUPYTER_CONFIG_DIR', 'config'), ('JUPYTER_RUNTIME_DIR', 'runtime'),
                         ('IPYTHONDIR', 'ipython'), ('TMP', 'temp'), ('TEMP', 'temp')]:
        path = out / suffix
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    env['JUPYTER_PATH'] = str(out / 'kernel' / 'share' / 'jupyter')
    env['PYTHONPATH'] = str(ROOT)
    env['MPLBACKEND'] = 'Agg'
    checks = []

    def command(name, argv):
        started = time.perf_counter()
        run = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env,
                             text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / (name + '.log')).write_text(run.stdout, encoding='utf-8')
        checks.append({'check': name, 'exit_code': run.returncode,
                       'seconds': time.perf_counter() - started,
                       'test_summary': re.findall(r'Ran \d+ tests? in [^\n]+|OK(?: \(skipped=\d+\))?|FAILED \([^\n]+', run.stdout)[-3:]})
        print(json.dumps(checks[-1]), flush=True)
        return run.returncode == 0

    plan = {'purpose': 'reproduction and execution audit, not new research accuracy',
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'checks': ['unit_tests', 'v5_protocol_tests', 'modal_notebooks', 'week12_fit', 'week11_week12_notebooks']}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    command('unit_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
    command('v5_protocol_tests', ['-I', 'qa/test_step_architecture_v5.py'])
    kernel_ok = command('kernel_setup', ['-m', 'ipykernel', 'install', '--prefix', str(out / 'kernel'), '--name', 'python3'])
    if kernel_ok:
        command('modal_notebooks', ['qa/verify_modal_labs.py', '--output', str(out / 'modal')])
    command('week12_fit', ['qa/run_week12_noise2noise.py', '--output', str(out / 'week12')])
    if checks[-1]['exit_code'] == 0:
        import numpy as np
        import pandas as pd
        actual = pd.read_csv(out / 'week12' / 'metrics.csv')
        expected = pd.read_csv(ROOT / 'results/week12_noise2noise/metrics.csv')
        numeric = expected.select_dtypes(include='number').columns
        same_labels = actual.drop(columns=numeric).equals(expected.drop(columns=numeric))
        close = actual.shape == expected.shape and same_labels and bool(np.allclose(
            actual[numeric], expected[numeric], rtol=.02, atol=1e-5, equal_nan=True))
        checks.append({'check': 'week12_all_metric_reproduction', 'exit_code': 0 if close else 1,
                       'rows': len(actual), 'relative_tolerance': .02, 'absolute_tolerance': 1e-5})
    if kernel_ok:
        # A child process inherits the isolated kernel directories, without
        # changing the user's global Jupyter installation or notebook files.
        command('week11_week12_notebooks', ['qa/run_local_followup.py', '--execute-notebooks', str(out / 'notebooks')])
    after = retained_hashes()
    changed = [p for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
    summary = {**plan, 'checks': checks, 'retained_files_checked': len(before),
               'retained_files_unchanged': not changed, 'changed_retained_files': changed,
               'python': platform.python_version(), 'platform': platform.system(),
               'packages': {p: importlib.metadata.version(p) for p in
                            ('numpy', 'scipy', 'pandas', 'scikit-learn', 'matplotlib', 'nbclient')},
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'source_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                   ('qa/run_local_followup.py', 'qa/test_step_architecture_v5.py',
                    'qa/verify_modal_labs.py', 'qa/run_week12_noise2noise.py',
                    'flowmllab/modal_experiments.py', 'flowmllab/noise2noise.py')}}
    summary['pass'] = all(c['exit_code'] == 0 for c in checks) and summary['retained_files_unchanged']
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['pass'] else 1


def execute_notebooks(destination):
    import nbformat
    from nbclient import NotebookClient
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('week11/W11_Shock_Vortex_Identification.ipynb', 'week12/W12_DSMC_Moment_Reconstruction.ipynb'):
        source = ROOT / 'notebooks' / name
        nb = nbformat.read(source, as_version=4)
        nbformat.validate(nb)
        NotebookClient(nb, timeout=300, kernel_name='python3', resources={'metadata': {'path': str(source.parent)}}).execute()
        nbformat.write(nb, destination / source.name)
        print('EXECUTED ' + name, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--execute-notebooks':
        execute_notebooks(Path(sys.argv[2]))
    else:
        raise SystemExit(main())
