"""Execute the cavitation notebook and check linked public assets.

The notebook executes sequentially in a fresh kernel using the QA interpreter.
No research training runs unless the notebook's optional switch is changed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import time
from urllib.parse import unquote, urlsplit

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def hashes():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ['data/week11_cavitation', 'results/week11_cavitation']
            for p in (ROOT/folder).rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--retain-executed', action='store_true')
    args = parser.parse_args()
    scratch = ROOT/'tmp/week11_cavitation_qa'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['IPYTHONDIR'] = str(scratch/'ipython')
    os.environ['JUPYTER_CONFIG_DIR'] = str(scratch/'config')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(scratch/'runtime')
    os.environ['JUPYTER_PATH'] = str(scratch/'jupyter')
    kernel = scratch/'jupyter/kernels/flowmllab-week11'
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name':'FlowMLLab QA', 'language':'python'}))
    os.chdir(ROOT)
    path = ROOT/'notebooks/week11/W11_Cavitation_Cloud_Detection.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    before, started = hashes(), time.perf_counter()
    executed = NotebookClient(nb, timeout=600, kernel_name='flowmllab-week11',
        resources={'metadata':{'path':str(ROOT)}}).execute()
    assert before == hashes(), 'Student execution modified retained evidence'
    code = [c for c in executed.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    figures = sum('image/png' in o.get('data', {}) for c in code for o in c.outputs)
    assert figures >= 2, 'Comparison figures were not rendered'
    links = 0
    for relative in ['README.md', 'COURSE_MAP.md', 'notebooks/README.md',
                     'notebooks/week11/README.md', 'lectures/README.md',
                     'data/week11_cavitation/README.md', 'results/week11_cavitation/README.md']:
        source = ROOT/relative
        for target in re.findall(r'\]\(([^\s)]+)\)', source.read_text(encoding='utf-8')):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            assert (source.parent/unquote(parsed.path)).exists(), (relative, target)
            links += 1
    nbformat.write(executed, path if args.retain_executed else scratch/path.name)
    report = dict(status='pass', execution_backend='fresh Jupyter kernel',
                  code_cells=len(code), figures=figures, relative_links=links,
                  retained_files_unchanged=len(before), seconds=time.perf_counter()-started,
                  optional_training='off; two-update isolation contract tested separately',
                  scope='Local CPU Run All; hosted Colab runtime not executed')
    (scratch/'execution_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
