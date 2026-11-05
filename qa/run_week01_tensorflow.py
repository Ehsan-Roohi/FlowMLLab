#!/usr/bin/env python3
"""Validate and optionally freshly execute the Week 1 TensorFlow self-study lab.

Usage: python qa/run_week01_tensorflow.py --execute
Install the course's ml and test extras in the active Python environment first.
Execution uses that same Python interpreter in a fresh kernel. This notebook
creates analytic teaching data and needs no CFD solver or external dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/week01/02_tensorflow_for_ai_fluids.ipynb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    cells = nb.cells
    source = '\n'.join(c.source for c in cells)
    assert 'Self-study revision: 2026-11-05' in source
    assert 'FLOWMLLAB_COLAB_LAUNCH_V1' in source
    assert 'FLOWMLLAB_COLAB_BOOTSTRAP_V1' in source
    assert 'notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb' in source
    assert not any('plt.bar(' in c.source for c in cells)
    if args.execute:
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        km = KernelManager(kernel_name='python3')
        km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        client = NotebookClient(nb, km=km, timeout=300,
                                resources={'metadata': {'path': str(NOTEBOOK.parent)}},
                                allow_errors=False, record_timing=True)
        started = time.perf_counter()
        client.execute()
        elapsed = time.perf_counter() - started
        code_cells = [c for c in nb.cells if c.cell_type == 'code']
        assert all(c.execution_count is not None for c in code_cells)
        errors = [o for c in code_cells for o in c.outputs if o.output_type == 'error']
        assert not errors
        images = sum('image/png' in o.get('data', {}) for c in code_cells for o in c.outputs)
        assert images >= 6, 'Expected retained physical/learning plots from the fresh run'
        NOTEBOOK.write_text(nbformat.writes(nb) + '\n', encoding='utf-8', newline='\n')
        try:
            tensorflow_version = importlib.metadata.version('tensorflow')
        except importlib.metadata.PackageNotFoundError:
            tensorflow_version = importlib.metadata.version('tensorflow-cpu')
        report = {
            'revision': '2026-11-05', 'python': platform.python_version(),
            'tensorflow': tensorflow_version,
            'numpy': importlib.metadata.version('numpy'),
            'keras': importlib.metadata.version('keras'),
            'code_cells_executed': len(code_cells), 'error_outputs': len(errors),
            'retained_plot_outputs': images, 'execution_seconds': round(elapsed, 2),
            'notebook_sha256': hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            'checks': ['tensor shapes and dtypes', 'broadcasting mismatch demonstration',
                       'analytic divergence and vorticity vs finite differences',
                       'manual chain rule vs automatic parameter gradients',
                       'one-step loss decrease and recovered linear coefficients',
                       'disjoint analytic training/validation/test locations',
                       'finite held-out error after validation-only weight selection',
                       'watched vs unwatched coordinates',
                       'analytic first and second derivatives',
                       'channel momentum residual and both wall conditions'],
            'execution_scope': 'Fresh CPU kernel; analytic teaching data; no new CFD or research training',
        }
        (ROOT / 'qa/week01_tensorflow_validation.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2))
    else:
        print(f'Notebook schema and entry points valid: {len(cells)} cells')


if __name__ == '__main__':
    main()
