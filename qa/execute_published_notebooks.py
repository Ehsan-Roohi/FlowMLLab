"""Execute canonical published notebooks without replacing their saved contents.

Legacy make_week*_notebook.py scripts do not preserve every later addition and
are not the source of truth for these CI checks. This helper executes a copy in
a fresh Jupyter kernel, retains every assertion, and byte-checks tracked teaching
source/data before and after. The explicit Week-7 retained profile disables only
the optional live CFD continuation; it is not a claim to test that continuation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
WEEK7 = 'notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb'
PROTECTED = ('notebooks', 'data', 'results', 'flowmllab', 'qa', 'lectures')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tracked_snapshot(root):
    names = subprocess.check_output(
        ['git', '-c', f'safe.directory={root.as_posix()}', 'ls-files', '-z', '--', *PROTECTED],
        cwd=root,
    ).decode('utf-8').split('\0')
    result = {}
    for name in filter(None, names):
        path = root / name
        require(path.resolve().is_relative_to(root) and path.is_file(),
                'Tracked teaching source/data is missing or leaves the checkout: ' + name)
        result[name] = digest(path)
    require(result, 'No tracked teaching source/data was found')
    return result


def require_unchanged(before, after):
    changed = [name for name in sorted(before.keys() | after.keys())
               if before.get(name) != after.get(name)]
    require(not changed, 'Notebook execution changed tracked teaching source/data: ' + ', '.join(changed[:20]))


def prepare_notebook(path, root, profile):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path.suffix == '.ipynb', 'Notebook must belong to this checkout')
    original_sha = digest(path)
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    code = [cell for cell in book.cells if cell.cell_type == 'code' and cell.source.strip()]
    require(code, 'Published notebook has no executable cells')
    overrides = []
    if profile == 'week7-retained':
        require(path.relative_to(root).as_posix() == WEEK7,
                'The Week-7 retained profile is valid only for the cylinder notebook')
        matches = []
        for cell in code:
            found = list(re.finditer(r'^RUN_LIVE_CONTINUATION = (True|False)$', cell.source, re.M))
            matches.extend((cell, match) for match in found)
        require(len(matches) == 1, 'Expected exactly one explicit optional Week-7 continuation switch')
        cell, match = matches[0]
        old = match.group(1)
        cell.source = cell.source[:match.start(1)] + 'False' + cell.source[match.end(1):]
        overrides.append(dict(cell_id=cell.id, variable='RUN_LIVE_CONTINUATION',
                              published_value=old == 'True', execution_value=False,
                              scope='Retained-evidence CI only; optional live continuation is not executed'))
    for cell in book.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    book.cells.insert(0, nbformat.v4.new_code_cell("get_ipython().run_line_magic('matplotlib', 'inline')"))
    return book, dict(path=path.relative_to(root).as_posix(), sha256=original_sha,
                      published_code_cells=len(code), execution_profile=profile,
                      in_memory_overrides=overrides, initialization_cells=1)


def execute(paths, profile='published', timeout=600, root=ROOT):
    root = Path(root).resolve()
    paths = [Path(path).resolve() for path in paths]
    require(len(paths) == len(set(paths)), 'Duplicate notebook execution was requested')
    before = tracked_snapshot(root)
    records = []
    try:
        for path in paths:
            book, record = prepare_notebook(path, root, profile)
            NotebookClient(book, timeout=timeout, kernel_name='python3', allow_errors=False,
                           resources={'metadata': {'path': str(path.parent)}}).execute()
            executed = [cell for cell in book.cells if cell.cell_type == 'code' and cell.source.strip()]
            require([cell.execution_count for cell in executed] == list(range(1, len(executed) + 1)),
                    'Every nonempty published code cell must execute in order')
            require(len(executed) == record['published_code_cells'] + 1, 'Published cell coverage changed')
            errors = [output for cell in executed for output in cell.get('outputs', [])
                      if output.output_type == 'error']
            require(not errors, 'Notebook retained execution errors')
            require(digest(path) == record['sha256'], 'Published notebook bytes changed')
            record.update(completed_published_code_cells=record['published_code_cells'],
                          source_notebook_unchanged=True, jupyter_kernel_executed=True)
            records.append(record)
    finally:
        require_unchanged(before, tracked_snapshot(root))
    return dict(status='passed', execution_method='Fresh real Jupyter kernel for each in-memory notebook copy',
                notebooks=records, tracked_source_data_files_checked=len(before),
                tracked_source_data_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                tracked_source_data_unchanged=True,
                scope='Current published source and retained evidence; no published notebook or scientific artifact is replaced')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', nargs='+', type=Path)
    parser.add_argument('--profile', choices=('published', 'week7-retained'), default='published')
    parser.add_argument('--timeout', type=int, default=600, help='Per-cell timeout in seconds')
    args = parser.parse_args()
    require(args.timeout > 0, 'Per-cell timeout must be positive')
    print(json.dumps(execute(args.notebooks, args.profile, args.timeout), indent=2))


if __name__ == '__main__':
    main()
