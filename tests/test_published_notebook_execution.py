"""Synthetic kernel stubs check source preservation, never claim a Jupyter run."""
import importlib.util
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('published_notebook_ci', ROOT/'qa/execute_published_notebooks.py')
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def notebook(root, relative='notebooks/demo.ipynb', source='answer = 42'):
    path = root/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    cell = nbformat.v4.new_code_cell(source, execution_count=77,
        outputs=[nbformat.v4.new_output('stream',name='stdout',text='retained output\n')])
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('Current lesson prose.'),cell])
    nbformat.write(book,path)
    return path


def snapshot_stub(root, paths):
    return lambda ignored: {p.relative_to(root).as_posix():ci.digest(p) for p in paths}


def test_week7_retained_profile_preserves_file_and_every_other_statement(tmp_path):
    code='RUN_LIVE_CONTINUATION = True\nassert 3 > 2\n'
    path=notebook(tmp_path,ci.WEEK7,code); original=path.read_bytes()
    book,record=ci.prepare_notebook(path,tmp_path.resolve(),'week7-retained')
    assert path.read_bytes()==original
    assert book.cells[2].source==code.replace('RUN_LIVE_CONTINUATION = True','RUN_LIVE_CONTINUATION = False')
    assert book.cells[1].source=='Current lesson prose.'
    assert record['in_memory_overrides'][0]['published_value'] is True
    assert record['in_memory_overrides'][0]['execution_value'] is False
    published,_=ci.prepare_notebook(path,tmp_path.resolve(),'published')
    assert published.cells[2].source==code


def test_omitted_code_cell_cannot_pass_with_saved_execution_numbers(tmp_path,monkeypatch):
    path=notebook(tmp_path)
    monkeypatch.setattr(ci,'tracked_snapshot',snapshot_stub(tmp_path,[path]))
    class PartialKernelStub:
        def __init__(self,book,**kwargs):self.book=book
        def execute(self):self.book.cells[0].execution_count=1
    monkeypatch.setattr(ci,'NotebookClient',PartialKernelStub)
    with pytest.raises(ValueError,match='Every nonempty published code cell'):
        ci.execute([path],root=tmp_path)


def test_source_data_guard_also_runs_when_execution_raises(tmp_path,monkeypatch):
    path=notebook(tmp_path); data=tmp_path/'retained.dat';data.write_bytes(b'original')
    monkeypatch.setattr(ci,'tracked_snapshot',snapshot_stub(tmp_path,[path,data]))
    class MutatingKernelStub:
        def __init__(self,book,**kwargs):pass
        def execute(self):
            data.write_bytes(b'changed')
            raise RuntimeError('Synthetic execution failure')
    monkeypatch.setattr(ci,'NotebookClient',MutatingKernelStub)
    with pytest.raises(ValueError,match='changed tracked teaching source/data: retained.dat'):
        ci.execute([path],root=tmp_path)


def test_success_does_not_overwrite_published_outputs_or_prose(tmp_path,monkeypatch):
    path=notebook(tmp_path);original=path.read_bytes()
    monkeypatch.setattr(ci,'tracked_snapshot',snapshot_stub(tmp_path,[path]))
    class CompleteKernelStub:
        def __init__(self,book,**kwargs):self.book=book
        def execute(self):
            count=0
            for cell in self.book.cells:
                if cell.cell_type=='code' and cell.source.strip():
                    count+=1;cell.execution_count=count
    monkeypatch.setattr(ci,'NotebookClient',CompleteKernelStub)
    report=ci.execute([path],root=tmp_path)
    assert path.read_bytes()==original
    assert report['notebooks'][0]['completed_published_code_cells']==1
    assert report['notebooks'][0]['sha256']==ci.digest(path)
    assert report['tracked_source_data_unchanged'] is True
