"""Execute Week 1.2's notebook and fresh-run branch; audit retained evidence.

Run after qa/run_week01_2.py and qa/build_week01_2.py. Uses the current
Python interpreter as the Jupyter kernel, rather than an unrelated system one.
Requires the project's test extras and pdfplumber. PDF visual review is separate.
"""
from pathlib import Path
import copy,hashlib,json,os,subprocess,sys
import numpy as np
import scipy
import nbformat
import pdfplumber
from pypdf import PdfReader
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
OUT=ROOT/'results/week01_2_pressure_velocity'
sys.path.insert(0,str(ROOT))
from flowmllab.pressure_velocity_lab import load_results


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(nb,kernel_dirs):
    manager=KernelManager(kernel_name='week01_2',kernel_spec_manager=KernelSpecManager(kernel_dirs=kernel_dirs))
    NotebookClient(nb,km=manager,timeout=600,allow_errors=False,
                   resources={'metadata':{'path':str(NOTE.parent)}}).execute()
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    outputs=[o for c in codes for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    return {'executed_code_cells':len(codes),'errors':0,
            'png_outputs':sum('image/png' in o.get('data',{}) for o in outputs)}


def main():
    kernels=ROOT/'tmp/week01_2_kernels';kernel=kernels/'week01_2';kernel.mkdir(parents=True,exist_ok=True)
    env={'PYTHONPATH':os.environ.get('PYTHONPATH',''),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 1.2 audit','language':'python','env':env}))
    manifest,rows=load_results(OUT)
    assert digest(ROOT/'flowmllab/pressure_velocity.py')==manifest['source_sha256']
    assert digest(ROOT/'common/w4utils.py')==manifest['reference_source_sha256']
    assert len(rows)==8 and all(r['converged'] for r in rows)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_pressure_velocity.py','-v'],
        cwd=ROOT,capture_output=True,text=True,env=os.environ.copy())
    (ROOT/'tmp/week01_2_tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    print('Five conservation/coupling tests passed.',flush=True)
    nb=nbformat.read(NOTE,as_version=4)
    cached=execute(nb,[str(kernels)])
    assert cached['png_outputs']==6
    nbformat.write(nb,NOTE)
    NOTE.write_bytes(NOTE.read_bytes().replace(b'\r\n',b'\n'))
    print('Default notebook executed:',cached,flush=True)
    fresh=copy.deepcopy(nb)
    cell=next(c for c in fresh.cells if c.cell_type=='code' and 'RUN_NEW = False' in c.source)
    cell.source=cell.source.replace('RUN_NEW = False','RUN_NEW = True')
    cell.source+='\nassert len(fresh)==3 and all(r["converged"] for r in fresh)\n'
    check=execute(fresh,[str(kernels)])
    assert check['png_outputs']==12
    nbformat.write(fresh,ROOT/'tmp/week01_2_fresh_audit.ipynb')
    print('Fresh SIMPLE/PISO/PIMPLE 32-cell branch executed:',check,flush=True)
    pdf=ROOT/'lectures/week01_2_pressure_velocity.pdf'
    reader=PdfReader(pdf);assert len(reader.pages)==11
    fonts=set()
    for page in reader.pages:
        for f in page['/Resources']['/Font'].values():
            obj=f.get_object();fonts.add(str(obj.get('/BaseFont','')))
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            assert not any(c['text']=='\x00' for c in page.chars)
            for c in page.chars:
                assert 20<c['x0']<page.width-20 and c['x1']<page.width-20
                assert 10<c['top']<page.height-10 and c['bottom']<page.height-10
    baseline=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    preserved={}
    for name in ['notebooks/week01/03_cavity_ghia.ipynb','common/w4utils.py']:
        original=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        # Git's LF representation may differ from the local checkout's CRLF.
        assert original.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n')
        preserved[name]=hashlib.sha256(original).hexdigest()
    validation={'date':'2026-10-09','baseline_commit':baseline,
        'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,
        'unit_tests':{'count':5,'failures':0,'command':'python -m unittest discover -s tests -p test_pressure_velocity.py -v'},
        'recorded_runs':8,'field_checksums_verified':True,'source_checksums_verified':True,
        'default_notebook':cached,'fresh_all_three_n32_notebook':check,
        'notebook_sha256':digest(NOTE),'pdf':{'pages':11,'sha256':digest(pdf),'fonts':sorted(fonts),
            'text_within_page_bounds':True},
        'preserved_week1_baseline_sha256':preserved,
        'qualification':'Local CPU notebook execution; hosted Colab execution was not tested. Re=100 steady results only. FV benchmark refinement trend is nonmonotone.'}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Evidence and PDF structural checks passed.',flush=True)


if __name__=='__main__':main()
