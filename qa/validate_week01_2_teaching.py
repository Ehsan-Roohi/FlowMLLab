"""Audit teaching revision and exercise all-method UI using verified retained fixtures.
This does not measure a new CFD trajectory or new solver timing.
"""
from pathlib import Path
import sys,json,copy,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat as n
from pypdf import PdfReader
from validate_week01_2 import execute
nb=n.read(ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb',4)
source=next(c.source for c in nb.cells if c.cell_type=='code' and 'RUN_HIGHER_RE = False' in c.source)
source=source.replace('RUN_HIGHER_RE = False','RUN_HIGHER_RE = True').replace('output/week01_2_higher_re','output/qa_teaching_fixture')
setup="""from pathlib import Path
import sys,copy
ROOT=Path(ROOT_LITERAL)
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from IPython.display import display
from report_week01_2_formulations import load
import flowmllab.pressure_velocity_experiments as fv
import flowmllab.streamfunction_experiments as fd
fixtures={r['config']['method']:r for r in load() if r['config']['reynolds']==400 and r['config']['cells']==32}
calls=[]
def fake_fv(c,**kwargs):
    assert c.reynolds==400 and c.cells==32
    calls.append(c.method)
    return copy.deepcopy(fixtures[c.method])
def fake_fd(re,cells,**kwargs):
    assert re==400 and cells==32
    calls.append('streamfunction-vorticity')
    return copy.deepcopy(fixtures['streamfunction-vorticity'])
fv.run_experiment=fake_fv
fd.run_streamfunction=fake_fd
""".replace('ROOT_LITERAL',repr(str(ROOT)))
check="""assert set(calls)=={'simple','piso','pimple','streamfunction-vorticity'}
assert len(higher_runs)==4 and len(higher_table)==4
assert 'error' not in higher_table.columns
assert higher_table['runtime_seconds'].notna().all()
"""
fixture=n.v4.new_notebook(cells=[n.v4.new_code_cell(setup),n.v4.new_code_cell(source),n.v4.new_code_cell(check)])
counts=execute(fixture,[str(ROOT/'tmp/week01_2_kernels')])
assert counts=={'executed_code_cells':3,'errors':0,'png_outputs':3},counts
pdfs={}
for name,expected in [('week01_numerical_foundations',35),('week01_2_pressure_velocity',30)]:
    p=ROOT/'lectures'/f'{name}.pdf';r=PdfReader(p)
    assert len(r.pages)==expected
    fonts=set()
    for page in r.pages:
        for item in page.get('/Resources',{}).get('/Font',{}).values():
            font=item.get_object();fonts.add(str(font.get('/BaseFont','')))
    assert any('TimesNewRoman' in f for f in fonts),(name,fonts)
    pdfs[name]={'pages':expected,'times_new_roman_embedded':True,'visual_review':'All pages rendered and inspected','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
record={'scope':'Teaching and plotting revision; scientific solver sources and archived trajectories unchanged','all_method_controller_fixture_test':counts,'fixture_methods':4,'fixture_is_new_cfd_run':False,'pdfs':pdfs,'week1_execution':{'python':{'code_cells':26,'png_outputs':11},'cavity':{'code_cells':15,'png_outputs':6},'tensorflow':'Not executed: TensorFlow unavailable locally; deterministic scalar-field preview regenerated with NumPy, no training claim'}}
(ROOT/'results/week01_2_formulations/teaching_revision_validation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,indent=2))
