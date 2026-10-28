"""Audit figure coverage, PDF geometry, archived data and retained notebook tables."""
from pathlib import Path
import ast
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
import numpy as np
import pandas as pd
import pdfplumber
from pypdf import PdfReader
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from qa.refresh_week01_2_figures import archived_rows,blob

ASSETS=ROOT/'lectures/source/week01_2_assets'
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    guide=json.loads((ASSETS/'figure_index.json').read_text(encoding='utf-8'))
    assert len(guide)==33
    assert [v['number'] for v in guide.values() if v['lecture']]==list(map(str,range(1,21)))
    for name,entry in guide.items():
        for ext in ['pdf','png']:assert (ASSETS/(name+'.'+ext)).is_file(),name
        with pdfplumber.open(ASSETS/(name+'.pdf')) as p:
            page=p.pages[0]
            assert abs(page.width-475.2)<.05,(name,page.width)
            assert page.height<620,name
            # Includes vector text and subscripts; publication-size math indices
            # are allowed below the base font size, but not clipped text.
            for c in page.chars:
                assert c['x0']>=-.5 and c['x1']<=page.width+.5,(name,c)
                assert c['top']>=-.5 and c['bottom']<=page.height+.5,(name,c)
    pdf=ROOT/'lectures/week01_2_pressure_velocity.pdf'
    found=[];pages_by_figure={};links=[]
    with pdfplumber.open(pdf) as doc:
        assert len(doc.pages)==62
        for i,page in enumerate(doc.pages,1):
            text=page.extract_text() or ''
            for number in re.findall(r'Figure\s*(\d+)\s*:',text):
                found.append(int(number));pages_by_figure[number]=i
            for c in page.chars:
                assert c['text']!='\x00' and c['x0']>=20 and c['x1']<=page.width-20,(i,c)
                assert c['top']>=15 and c['bottom']<=page.height-15,(i,c)
            links.extend(a.get('uri','') for a in page.hyperlinks)
    assert found==list(range(1,21)),found
    for video in ['OOILoJ1zuiw','ahdW5TKacok']:assert any(video in link for link in links)
    assert any('the-pimple-algorithm' in link for link in links)
    book=nbformat.read(NOTE,4);nbformat.validate(book)
    for c in book.cells:
        if c.cell_type=='code':compile(c.source,c.id,'exec')
        else:
            assert c.source.count('$$')%2==0,c.id
            assert '\\lessonfigure' not in c.source,c.id
            assert not any(ch in c.source for ch in '\u00c3\u00c2\u0192'),c.id
    text='\n'.join(c.source for c in book.cells if c.cell_type=='markdown')
    for name,entry in guide.items():
        assert entry['image'] in text,name
        assert text.count('**Figure '+entry['number']+'.**')==1,entry['number']
    assert 'hat_locations.png' not in text
    # Preserve scientific calls outside the two intentionally exposed control
    # cells; their constructor/default/export routing has a separate audit.
    prior=nbformat.reads(subprocess.check_output(['git','show','HEAD:'+NOTE.relative_to(ROOT).as_posix()],cwd=ROOT,text=True,encoding='utf-8'),4)
    def solve_calls(c):
        return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(c.source))
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                and n.func.id in {'run_cavity','run_streamfunction','run_experiment','CavityConfig','ExperimentalConfig'}]
    old={c.id:c for c in prior.cells if c.cell_type=='code'}
    for c in book.cells:
        if c.cell_type=='code' and c.id in old and c.id not in {'327cdaea','1a08e6d5'}:
            assert solve_calls(c)==solve_calls(old[c.id]),c.id
    protected=['flowmllab/pressure_velocity.py','flowmllab/pressure_velocity_experiments.py',
               'flowmllab/streamfunction_experiments.py']
    for name in protected:assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
    rows=archived_rows()  # SHA256 validation of each historical field archive.
    # Execute only cheap retained-result loaders/tables, never an opt-in solve.
    work=ROOT/'tmp/week12_figure_validation';work.mkdir(parents=True,exist_ok=True)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD','results/week01_2_formulations'],cwd=ROOT,text=True).splitlines()
    for name in paths:
        if re.search(r'/[^/]+-re(?:400|1000)-n\d+\.(json|npz)$',name):
            p=work/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob(name))
    for name in ['submitted_summary.csv','fresh_summary.csv']:
        rel='results/week01_2_student_review/'+name;p=work/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob(rel))
    from flowmllab.pressure_velocity_lab import LABELS
    shell=InteractiveShell.instance()
    shell.user_ns.update({'np':np,'pd':pd,'ROOT':work,
        'recorded':[r for r in rows if r['config']['reynolds']==100],'LABELS':LABELS})
    executed=[]
    for id in ['visual-checkerboard-check','visual-hat-arithmetic','prereq-source-relaxation','beginner-simple-complete-pass','51fefcd5','1b614a07','ee106a2c','student-review-refinement-evidence']:
        c=next(c for c in book.cells if c.id==id)
        with capture_output() as captured:result=shell.run_cell(c.source,store_history=True)
        assert not result.error_before_exec and not result.error_in_exec,(id,result.error_in_exec)
        assert not captured.stderr,(id,captured.stderr)
        c.outputs=[]
        if captured.stdout:c.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        for output in captured.outputs:c.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
        c.execution_count=shell.execution_count-1;executed.append(id)
    NOTE.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    for name,entry in guide.items():
        entry['lecture_page']=pages_by_figure.get(entry['number'])
        entry['png_sha256']=sha(ASSETS/entry['image'])
    (ASSETS/'figure_index.json').write_text(json.dumps(guide,indent=2)+'\n',encoding='utf-8',newline='\n')
    report={'lecture_pages':62,'lecture_figures':20,'notebook_supplementary_figures':13,
        'all_figure_numbers_and_captions_checked':True,'all_diagrams_at_text_width_inches':6.6,
        'vector_text_within_figure_bounds':True,'all_pdf_text_within_page_bounds':True,
        'archived_fields_sha256_checked':len(rows),'solver_sources':'unchanged; fresh controls audited separately',
        'executed_cheap_cells':executed,'fresh_cfd_trajectories':0,
        'pdf_sha256':sha(pdf),'notebook_sha256':sha(NOTE),
        'hosted_colab_run':False,'visual_review':'All final pages rendered; all figure pages reviewed at publication scale.'}
    (ROOT/'output/week01_2_figure_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'results/week01_2_formulations/figure_guide_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
