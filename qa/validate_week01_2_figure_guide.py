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
from qa.build_week01_2_lecture import LECTURE_PAGES

ASSETS=ROOT/'lectures/source/week01_2_assets'
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    guide=json.loads((ASSETS/'figure_index.json').read_text(encoding='utf-8'))
    assert len(guide)==34
    assert sorted(int(v['number']) for v in guide.values() if v['lecture'])==list(range(1,22))
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
        assert len(doc.pages)==LECTURE_PAGES
        for i,page in enumerate(doc.pages,1):
            text=page.extract_text() or ''
            for number in re.findall(r'Figure\s*(\d+)\s*:',text):
                found.append(int(number));pages_by_figure[number]=i
            for c in page.chars:
                assert c['text']!='\x00' and c['x0']>=20 and c['x1']<=page.width-20,(i,c)
                assert c['top']>=15 and c['bottom']<=page.height-15,(i,c)
            links.extend(a.get('uri','') for a in page.hyperlinks)
    assert found==list(range(1,22)),found
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
    for name in protected:
        working=(ROOT/name).read_bytes()
        committed=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        # Git may check out CRLF on Windows; compare all source bytes after
        # normalizing only that transport-level line-ending difference.
        assert working.replace(b'\r\n',b'\n')==committed.replace(b'\r\n',b'\n'),name
    rows=archived_rows()  # SHA256 validation of each historical field archive.
    # Run the complete cheap path in order and save its tables. Explicitly reject
    # an accidentally enabled CFD flag before execution.
    for c in book.cells:
        if c.cell_type != 'code':continue
        for node in ast.walk(ast.parse(c.source)):
            if isinstance(node,ast.Assign):
                for target in node.targets:
                    if isinstance(target,ast.Name) and target.id.startswith('RUN_'):
                        assert isinstance(node.value,ast.Constant) and node.value.value is False,target.id
    from qa.execute_cavity_lesson import execute
    book=execute(book,ROOT)
    executed=[c.id for c in book.cells if c.cell_type=='code']
    assert len(executed)==14,len(executed)
    required=['51fefcd5','1b614a07','ee106a2c','student-review-refinement-evidence']
    for id in required:
        c=next(c for c in book.cells if c.id==id)
        assert c.execution_count and any('text/html' in o.get('data',{}) for o in c.outputs),id
    NOTE.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    for name,entry in guide.items():
        entry['lecture_page']=pages_by_figure.get(entry['number'])
        entry['png_sha256']=sha(ASSETS/entry['image'])
    (ASSETS/'figure_index.json').write_text(json.dumps(guide,indent=2)+'\n',encoding='utf-8',newline='\n')
    report={'lecture_pages':LECTURE_PAGES,'lecture_figures':21,'notebook_supplementary_figures':13,
        'all_figure_numbers_and_captions_checked':True,'all_diagrams_at_text_width_inches':6.6,
        'vector_text_within_figure_bounds':True,'all_pdf_text_within_page_bounds':True,
        'archived_fields_sha256_checked':len(rows),'solver_sources':'unchanged; fresh controls audited separately',
        'executed_cheap_cells':executed,'fresh_cfd_trajectories':0,
        'pdf_sha256':sha(pdf),'notebook_sha256':sha(NOTE),
        'hosted_colab_run':False,'visual_review':'Programmatic text/page bounds checked; manual rendered-page review is recorded separately in refinement_release_validation.json.'}
    (ROOT/'output/week01_2_figure_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    (ROOT/'results/week01_2_formulations/figure_guide_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
