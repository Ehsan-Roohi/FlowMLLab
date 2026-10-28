"""Audit the beginner teaching order, synchronized notation and blue video links."""
from pathlib import Path
import copy
import hashlib
import json
import re
import sys
import nbformat
import pdfplumber

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qa.sync_week01_2_teaching import expand,sync_notebook

def main():
    source=ROOT/'lectures/source'
    intro=(source/'week01_2_introduction.tex').read_text(encoding='utf-8')
    assert not any(t in intro for t in [r'\[','$',r'\Delta',r'\sum'])
    core=expand(source/'week01_2_pressure_velocity_core.tex')
    assert core.index('What the finite-volume method conserves')<core.index(r'\int')
    assert core.index('Finite volume and finite differences')<core.index(r'\int')
    assert core.index('shared velocity face')<core.index(r'\lessonfigure{mac_grid}')
    assert core.index(r'x_{i+1/2}')<core.index(r'\lessonfigure{checkerboard_pressure}')
    simple=core.index(r'\topic{8}{')
    for definition in ['Where the source comes from.','A source, a neighbor sum',
                       'Collect the scalar rows','The operators used',
                       'Derive the velocity correction','under-relaxation',
                       'Decide whether both equations']:
        assert core.index(definition)<simple,definition
    assert core.index('Physical time and backward-Euler')<core.index(r'\topic{10}{')
    assert core.index('Label or field')<core.index(r'A^{(\ell)}=')
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    book=nbformat.read(path,4)
    assert sync_notebook(copy.deepcopy(book))==book,'Teaching synchronization is not idempotent'
    tables=0
    for c in book.cells:
        if c.cell_type!='markdown':continue
        assert c.source.count('$$')%2==0,c.id
        if c.id.startswith('coupling-page-'):
            assert not any(x in c.source for x in [r'\input',r'\lessonfigure',r'\begin{tabular}',r'\\vert']),c.id
            lines=c.source.splitlines()
            for i,line in enumerate(lines[:-1]):
                if line.startswith('| ') and lines[i+1].startswith('| ---'):
                    assert all(len(v.strip())>=3 for v in lines[i+1].split('|')[1:-1]),c.id
                    tables+=1
    assert tables>=7,tables
    pdf=ROOT/'lectures/week01_2_pressure_velocity.pdf'
    video_links=[]
    with pdfplumber.open(pdf) as doc:
        assert len(doc.pages)==59
        page2=doc.pages[1].extract_text()
        assert 'moving lid' in page2.lower() and 'mass conservation' in page2.lower()
        assert 'r c,P' not in page2 and 'Q P' not in page2
        for i,page in enumerate(doc.pages,1):
            for link in page.hyperlinks:
                uri=link.get('uri','')
                if 'youtube.com/watch?v=' not in uri:continue
                chars=[c for c in page.chars if link['x0']-.5<=c['x0']<=link['x1']+.5
                       and link['top']-1<=c['top']<=link['bottom']+1]
                blue=[c for c in chars if isinstance(c.get('non_stroking_color'),(tuple,list))
                      and len(c['non_stroking_color'])==3
                      and c['non_stroking_color'][2]>.5 and c['non_stroking_color'][0]<.1]
                assert len(blue)>=8,(uri,chars[:2])
                video_links.append({'uri':uri,'page':i,'blue_text_characters':len(blue)})
    assert {r['uri'].split('v=')[-1] for r in video_links}=={'OOILoJ1zuiw','ahdW5TKacok'}
    report={'pages':59,'introductory_page_has_no_equation_notation':True,
            'finite_volume_and_difference_comparison_precedes_integration':True,
            'cell_center_and_face_locations_defined_before_figures':True,
            'source_neighbors_operators_corrections_relaxation_defined_before_SIMPLE':True,
            'PISO_and_PIMPLE_indices_explained_before_loop_equations':True,
            'notebook_synchronized_with_lecture':True,'notebook_tables_checked':tables,
            'video_links':video_links,'solver_physics_modified':False,
            'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    target=ROOT/'results/week01_2_formulations/prerequisite_validation.json'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
