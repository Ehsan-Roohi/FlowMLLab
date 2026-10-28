"""Synchronize the complete, ordered Week 1.2 teaching text without rerunning CFD."""
from pathlib import Path
import json
import re
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'lectures/source'
NOTE=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
URL='https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/'

RELAXATION_CODE='''# One frozen momentum row: use unrounded coefficients in the calculation.
import numpy as np
row_a = 0.64
row_neighbors = np.array([0.12, 0.20, 0.16, 0.16]) @ np.array([0.04, 0.02, 0.03, 0.01])
row_source = 0.0
row_pressure_gradient = (0.008 - 0.010) / 0.25
row_unrelaxed = (row_neighbors + row_source - row_pressure_gradient) / row_a
row_alpha = 0.7
row_old_velocity = 0.02
row_relaxed_a = row_a / row_alpha
row_relaxed_source = row_source + (1 / row_alpha - 1) * row_a * row_old_velocity
row_predictor = (row_neighbors + row_relaxed_source - row_pressure_gradient) / row_relaxed_a
print(f"Neighbor contributions: {row_neighbors:.6f}")
print(f"Original source b: {row_source:.6f}; relaxed source: {row_relaxed_source:.9f}")
print(f"Unrelaxed row velocity: {row_unrelaxed:.6f}; relaxed predictor: {row_predictor:.6f}")
assert np.isclose(row_unrelaxed, 0.03625)
assert np.isclose(row_predictor, 0.031375)
assert np.isclose(row_predictor, row_alpha * row_unrelaxed + (1-row_alpha) * row_old_velocity)
# Separate wall-face example: the distance to a prescribed lid is h/2.
wall_weight = 2 / (100 * 0.25**2)
wall_source = wall_weight * 1.0
print(f"Separate lid-face contribution: diagonal={wall_weight:.6f}, source={wall_source:.6f}")
assert np.isclose(wall_weight, 0.32) and np.isclose(wall_source, 0.32)
'''

def expand(path):
    s=path.read_text(encoding='utf-8')
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(path.parent/m.group(1)),s)

def group_end(text,start):
    assert text[start]=='{'
    depth=0
    for i in range(start,len(text)):
        if i and text[i-1]=='\\':continue
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return i+1
    raise ValueError('Unclosed TeX group')

def tables(text):
    begin=r'\begin{tabular}'
    while begin in text:
        a=text.index(begin);start=a+len(begin)
        while text[start].isspace():start+=1
        end=group_end(text,start)
        stop=text.index(r'\end{tabular}',end)
        body=text[end:stop]
        body=re.sub(r'\\(?:toprule|midrule|bottomrule|hline)','',body)
        rows=[r.strip() for r in body.split('\\\\') if r.strip()]
        cells=[[x.strip().replace(r'\|',r'\Vert').replace('|',r'\vert') for x in r.split('&')] for r in rows]
        assert len(set(map(len,cells)))==1,cells
        md='\n\n'+'\n'.join('| '+' | '.join(row)+' |' for row in [cells[0],['---']*len(cells[0]),*cells[1:]])+'\n\n'
        text=text[:a]+md+text[stop+len(r'\end{tabular}'):]
    return text

def markdown(text,guide):
    # Every figure occupies one full Markdown block and shares its PDF number.
    def picture(m):
        name=m.group(1);entry=guide[name]
        return '\n\n!['+name.replace('_',' ')+']('+URL+entry['image']+')\n\n**Figure '+entry['number']+'.** '+entry['caption']+'\n\n'
    text=re.sub(r'\\lessonfigure\{([^}]+)\}\{[^\n]*\}\{[^}]+\}',picture,text)
    text=tables(text)
    # Tables have already been parsed with balanced column-specification braces.
    from qa.week01_foundations_lesson import markdown as convert
    text=convert(text).replace(r'\small','')
    # The legacy converter normalizes prose dashes. Restore GFM table rules.
    text=re.sub(r'(?m)^\|[\s|:-]+\|$',
        lambda m:'| '+' | '.join('---' for c in m.group().split('|')[1:-1])+' |',text)
    text=text.replace(r'\begin{enumerate}','').replace(r'\end{enumerate}','')
    text=text.replace(r'\begin{itemize}','').replace(r'\end{itemize}','')
    text=re.sub(r'\\item\s*','\n- ',text)
    for token in [r'\topic',r'\lessonfigure',r'\input',r'\begin{tabular}',r'\raggedright']:
        assert token not in text,token
    assert text.count('$$')%2==0
    return text.strip()

def sync_notebook(book):
    guide=json.loads((SOURCE/'week01_2_assets/figure_index.json').read_text(encoding='utf-8'))
    old={c.id:c for c in book.cells}
    source=expand(SOURCE/'week01_2_introduction.tex')+'\n'+r'\newpage'+'\n'+expand(SOURCE/'week01_2_pressure_velocity_core.tex')
    # Preserve the exact recorded-data/opt-in-solver tail and its executed tables.
    evidence=next(i for i,c in enumerate(book.cells) if c.id=='pressure-recorded-evidence')
    tail=book.cells[evidence:]
    lessons=[];algebra_done=False;matrix_done=False
    for i,part in enumerate(source.split(r'\newpage'),1):
        if not part.strip():continue
        cell=nbformat.v4.new_markdown_cell(markdown(part,guide),id=f'coupling-page-{i:02d}')
        cell.metadata['pressure_coupling_lesson']=True
        lessons.append(cell)
        if 'Read the two pressure-gradient formulas' in part:lessons.append(old['visual-checkerboard-check'])
        if 'Crossing a face while carrying horizontal momentum' in part:lessons.append(old['visual-hat-arithmetic'])
        if 'Boundary cells and pressure reference' in part:
            lessons.append(old['13faeb86']);matrix_done=True
        if 'Prepare a steady iteration: under-relaxation' in part:
            c=nbformat.v4.new_code_cell(RELAXATION_CODE,id='prereq-source-relaxation')
            if c.id in old and old[c.id].source==c.source:c=old[c.id]
            lessons.append(c)
        if 'PIMPLE: a nonlinear coefficient example' in part:
            lessons.append(old['pressure-algebra-examples']);algebra_done=True
    assert algebra_done and matrix_done
    # Retained review explanations stay after the concepts they use.
    for id in ['student-review-coefficient-normalization','student-review-old-time-level']:
        if id in old:
            after='Physical time and backward-Euler momentum'
            index=next(i for i,c in enumerate(lessons) if c.cell_type=='markdown' and after in c.source)+1
            lessons.insert(index,old[id])
    from qa.week01_2_equation_lesson import REFERENCES
    refs=nbformat.v4.new_markdown_cell(REFERENCES,id='coupling-reading-references')
    refs.metadata['pressure_coupling_lesson']=True;lessons.append(refs)
    book.cells=[old['829339c3'],old['ed71508c'],*lessons,*tail]
    nbformat.validate(book)
    return book

def main():
    book=sync_notebook(nbformat.read(NOTE,4))
    NOTE.write_text(nbformat.writes(book),encoding='utf-8',newline='\n')
    print('Synchronized teaching cells:',sum(c.metadata.get('pressure_coupling_lesson',False) for c in book.cells))

if __name__=='__main__':main()
