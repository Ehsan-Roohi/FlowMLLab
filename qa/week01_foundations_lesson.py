"""Keep the explicit Week 1 derivations synchronized with the TeX lesson.
Only Markdown cells change; solver code and executed outputs are retained.
"""
from pathlib import Path
import re
import nbformat as n
ROOT=Path(__file__).resolve().parents[1]


def markdown(tex):
    tex=re.sub(r'^%.*$', '', tex, flags=re.M)
    tex=re.sub(r'\\topic\{([^}]*)\}\{([^}]*)\}',lambda m:'### '+m[1]+': '+m[2],tex)
    tex=re.sub(r'\\label\{[^}]*\}', '', tex)
    tex=re.sub(r'\\eqref\{[^}]*\}', 'the derived equation', tex)
    tex=re.sub(r'\\textbf\{([^{}]*)\}',r'**\1**',tex)
    tex=re.sub(r'\\emph\{([^{}]*)\}',r'*\1*',tex)
    tex=re.sub(r'\\texttt\{([^{}]*)\}',r'`\1`',tex)
    tex=re.sub(r'\\checkpoint\{([^{}]*)\}',r'**Check your understanding.** \1',tex)
    tex=re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}',
        '```text\n             N = f[j+1,i]\n                  |\nW = f[j,i-1] -- C = f[j,i] -- E = f[j,i+1]\n                  |\n             S = f[j-1,i]\n```',tex,flags=re.S)
    def table(m):
        body=m[1].replace('\\toprule','').replace('\\midrule','').replace('\\bottomrule','')
        body=re.sub(r'`([^`]*)`',lambda z:z[1].replace('\\_','_'),body)
        rows=[r.strip().replace('\\hline','').strip() for r in body.split('\\\\') if r.strip()]
        if not rows:return ''
        cells=[[c.strip().replace('\\texttt','') for c in row.split('&')] for row in rows]
        return '\n\n'+'\n'.join('| '+' | '.join(row)+' |' for row in [cells[0],['---']*len(cells[0]),*cells[1:]])+'\n\n'
    tex=re.sub(r'\\begin\{tabular\}\{[^\n]*?\}\s*(.*?)\\end\{tabular\}',table,tex,flags=re.S)
    tex=tex.replace('\\begin{center}','').replace('\\end{center}','')
    tex=tex.replace('\\begin{align*}','$$\n\\begin{aligned}').replace('\\end{align*}','\\end{aligned}\n$$')
    for kind in ['equation*','equation']:
        tex=tex.replace('\\begin{'+kind+'}','$$').replace('\\end{'+kind+'}','$$')
    tex=tex.replace('\\[','$$').replace('\\]','$$')
    tex=tex.replace('~',' ').replace('--','-').replace('\\_','_')
    return tex.strip()


def lesson_cells(tex,group):
    cells=[]
    for i,part in enumerate(tex.split('\\newpage')):
        if not part.strip():continue
        cell=n.v4.new_markdown_cell(markdown(part))
        cell.id=f'week1-{group}-{i}'
        cell.metadata['explicit_derivation']=group
        cells.append(cell)
    return cells


def update_notebook(path=None):
    path=Path(path or ROOT/'notebooks/week01/03_cavity_ghia.ipynb')
    nb=n.read(path,4)
    nb.cells=[c for c in nb.cells if not c.metadata.get('explicit_derivation')]
    transport=(ROOT/'lectures/source/week01_transport_derivation.tex').read_text(encoding='utf-8')
    finite=(ROOT/'lectures/source/week01_finite_differences.tex').read_text(encoding='utf-8')
    finite=finite[finite.index('\\topic{7.2}'):]
    governing=next(c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith('## 2. From '))
    governing.source=r"""## 2. From momentum and continuity to streamfunction and vorticity

The dimensional conservation equations are stated above. Scale position by cavity side $L$, velocity by lid speed $U$, time by $L/U$, and pressure difference by $\rho U^2$. The following derivation uses dimensionless variables and $Re=UL/\nu$. All derivatives are written explicitly.

For a smooth two-dimensional incompressible field in this simply connected cavity, define the streamfunction by

$$u=\frac{\partial\psi}{\partial y},\qquad v=-\frac{\partial\psi}{\partial x}.$$

Substituting these definitions into continuity gives zero because smooth mixed derivatives commute. Vorticity is $\omega=\partial v/\partial x-\partial u/\partial y$. Substitution of the streamfunction definitions gives

$$\omega=-\frac{\partial^2\psi}{\partial x^2}-\frac{\partial^2\psi}{\partial y^2}.$$

Thus $\nabla^2\psi=-\omega$. The next four cells derive the vorticity evolution equation and distinguish it from the pressure derivation.
"""
    idx=nb.cells.index(governing)+1
    nb.cells[idx:idx]=lesson_cells(transport,'transport')
    idx=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 6. Solve'))
    nb.cells[idx+1:idx+1]=lesson_cells(finite,'poisson')
    n.validate(nb);n.write(nb,path)
    path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    return nb

if __name__=='__main__':update_notebook()
