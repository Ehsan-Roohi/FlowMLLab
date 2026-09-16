"""Add an independently validated inverse problem before the W14 research audit."""
from pathlib import Path
import sys
import nbformat
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qa'))
from classroom_cells import inverse
path=ROOT/'notebooks/week14/W14_pyCALC_RANS_PINN_NN.ipynb'
nb=nbformat.read(path,4)
if nb.metadata.get('classroom_revision')!=1:
    nb.cells[2:2]=inverse()
    nb.metadata['classroom_revision']=1
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[]; cell.execution_count=None
    nbformat.write(nb,path)
for c in nb.cells:
    if c.cell_type=='code' and "display(pd.DataFrame(summary['profile_metrics']))" in c.source:
        c.source=c.source.replace("display(pd.DataFrame(summary['profile_metrics']))", "profiles=pd.DataFrame(summary['profile_metrics'])\n    display(profiles[~profiles['case'].isin(['baseline fresh restart','pinn fresh restart'])])\n    print('Identical restart rows are omitted: consistency checks, not independent validation.')")
        c.outputs=[]; c.execution_count=None
nbformat.write(nb,path)
