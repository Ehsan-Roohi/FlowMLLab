"""Small post-execution prose corrections; does not fabricate code outputs."""
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parents[1]
paths=list((ROOT/'notebooks').glob('**/*.ipynb'))
for p in paths:
    n=nbformat.read(p,4)
    if n.metadata.get('classroom_revision')!=1: continue
    for c in n.cells:
        if c.cell_type!='markdown': continue
        if 'maximum absolute error of 1.78' in c.source and 'np.clip' not in c.source:
            c.source+='\n\nImplementation detail: `predict_surrogate` applies `np.clip(raw_prediction, -1, 1)` before returning values. The reported physical integrals therefore use clipped predictions; the raw MLP itself has an unbounded linear head. The printed range audits returned values, not the raw head. An optimizer iteration-limit warning means optimization stopped at its budget, not that convergence was demonstrated.'
        if 'Build and train a small rectangular-cavity' in c.source and '200-step' not in c.source:
            c.source+='\n\nThe 200-step default is budget-limited, not a converged CFD solution. Exact wall and continuity checks validate the representation, not momentum accuracy. Inspect the held-out momentum residual before interpreting its field as a physical solution.'
    nbformat.write(n,p)
