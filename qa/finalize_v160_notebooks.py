from pathlib import Path
import nbformat
R=Path(__file__).resolve().parents[1]
for relative in ['notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb','notebooks/week15/W15_Geometry_Operators_Step_Audit.ipynb']:
 p=R/relative;n=nbformat.read(p,4)
 url='https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/'+relative
 if url not in '\n'.join(c.source for c in n.cells):n.cells[0].source+='\n\n[Open in Colab]('+url+')'
 if 'week13' in relative and 'MIE690A article-aligned validation v4' not in n.cells[0].source:n.cells[0].source+='\n\n<!-- MIE690A article-aligned validation v4 -->'
 if 'week15' in relative:
  start=n.cells[0].source.index('### Run')
  n.cells[0].source=n.cells[0].source[:start]+'''### Run
Use a complete repository checkout with numpy, pandas, matplotlib, scipy,
nbformat and ipykernel. Uploading just two archives is insufficient: the full
lab also reads the geometry-holdout and DSMC V5 directories. In Colab, run
`!git clone --depth 1 https://github.com/Ehsan-Roohi/FlowMLLab.git` and
`%cd /content/FlowMLLab` before this notebook. No paid GPU is needed.
\n[Open in Colab]('''+url+')'
 nbformat.write(n,p)
