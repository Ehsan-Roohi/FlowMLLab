"""Build the Week 16 lecture and execute its data-based teaching notebook."""
from pathlib import Path
import json,html,sys,os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'results/week16_lowboom'


def notebook():
    """Execute the curated Week 16 teaching notebook without regenerating its content.

    The notebook itself is the maintained instructional source. Keeping execution
    separate prevents the build script from overwriting the pedagogical Markdown,
    equations, exercises, and lightweight no-output source committed to the repo.
    """
    target=ROOT/'notebooks/week16/W16_Supersonic_Shape_Optimization.ipynb'
    nb=nbf.read(target,as_version=4)
    assert nb.nbformat==4
    assert len(nb.cells)>=30
    assert any(c.cell_type=='markdown' and 'Why this is not yet a ground sonic-boom calculation' in ''.join(c.source) for c in nb.cells)
    assert any(c.cell_type=='markdown' and 'Proper Orthogonal Decomposition' in ''.join(c.source) for c in nb.cells)
    source_nb=nbf.from_dict(json.loads(nbf.writes(nb)))
    if os.environ.get('FLOWMLLAB_INPROCESS')=='1':
        from execute_inprocess import execute
        executed=execute(source_nb)
        method='IPython in-process'
    else:
        executed=NotebookClient(
            source_nb,
            timeout=600,
            kernel_name='python3',
            resources={'metadata':{'path':str(ROOT)}}
        ).execute()
        method='nbclient with a real Jupyter kernel'
    # Keep the committed notebook lightweight: validate execution but do not
    # persist generated figures/tables as embedded output blobs.
    for cell in source_nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    nbf.write(source_nb,target)
    print('Validated educational notebook:',len(source_nb.cells),'cells')
    report=json.loads((E/'release_check.json').read_text())
    report['notebook_executed']=True
    report['notebook_execution_method']=method
    report['notebook_source_kept_lightweight']=True
    (E/'release_check.json').write_text(json.dumps(report,indent=2))

def lecture():
    """Validate the curated Week 16 lecture PDF without regenerating it.

    The canonical lecture is authored as a Times-style LaTeX document and
    committed as a reviewed PDF.  Keeping this function validation-only
    prevents the old ReportLab generator from overwriting the pedagogical
    source, equations, figures, and typography during publication builds.
    """
    target=ROOT/'lectures/week16_supersonic_shape_optimization.pdf'
    source=ROOT/'lectures/source/week16_supersonic_shape_optimization.tex'
    assert target.is_file() and target.stat().st_size>100_000
    assert source.is_file()
    tex=source.read_text()
    required=[
        'Supersonic Shape Optimization and Sonic-Boom Physics',
        'Prandtl--Meyer expansion',
        'Generalized Burgers-type propagation',
        'Axisymmetric compressible Euler equations',
        'Taylor--Maccoll cone benchmark',
        'Proper Orthogonal Decomposition',
        'NASA SEEB-ALR: a useful failed research extension',
    ]
    for phrase in required:
        assert phrase in tex, f'missing lecture topic: {phrase}'
    try:
        from pypdf import PdfReader
        pages=len(PdfReader(str(target)).pages)
        assert pages>=18
    except ImportError:
        pages=None
    print('Validated curated Week 16 lecture:',target,'pages=',pages)

if __name__=='__main__':
    notebook();lecture()
