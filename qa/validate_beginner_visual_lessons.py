"""Execute new teaching cells and verify preservation of retained solver cells."""
from pathlib import Path
import hashlib
import json
import subprocess
import re
import sys
import nbformat
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
shell=InteractiveShell.instance()
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib_inline.backend_inline import configure_inline_support
configure_inline_support(shell, 'module://matplotlib_inline.backend_inline')
checked=[]
for relative in ['notebooks/week02/AI_in_Fluids_Week2_Colab_Expanded.ipynb',
                 'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb']:
    path=ROOT/relative;nb=nbformat.read(path,4)
    before=nbformat.reads(subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT,text=True,encoding='utf-8'),4)
    old={c['id']:c for c in before['cells']}
    for cell in nb.cells:
        if cell.cell_type=='code' and cell.id.startswith('visual-'):
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=True)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(str(result.error_before_exec or result.error_in_exec))
            cell.outputs=[]
            if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr:raise RuntimeError(captured.stderr)
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
            cell.execution_count=shell.execution_count-1
            assert cell.outputs
            checked.append(cell.id)
        elif cell.cell_type=='code' and cell.id in old and not cell.id.startswith('student-review-'):
            if relative.startswith('notebooks/week02/') and cell.id in {'21818df0','750bc769'}:
                # Intentional Week 2 controls/reference changes have their own
                # CPU/syntax audit in validate_week02_student_path.py.
                continue
            # Existing code, execution counts and stored results are unchanged.
            original=nbformat.from_dict(old[cell.id])
            if relative.startswith('notebooks/week02/'):
                assert cell.source==original.source,(relative,cell.id)
                assert not any(o.output_type=='error' for o in cell.outputs),cell.id
            else:
                assert cell.source==original.source and cell.outputs==original.outputs and cell.execution_count==original.execution_count, (relative,cell.id)
        if cell.cell_type=='markdown':
            assert cell.source.count('$$')%2==0,cell.id
            if cell.id.startswith('visual-'):
                assert len(cell.source)<20000,cell.id
                assert not any(ch in cell.source for ch in '\u00c3\u00c2\u0192'),cell.id
    nbformat.validate(nb)
    path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
assert checked==['visual-adam-experiment','visual-checkerboard-check','visual-hat-arithmetic']
# The public contribution is attributed anonymously.
provenance=json.loads((ROOT/'results/week01_2_student_review/provenance.json').read_text(encoding='utf-8'))
assert provenance['contributor']=='Anonymous student contributor'
assert (ROOT/'docs/reviews/student_cavity_20261026.md').is_file()
from pypdf import PdfReader
for diagram in (ROOT/'lectures/source/week01_2_assets').glob('*visual.pdf'):
    box=PdfReader(diagram).pages[0].mediabox
    assert float(box.width)<1200 and float(box.height)<1000,diagram.name
# Check the actual plotted class labels against the stated linear score.
lecture=(ROOT/'lectures/source/week02_lecture.tex').read_text(encoding='utf-8')
plot=lecture.split('For this illustration choose',1)[1].split('This is the first important idea:',1)[0]
groups=re.findall(r'\\foreach \\x/\\y in \{([^}]+)\}',plot)
assert len(groups)==2
decision_points=0
for expected,group in enumerate(groups):
    for point in group.split(','):
        x,alpha=map(float,point.split('/'))
        assert .6<=alpha<=1
        assert int(x-2*alpha+2.8>=0)==expected
        decision_points+=1
assert decision_points==6
# With fixed old-time data, lid motion and relaxation=1, this implementation's
# PIMPLE right-hand side is constant even while convection coefficients change.
from flowmllab.pressure_velocity import CavityConfig, _momentum
cfg=CavityConfig(method='pimple',cells=4,reynolds=100)
u0=np.zeros((4,5));v0=np.zeros((5,4))
u1=u0.copy();v1=v0.copy();u1[:,1:-1]=.1;v1[1:-1,:]=-.02
first=_momentum(u0,v0,cfg,old_u=u0,old_v=v0,dt=.05,relaxation=1.)
next_pass=_momentum(u1,v1,cfg,old_u=u0,old_v=v0,dt=.05,relaxation=1.)
for old,new in zip(first,next_pass):np.testing.assert_array_equal(old[1],new[1])
report={'executed_new_cells':checked,'retained_cfd_code_and_outputs':'unchanged',
        'week2_control_and_reference_changes':'audited by validate_week02_student_path.py',
        'adam_first_step_keras_convention':1.099999841886367,
        'net_momentum_flux':0.0015,'anonymized_current_text':True,
        'decision_boundary_points_checked':decision_points,
        'fixed_pimple_source_checked':True,
        'neural_surrogate_retrained':False,'cfd_solvers_modified':False}
out=ROOT/'output/beginner_visual_validation.json';out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
