"""Execute ordinary Python notebook cells when local kernel sockets are unavailable.
Runs in a fresh process through IPython; captures real text, tables and PNGs.
No fabricated outputs and no exception suppression. This runner intentionally
supports this Python-only notebook, not arbitrary kernel or shell magics.
"""
import base64,io
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display,Image
import matplotlib.pyplot as plt

def execute(nb):
    shell=InteractiveShell.instance()
    def show(*args,**kwargs):
        for n in plt.get_fignums():
            fig=plt.figure(n);b=io.BytesIO();fig.savefig(b,format='png',dpi=130,bbox_inches='tight');display(Image(data=b.getvalue()))
        plt.close('all')
    original=plt.show;plt.show=show;count=0
    try:
        for cell in nb.cells:
            if cell.cell_type!='code':continue
            count+=1
            with capture_output(stdout=True,stderr=True,display=True) as cap:
                result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec:raise result.error_before_exec
            if result.error_in_exec:raise result.error_in_exec
            outputs=[]
            if cap.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
            if cap.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
            for out in cap.outputs:outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
            cell.outputs=outputs;cell.execution_count=count
    finally:plt.show=original
    nb.metadata['execution']={'method':'IPython in-process in a fresh Python process; sandbox prohibits kernel sockets','executed_code_cells':count}
    return nb
