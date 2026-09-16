"""Execute plain-Python notebook cells without opening Jupyter network sockets.

The managed runtime disallows local kernel sockets. This runner is explicitly
not an nbclient/Jupyter-kernel certification; it executes the same cell sources,
in order, in one fresh Python namespace and captures text, tables and figures.
"""
import base64
import contextlib
import io
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import IPython.display


def execute(notebook, cwd):
    namespace={'__name__':'lesson_execution'}
    active=[]
    def display(*objects,**kwargs):
        for obj in objects:
            if isinstance(obj,IPython.display.Image):
                payload=obj.data
                if isinstance(payload,bytes):payload=base64.b64encode(payload).decode()
                active.append(nbformat.v4.new_output('display_data',data={'image/png':payload},metadata={}))
            elif hasattr(obj,'_repr_html_'):
                active.append(nbformat.v4.new_output('display_data',data={'text/html':obj._repr_html_(),'text/plain':str(obj)},metadata={}))
            else:
                active.append(nbformat.v4.new_output('display_data',data={'text/plain':str(obj)},metadata={}))
    def show(*args,**kwargs):
        for number in plt.get_fignums():
            fig=plt.figure(number);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=130,bbox_inches='tight')
            active.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buffer.getvalue()).decode()},metadata={}))
            plt.close(fig)
    original_display=IPython.display.display;original_show=plt.show;original_cwd=Path.cwd()
    IPython.display.display=display;plt.show=show
    try:
        os.chdir(cwd);execution_count=0
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            execution_count+=1;active=[];stream=io.StringIO()
            with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                exec(compile(cell.source,f'notebook_cell_{execution_count}','exec'),namespace)
            if stream.getvalue():active.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue()))
            cell.outputs=active;cell.execution_count=execution_count
        notebook.metadata['execution_audit']={'method':'Sequential plain-Python cells in a fresh process; socket-free runner','jupyter_kernel_executed':False,'reason':'Managed runtime denied local kernel sockets','code_cells':execution_count}
    finally:
        IPython.display.display=original_display;plt.show=original_show;os.chdir(original_cwd)
    nbformat.validate(notebook)
    return notebook
