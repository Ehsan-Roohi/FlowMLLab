"""Execute selected late-course default notebooks in fresh local kernels.

Outputs stay on the project drive. The per-notebook receipt identifies actual
executed bytes; optional research fits are not enabled by this runner.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('FLOWMLLAB_LATE_OUTPUT',str(ROOT/'tmp/course_refresh_late')))

class CurrentKernel(KernelManager):
    def format_kernel_cmd(self,extra_arguments=None):
        args=super().format_kernel_cmd(extra_arguments)
        args[0]=sys.executable
        return args

def digest(data):return hashlib.sha256(data).hexdigest()

def execute(path):
    relative=path.relative_to(ROOT).as_posix()
    nb=nbf.read(path,as_version=4)
    source_sha=digest('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode())
    started=time.perf_counter()
    def progress(cell,cell_index):
        (OUT/'progress.json').write_text(json.dumps(dict(notebook=relative,cell=cell_index,
            id=cell.get('id'),started=time.time())),encoding='utf8',newline='\n')
    client=NotebookClient(nb,timeout=1200,startup_timeout=240,kernel_name='python3',
        kernel_manager_class=CurrentKernel,resources={'metadata':{'path':str(ROOT)}},
        on_cell_start=progress)
    try:client.execute()
    except Exception:
        failed=OUT/'failed'/path.name;failed.parent.mkdir(parents=True,exist_ok=True)
        failed.write_text(nbf.writes(nb),encoding='utf8',newline='\n')
        raise
    image_count=0;new_images=[]
    for c in nb.cells:
        for j,o in enumerate(c.get('outputs',[])):
            if o.get('output_type')=='error':raise AssertionError(relative)
            if 'image/png' not in o.get('data',{}):continue
            image_count+=1
            # Preserve new demonstrations separately for readable visual review.
            if c.id.startswith('late-') or '/week1' in relative or '/week2' in relative:
                png=base64.b64decode(o.data['image/png'])
                dst=OUT/'figures'/path.stem/f'{c.id}-{j}.png'
                dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(png)
                new_images.append(dict(path=dst.relative_to(ROOT).as_posix(),sha256=digest(png)))
    path.write_text(nbf.writes(nb),encoding='utf8',newline='\n')
    row=dict(notebook=relative,method='fresh Jupyter kernel',python=sys.version,
        executable=sys.executable,seconds=time.perf_counter()-started,
        code_sha256=source_sha,notebook_sha256=digest(path.read_bytes()),
        code_cells=sum(c.cell_type=='code' for c in nb.cells),image_outputs=image_count,
        status='passed',review_images=new_images)
    receipt=OUT/'execution.json'
    records=json.loads(receipt.read_text()) if receipt.exists() else []
    records=[r for r in records if r['notebook']!=relative]+[row]
    receipt.write_text(json.dumps(records,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in row.items() if k!='review_images'}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('notebooks',nargs='+')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('TEMP','TMP','MPLCONFIGDIR','JUPYTER_RUNTIME_DIR','IPYTHONDIR'):
        target=OUT/name.lower();target.mkdir(exist_ok=True);os.environ[name]=str(target)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    for path in args.notebooks:execute(ROOT/path)

if __name__=='__main__':main()
