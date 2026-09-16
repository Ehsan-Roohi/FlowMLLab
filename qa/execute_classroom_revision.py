"""Execute revised notebooks with the current interpreter; preserve errors."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
root=Path(__file__).resolve().parents[1]
paths=[root/p for p in sys.argv[1:]] if len(sys.argv)>1 else sorted((root/'notebooks').glob('**/*.ipynb'))
for path in paths:
    nb=nbformat.read(path,4)
    if len(sys.argv)==1 and nb.metadata.get('classroom_revision')!=1: continue
    print('EXECUTE',path,flush=True)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    try:
        NotebookClient(nb,km=km,timeout=1800,resources={'metadata':{'path':str(path.parent)}}).execute()
    finally:
        nbformat.write(nb,path)
    print('PASS',path.name,flush=True)
