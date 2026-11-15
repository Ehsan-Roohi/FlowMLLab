"""Execute the three new cavity notebooks in fresh Jupyter kernels before release."""
from pathlib import Path
import hashlib,json,time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOKS=[
 'notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb',
 'notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb',
 'notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb',
]

def main():
    evidence=[]
    for rel in NOTEBOOKS:
        path=ROOT/rel;before=path.read_bytes();doc=nbformat.read(path,as_version=4)
        start=time.perf_counter()
        NotebookClient(doc,timeout=300,kernel_name='python3',allow_errors=False,
                       resources={'metadata':{'path':str(ROOT)}}).execute()
        assert path.read_bytes()==before,'The stored student notebook must stay unchanged'
        evidence.append({'notebook':rel,'sha256':hashlib.sha256(before).hexdigest(),
                         'code_cells':sum(c.cell_type=='code' for c in doc.cells),
                         'seconds':time.perf_counter()-start,'fresh_kernel_passed':True})
        print('Fresh kernel passed:',rel,flush=True)
    out=ROOT/'output/qa/cavity_notebook_execution.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'scope':'Three new cavity notebooks; older notebooks are not rerun',
                              'notebooks':evidence},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
