"""Execute updated teaching notebooks in an isolated CPU kernel and save evidence.

Only this explicit QA command refreshes outputs. Student notebooks are never
treated as proof of a fresh run until the kernel finishes without errors.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks',nargs='+')
    args=parser.parse_args()
    kernels=ROOT/'tmp/feedback_kernels'
    spec=kernels/'feedback';spec.mkdir(parents=True,exist_ok=True)
    # The kernel inherits the process environment; serialize only our overrides.
    env=dict(PYTHONPATH=os.pathsep.join([str(ROOT/'.deps'),str(ROOT),str(ROOT/'common')]),
               IPYTHONDIR=str(ROOT/'tmp/feedback_ipython'),MPLBACKEND='module://matplotlib_inline.backend_inline',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'FlowMLLab CPU feedback QA','language':'python','env':env}),encoding='utf-8')
    result_path=ROOT/'output/feedback_execution.json';result_path.parent.mkdir(exist_ok=True)
    results=json.loads(result_path.read_text()) if result_path.exists() else {}
    for name in args.notebooks:
        path=next((ROOT/'notebooks').rglob(name))
        key=path.relative_to(ROOT).as_posix()
        n=nbformat.read(path,as_version=4)
        # These are independent kernels. No training/validation state is shared.
        km=KernelManager(kernel_name='feedback',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        client=NotebookClient(n,timeout=900,kernel_name='feedback',km=km,
                              resources={'metadata':{'path':str(ROOT)}})
        start=time.perf_counter();print('RUN',name,flush=True)
        try:
            client.execute()
            nbformat.validate(n)
            nbformat.write(n,path)
            results[key]={'status':'passed','seconds':round(time.perf_counter()-start,2),
                          'code_cells':sum(c.cell_type=='code' for c in n.cells),
                          'code_sha256':hashlib.sha256(json.dumps([c.source for c in n.cells if c.cell_type=='code']).encode()).hexdigest(),
                          'source_sha256':hashlib.sha256(''.join(c.source for c in n.cells).encode()).hexdigest()}
            print('PASS',name,results[key]['seconds'],flush=True)
        except Exception as exc:
            results[key]={'status':'failed','seconds':round(time.perf_counter()-start,2),'error':str(exc)[-4500:]}
            print('FAIL',name,str(exc)[-1600:],flush=True)
        result_path.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
