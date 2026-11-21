"""Execute explicitly named CPU lessons and preserve a per-notebook evidence receipt."""
from pathlib import Path
import argparse,os,sys,json,time,hashlib,base64,platform
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
TMP=Path(os.environ.get('FLOWMLLAB_EARLY_TMP',str(ROOT/'tmp/course_refresh_early')));TMP.mkdir(parents=True,exist_ok=True)
for key,name in [('MPLCONFIGDIR','mpl'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','runtime'),('JUPYTER_CONFIG_DIR','config'),('TEMP','temp'),('TMP','temp')]:
 p=TMP/name;p.mkdir(exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2')
os.environ['PYTHONPATH']=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
os.environ['JUPYTER_PATH']=str(Path(sys.prefix)/'share/jupyter')
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

def run(relative):
 path=ROOT/relative;nb=nbformat.read(path,as_version=4)
 out=ROOT/'results/course_refresh_early'/path.stem;out.mkdir(parents=True,exist_ok=True)
 work=ROOT/'student_runs/course_refresh_early'/path.stem;work.mkdir(parents=True,exist_ok=True)
 os.environ['FLOWMLLAB_EARLY_OUTPUT']=str(out)
 km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
 print('START',relative,flush=True);start=time.perf_counter()
 def progress(cell_index,**kwargs):print('CELL',cell_index,round(time.perf_counter()-start,1),flush=True)
 client=NotebookClient(nb,timeout=7200,km=km,resources={'metadata':{'path':str(work)}},on_cell_start=progress)
 failure=None
 try:client.execute()
 except Exception as exc:failure=repr(exc)
 elapsed=time.perf_counter()-start
 path.write_text(nbformat.writes(nb)+"\n",encoding="utf-8",newline="\n");nbformat.validate(nb)
 figures=[]
 for i,c in enumerate(nb.cells):
  for j,o in enumerate(c.get('outputs',[])):
   if 'image/png' in o.get('data',{}):
    p=out/f'cell_{i:02d}_{j:02d}.png';p.write_bytes(base64.b64decode(o['data']['image/png']));figures.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 record={'notebook':relative,'utc':datetime.now(timezone.utc).isoformat(),'elapsed_run_all_seconds':elapsed,'host':platform.node(),'kernel':sys.executable,'python':sys.version,'cpu_only':True,'failure':failure,'code_cells':sum(c.cell_type=='code' for c in nb.cells),'executed_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),'inline_figures':figures,'notebook_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),'claim':'Fresh default CPU Run All; no hosted Colab or optional GPU execution claimed.'}
 (out/'execution.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
 print('FINISH',relative,round(elapsed,2),'figures',len(figures),'failure',failure,flush=True)
 if failure:raise RuntimeError(failure)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('notebooks',nargs='+');args=ap.parse_args()
 for f in args.notebooks:run(f)