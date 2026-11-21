"""Execute selected default CPU notebooks and retain source/output identities."""
from pathlib import Path
import hashlib,json,os,sys,time,traceback
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
TMP=Path(os.environ.get('FLOWMLLAB_QA_TMP',str(ROOT/'tmp/course_middle')));TMP.mkdir(parents=True,exist_ok=True)
for key in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','JUPYTER_RUNTIME_DIR','IPYTHONDIR'):
    directory=TMP/key.lower();directory.mkdir(exist_ok=True);os.environ[key]=str(directory)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',FLOWMLLAB_COURSE_REFRESH_EXPORT='1')
os.environ.pop('FLOWMLLAB_WRITE_RESULTS',None)
sys.path.insert(0,str(ROOT))
from qa.course_middle_notebook_updates import TARGETS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=sys.argv[1:] or sorted(TARGETS)
for name in names:
    path=next((ROOT/'notebooks').glob('week*/'+name))
    nb=nbformat.read(path,as_version=4)
    ident=path.stem
    record={'path':path.relative_to(ROOT).as_posix(),'source_sha256':sha(path),'python':sys.executable,'mode':'default CPU Run All','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    started=time.perf_counter()
    try:
        km=KernelManager(kernel_name='python3');km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        def progress(cell,cell_index,**kwargs):
            note={'path':record['path'],'cell':cell_index,'cell_type':cell.cell_type,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            (TMP/(ident+'_progress.json')).write_text(json.dumps(note)+'\n',encoding='utf-8',newline='\n')
            print(json.dumps(note),flush=True)
        NotebookClient(nb,km=km,timeout=1800,on_cell_start=progress,resources={'metadata':{'path':str(path.parent)}}).execute()
        path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
        outputs=[o for c in nb.cells for o in c.get('outputs',[])]
        record.update(status='passed',cell_errors=sum(o.get('output_type')=='error' for o in outputs),figures=sum('image/png' in o.get('data',{}) for o in outputs),notebook_sha256=sha(path))
    except Exception as exc:
        record.update(status='failed',error=str(exc)[-4000:])
        nbformat.write(nb,TMP/(ident+'_failed.ipynb'))

    if 'km' in locals() and km.has_kernel:
        km.shutdown_kernel(now=True)
    record['elapsed_s']=time.perf_counter()-started
    (TMP/(ident+'_execution.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record),flush=True)
    if record['status']!='passed':sys.exit(1)
