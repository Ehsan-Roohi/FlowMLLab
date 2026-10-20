"""Verify formulation archives, then execute the Week 1.2 retained-results notebook."""
from pathlib import Path
import sys,os,json,hashlib,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
import nbformat as n
from week01_2_formulation_notebook import append_formulations
from report_week01_2_formulations import main as report
from validate_week01_2 import execute

def main():
 for pattern in ['test_pressure_velocity.py','test_pressure_velocity_experiments.py']:
  subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,check=True)
 bundle=ROOT/'results/week01_2_formulations/provenance/source_bundle.zip'
 with zipfile.ZipFile(bundle) as z:
  for p in (ROOT/'results/week01_2_formulations').glob('*-re*-n*.json'):
   r=json.loads(p.read_text());source='flowmllab/streamfunction_experiments.py' if r['config']['method']=='streamfunction-vorticity' else 'flowmllab/pressure_velocity.py'
   archived=z.read(source)
   assert hashlib.sha256(archived).hexdigest()==r['source_sha256'][source]
   assert archived.replace(b'\r\n',b'\n')==(ROOT/source).read_bytes().replace(b'\r\n',b'\n')
 report()
 path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
 nb=n.read(path,4);nb.cells=[c for c in nb.cells if not c.metadata.get('formulation_comparison')]
 append_formulations(nb)
 for c in nb.cells:
  if c.cell_type=='code':compile(c.source,str(path),'exec')
 n.validate(nb)
 kernels=ROOT/'tmp/week01_2_kernels';folder=kernels/'week01_2';folder.mkdir(exist_ok=True,parents=True)
 (folder/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Week 1.2 audit','language':'python','env':{'PYTHONPATH':os.environ.get('PYTHONPATH',''),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPLBACKEND':'module://matplotlib_inline.backend_inline'}}))
 counts=execute(nb,[str(kernels)]);assert counts=={'executed_code_cells':7,'errors':0,'png_outputs':16},counts
 for c in nb.cells:c.get('metadata',{}).pop('execution',None)
 n.write(nb,path);path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
 target=ROOT/'results/week01_2_formulations/validation.json';v=json.loads(target.read_text())
 v.update(executed_core_source_bundle_verified=True,source_bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),notebook_execution=counts,notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),unit_tests_passed=7,fresh_notebook_branch='Adapter exercised by retained scientific runs; opt-in notebook branch compiled, not separately rerun',hosted_colab_execution=False)
 attempts=ROOT/'results/week01_2_formulations/attempts';verified=0
 for p in attempts.glob('*-initial.json'):
  r=json.loads(p.read_text());archive=p.with_suffix('.npz')
  assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['field_sha256'];verified+=1
 v['retained_initial_attempts_hash_verified']=verified
 target.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
