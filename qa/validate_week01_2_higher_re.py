"""Validate retained higher-Re runs and execute both notebook control paths."""
from pathlib import Path
import sys,os,json,hashlib,copy,subprocess
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'qa'))
from validate_week01_2 import execute
from report_week01_2_higher_re import main as report

def main():
    for pattern in ['test_pressure_velocity.py','test_pressure_velocity_experiments.py']:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=ROOT,check=True)
    report()
    kernels=ROOT/'tmp/week01_2_kernels';kernel=kernels/'week01_2';kernel.mkdir(parents=True,exist_ok=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 1.2 audit','language':'python','env':{'PYTHONPATH':os.environ.get('PYTHONPATH',''),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}}))
    note=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    n=nbformat.read(note,as_version=4);default=execute(n,[str(kernels)])
    assert default=={'executed_code_cells':6,'errors':0,'png_outputs':9},default
    nbformat.write(n,note);note.write_bytes(note.read_bytes().replace(b'\r\n',b'\n'))
    # Exercise the fresh interactive branch on a deliberately tiny integration
    # test mesh, not as extra physical accuracy evidence.
    fresh_cell=copy.deepcopy(next(c for c in n.cells if c.cell_type=='code' and 'RUN_HIGHER_RE = False' in c.source))
    fresh_cell.source=fresh_cell.source.replace('RUN_HIGHER_RE = False','RUN_HIGHER_RE = True').replace('HIGHER_CELLS = 32','HIGHER_CELLS = 8')
    fresh_cell.source+='\nassert len(higher_runs)==3 and all(r["converged"] for r in higher_runs)\n'
    check=nbformat.v4.new_notebook(cells=[copy.deepcopy(n.cells[1]),fresh_cell],metadata=n.metadata)
    fresh=execute(check,[str(kernels)])
    assert fresh['executed_code_cells']==2 and fresh['errors']==0 and fresh['png_outputs']==3
    record={'date':'2026-10-15','default_execution':default,'fresh_interactive_branch_integration_test':{'cells':8,**fresh,'accuracy_claim':False},'unit_tests_passed':7}
    p=ROOT/'results/week01_2_pressure_velocity/validation.json';v=json.loads(p.read_text())
    v['notebook_sha256']=hashlib.sha256(note.read_bytes()).hexdigest();v['default_notebook']=default
    v['higher_re_notebook_extension']={**record,'scope':'Experimental extension only; original Re=100 source and fields unchanged.'}
    p.write_text(json.dumps(v,indent=2)+'\n')
    p=ROOT/'results/week01_2_higher_re/validation.json';v=json.loads(p.read_text());v['notebook_checks']=record;p.write_text(json.dumps(v,indent=2)+'\n')
    print('NOTEBOOK_CHECKS',json.dumps(record),flush=True)

if __name__=='__main__':main()
