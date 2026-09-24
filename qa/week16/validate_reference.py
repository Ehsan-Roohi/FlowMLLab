"""Release gate for retained source identity and independently recomputed metrics."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from freeze_model import audit as checkpoint_report
from seeb_reference import verify_run
from recompute_neural_cfd import report as recomputed_checkpoint_report

def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(float,int)) and bool(np.isclose(a,b,rtol=1e-10,atol=1e-12))
    return a==b

from independent_audit_report import build_report as neural_report
from reference_report import build_report as cfd_report
ROOT=Path(__file__).resolve().parents[2]

def validate():
    r=ROOT/'results/week16_lowboom/reference'
    src=ROOT/'cases/week16_lowboom/reference'
    manifest=json.loads((src/'source_manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((src/row['file']).read_bytes()).hexdigest()==row['sha256']
    neural=neural_report(write=False)
    saved=json.loads((r/'neural_audit.json').read_text())
    assert same(neural,saved),'Neural audit report is stale'
    assert neural['passed'] and all(neural['identity_checks'].values())
    cfd=cfd_report(write=False)
    assert same(cfd,json.loads((r/'seeb_validation.json').read_text())),'CFD comparison is stale'
    assert cfd['passed'],cfd['checks']
    checkpoint=checkpoint_report(write=False)
    assert checkpoint['passed']
    assert same(checkpoint,json.loads((r/'checkpoint_audit.json').read_text())), 'Checkpoint audit is stale'
    recomputed=recomputed_checkpoint_report(write=False)
    assert recomputed['passed']
    assert same(recomputed,json.loads((r/'recomputed_checkpoint_audit.json').read_text())), 'Recomputed checkpoint audit is stale'
    for row in cfd['runs']:
        d=r/f"seeb_level_{row['level']:g}";m=row['metadata']
        verify_run(d)
        for name,expected in m['source_sha256'].items():
            assert hashlib.sha256((ROOT/'qa/week16'/name).read_bytes()).hexdigest()==expected, 'CFD generating source changed'
        assert m['fixed_cfl'] and m['max_cfl']==5 and m['entropy_fix_coeff']==.05
        assert m['limiter_freeze_iteration']==round(2000*row['level']) and m['convergence_start_iteration']==round(2000*row['level'])+500
        assert m['returncode']==0 and m['converged']
        assert m['density_residual_log10']<=-9 and m['residual_drop']>=5
        assert m['min_pressure']>0 and m['min_density']>0
        assert hashlib.sha256((d/'flow.cfg').read_bytes()).hexdigest()==m['flow.cfg_sha256']
        assert 'AXISYMMETRIC= YES' in (d/'flow.cfg').read_text()
        assert 'MACH_NUMBER= 1.6' in (d/'flow.cfg').read_text()
        assert m['cad_sha256']==hashlib.sha256((ROOT/'cases/week16_lowboom/reference/SEEB-ALR-as-built.stp').read_bytes()).hexdigest()
    return {'status':'pass','neural_cases':neural['cases'],'recomputed_checkpoint_cases':recomputed['cases'],'NASA_meshes':len(cfd['runs']),'finest_cells':cfd['runs'][-1]['cells']}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
