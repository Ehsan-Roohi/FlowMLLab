"""Release gate for retained source identity and independently recomputed metrics."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
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
    assert neural==saved,'Neural audit report is stale'
    assert neural['passed'] and all(neural['identity_checks'].values())
    cfd=cfd_report(write=False)
    assert cfd==json.loads((r/'seeb_validation.json').read_text()),'CFD comparison is stale'
    assert cfd['passed'],cfd['checks']
    for row in cfd['runs']:
        d=r/f"seeb_level_{row['level']:g}";m=row['metadata']
        assert m['returncode']==0 and m['converged']
        assert m['density_residual_log10']<=-9 and m['residual_drop']>=5
        assert m['min_pressure']>0 and m['min_density']>0
        assert hashlib.sha256((d/'flow.cfg').read_bytes()).hexdigest()==m['flow.cfg_sha256']
        assert 'AXISYMMETRIC= YES' in (d/'flow.cfg').read_text()
        assert 'MACH_NUMBER= 1.6' in (d/'flow.cfg').read_text()
        assert m['cad_sha256']==hashlib.sha256((ROOT/'cases/week16_lowboom/reference/SEEB-ALR-as-built.stp').read_bytes()).hexdigest()
    return {'status':'pass','neural_cases':neural['cases'],'NASA_meshes':len(cfd['runs']),'finest_cells':cfd['runs'][-1]['cells']}

if __name__=='__main__':print(json.dumps(validate(),indent=2))
