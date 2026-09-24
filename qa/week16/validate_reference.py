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

def validate(scope='validated_core'):
    """Accept the declared teaching scope; failed NASA work is never a passing gate."""
    assert scope in {'validated_core', 'full_nasa'}, 'Unknown release scope'
    r=ROOT/'results/week16_lowboom/reference'
    src=ROOT/'cases/week16_lowboom/reference'
    manifest=json.loads((src/'source_manifest.json').read_text())
    for row in manifest['files']:
        assert hashlib.sha256((src/row['file']).read_bytes()).hexdigest()==row['sha256']
    from geometry_volume_audit import report as volume_report
    from clean_campaign_v801 import assemble as clean_campaign_report
    from clean_model_v801 import audit as clean_model_report
    from weakwall_report import report as weakwall_report
    from weakwall_design_report import report as design_report
    from cone_refinement_v801 import build_report as cone_report
    volume=volume_report(write=False);assert volume['passed']
    campaign=clean_campaign_report(write=False);assert campaign['passed'] and campaign['cases']==44
    model=clean_model_report(write=False);assert model['passed']
    for build,filename in [(weakwall_report,'weakwall_checkpoint_audit.json'),(design_report,'weakwall_design_audit.json'),(cone_report,'cone_refinement_v801.json')]:
        actual=build(write=False)
        assert actual['passed'],filename
        assert same(actual,json.loads((r/filename).read_text())),f'Stale accepted report: {filename}'
    result={'status':'pass','release_scope':scope,'clean_dataset_cases':44,
            'clean_model_finer_wave_error':model['finer_mesh']['wave_relative_l2'],
            'NASA':{'status':'failed_deferred','passed':False,'included_in_release':False,
                    'reason':'Resolved NASA CFD has not passed the complete numerical, physical, mesh-family and experimental gates.'}}
    if scope=='validated_core':return result
    cfd=cfd_report(write=False)
    assert same(cfd,json.loads((r/'seeb_validation.json').read_text())),'CFD comparison is stale'
    assert cfd['passed'],cfd['checks']
    for row in cfd['runs']:
        d=r/row['folder'];m=row['metadata']
        verify_run(d)
        from seeb_resolved import physical_audit
        assert physical_audit(d)['passed']
        assert m['solver_version']=='8.0.1'
        from install_su2_801 import BINARY_SHA256
        assert m['solver_binary_sha256']==BINARY_SHA256
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
    result['NASA']={'status':'pass','passed':True,'included_in_release':True,'meshes':len(cfd['runs']),'finest_cells':cfd['runs'][-1]['cells']}
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['validated_core','full_nasa'],default='validated_core')
    print(json.dumps(validate(parser.parse_args().scope),indent=2))
