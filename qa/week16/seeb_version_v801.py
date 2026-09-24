"""Controlled version replay of a retained resolved-nose NASA case.

Copy the verified original mesh and configuration byte-for-byte. Only the pinned
solver executable changes; all numerical, physical and comparison gates remain.
"""
from pathlib import Path
import argparse,json,os,shutil,subprocess,time
import numpy as np
import seeb_reference as base
from seeb_resolved import physical_audit
from install_su2_801 import BINARY_SHA256,ASSET_SHA256,URL
ROOT=Path(__file__).resolve().parents[2]
SOURCES=['seeb_version_v801.py','seeb_reference.py','seeb_resolved.py','cfd.py','analyze.py','install_su2_801.py']


def run(reference_folder,name=None):
    reference=Path(reference_folder).resolve()
    old=json.loads((reference/'metadata.json').read_text());level=old['level']
    for filename,expected in old['sha256'].items():
        if base.sha256(reference/filename)!=expected:raise ValueError(f'Changed original evidence: {filename}')
    name=name or f'seeb_resolved_v801_level_{level:g}'
    if Path(name).name!=name:raise ValueError('Output name must be a single directory name')
    exe=Path(os.environ.get('SU2_CFD',str(ROOT/'.tools/week16_su2_801/bin/SU2_CFD'))).resolve()
    if base.sha256(exe)!=BINARY_SHA256:raise ValueError('Not the pinned official SU2 8.0.1 executable')
    folder=ROOT/'results/week16_lowboom/reference'/name;folder.mkdir(parents=True,exist_ok=False)
    source={n:base.sha256(ROOT/'qa/week16'/n) for n in SOURCES}
    meta={k:old[k] for k in ['level','mesh_family','cells','nodes','min_scaled_jacobian','nose_x_over_L','nose_radius_over_L',
         'tail_x_over_L','tail_radius_over_L','nose_cap_cells','height_over_L','end_over_L','shear_dx_dr','spacing_diagnostics',
         'cad_sha256','mach','axisymmetric','model','geometry','max_cfl','entropy_fix_coeff','convective_flux','fixed_cfl',
         'limiter_freeze_iteration','convergence_start_iteration']}
    meta.update(status='copying',converged=False,solver_version='8.0.1',solver_binary_sha256=BINARY_SHA256,
                official_asset_url=URL,official_asset_sha256=ASSET_SHA256,source_sha256=source,
                original_generating_source_sha256=old['source_sha256'],reference_metadata_sha256=base.sha256(reference/'metadata.json'),
                reference_solver_version='8.5.0',reference_threads=old.get('threads'),threads=1,full_reference_validation_at_generation=False,
                scope='Controlled solver-version replay; full validation additionally requires the complete mesh family and experimental comparison.')
    failure=None
    try:
        shutil.copyfile(reference/'metadata.json',folder/'reference_metadata.json')
        for filename in ['mesh.su2','flow.cfg','cad_meridian.npz']:
            shutil.copyfile(reference/filename,folder/filename)
            if base.sha256(folder/filename)!=base.sha256(reference/filename):raise ValueError('Copy identity failed')
        if (reference/'mesh.msh').exists():shutil.copyfile(reference/'mesh.msh',folder/'mesh.msh')
        base.check_su2_mesh(folder/'mesh.su2',meta['cells'],meta['nodes'])
        meta['unchanged_reference_hashes']={n:base.sha256(reference/n) for n in ['mesh.su2','flow.cfg','cad_meridian.npz']}
        meta['sha256']={n:base.sha256(folder/n) for n in ['mesh.su2','flow.cfg','cad_meridian.npz']}
        for n in ['mesh.su2','flow.cfg']:meta[n+'_sha256']=meta['sha256'][n]
        meta['status']='running';base.atomic_json(folder/'metadata.json',meta)
        start=time.time();env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.run([str(exe),'flow.cfg'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
            log.flush();os.fsync(log.fileno())
        meta.update(returncode=proc.returncode,wall_seconds=time.time()-start)
        if proc.returncode:raise RuntimeError(f'SU2 exited {proc.returncode}; see retained solver.log')
        meta.update(base.extract(folder))
        meta['converged']=bool(meta['density_residual_log10']<=-9 and meta['residual_drop']>=5 and meta['drag_tail_relative_range']<1e-4)
        meta['status']='converged' if meta['converged'] else 'not_converged'
        meta['sha256'].update({n:base.sha256(folder/n) for n in ['history.csv','restart_flow.csv','signature.npz','solver.log']})
        base.atomic_json(folder/'metadata.json',meta)
        base.verify_run(folder)
    except BaseException as exc:
        failure=exc;meta['status']='failed';meta['error']=f'{type(exc).__name__}: {exc}'
    finally:
        try:audit=physical_audit(folder)
        except Exception as exc:audit={'passed':False,'error':str(exc),'checks':{'physical_fields_available_and_readable':False}}
        base.atomic_json(folder/'physical_audit.json',audit)
        meta['physical_plausibility_passed']=audit['passed']
        meta['physical_audit_sha256']=base.sha256(folder/'physical_audit.json')
        if meta.get('converged') and not audit['passed']:meta['status']='physical_validation_failed'
        meta.setdefault('sha256',{}).update({p.name:base.sha256(p) for p in folder.iterdir() if p.is_file() and p.name not in ['metadata.json','metadata.json.partial']})
        meta['generating_sources_unchanged']=source=={n:base.sha256(ROOT/'qa/week16'/n) for n in SOURCES}
        base.atomic_json(folder/'metadata.json',meta)
        if not meta['generating_sources_unchanged']:raise RuntimeError('Generating sources changed')
    if failure is not None:raise failure
    base.verify_run(folder)
    if not meta['converged'] or not meta['physical_plausibility_passed']:
        raise RuntimeError('Unchanged NASA numerical/physical acceptance checks failed')
    return meta

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-folder',type=Path,required=True);p.add_argument('--name')
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
