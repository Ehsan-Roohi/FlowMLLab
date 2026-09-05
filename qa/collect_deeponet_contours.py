"""Run on Unity: package existing NOUT0098 fields, without rerunning DSMC."""
from pathlib import Path
import argparse
import hashlib
import json
import uuid
import zipfile

def collect(project, output):
    project = project.resolve()
    cases = {
        'deeponet': project/'DS2V_UNIFIED_M10_PRODUCTION_V2_20260724_212309'/'neural',
        'exact': project/'DS2V_UNIFIED_EXACT_M10_LONG_20260726_233749',
    }
    selected = {}
    available = {}
    for label, case in cases.items():
        fields = sorted(p for p in case.rglob('DS2FF.DAT')
                        if p.is_file() and p.stat().st_size > 0
                        and p.resolve().is_relative_to(project))
        available[label] = [str(p.relative_to(project)) for p in fields]
        selected[label] = [p for p in fields if p.parent.name == 'NOUT0098']
    if any(not paths for paths in selected.values()):
        print('Cannot find both NOUT0098 full fields. No substitute run was selected.')
        print(json.dumps(available, indent=2))
        return False
    files = set()
    for label, fields in selected.items():
        case = cases[label]
        for field in fields:
            files.add(field)
            for name in ['METADATA.env','DS2V_UNIFIED_AUDIT.json','DS2SU - Copy.DAT','run_tail.txt']:
                p=field.parent/name
                if p.is_file(): files.add(p)
        for folder in [case,case/'input',case/'results']:
            for name in ['DS2VD.DAT','DS2VD.TXT','INPUT_RUNTIME_SHA256.txt','RUN_MANIFEST.env',
                         'TABLE_SHA256.txt','unified_table.ulj.sha256']:
                p=folder/name
                if p.is_file(): files.add(p)
    manifest = {'purpose':'Existing full-field data for the audited exact/DeepONet cylinder pair',
                'requested_snapshot':98,'files':[]}
    for p in sorted(files):
        if not p.resolve().is_relative_to(project): raise ValueError('Path outside selected project')
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): digest.update(chunk)
        manifest['files'].append({'path':str(p.relative_to(project)), 'size':p.stat().st_size,'sha256':digest.hexdigest()})
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(files): z.write(p,p.relative_to(project).as_posix())
        z.writestr('CONTOUR_EXPORT_MANIFEST.json',json.dumps(manifest,indent=2))
    print('Upload this file:',output.resolve())
    return True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path('/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2'))
    parser.add_argument('--output',type=Path,default=Path('DEEPO_NET_CYLINDER_CONTOURS_'+uuid.uuid4().hex[:8]+'.zip'))
    args=parser.parse_args()
    raise SystemExit(0 if collect(args.project,args.output) else 2)
