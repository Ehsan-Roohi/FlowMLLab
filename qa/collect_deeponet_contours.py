"""Run on Unity: package explicitly selected fields, without rerunning DSMC."""
from pathlib import Path
import argparse
import hashlib
import json
import uuid
import zipfile

def collect(project, output, deeponet_nout=98, exact_nout=98):
    requested = {'deeponet': deeponet_nout, 'exact': exact_nout}
    if any(n < 1 for n in requested.values()):
        raise ValueError('Snapshot numbers must be positive')
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
        selected[label] = [p for p in fields if p.parent.name == f'NOUT{requested[label]:04d}']
    if any(not paths for paths in selected.values()):
        print('Cannot find both requested full fields:', requested)
        print('No substitute run or snapshot was selected.')
        print(json.dumps(available, indent=2))
        return False
    if deeponet_nout != exact_nout:
        print('WARNING: different output numbers; these are not synchronized fields.')
    print('Selected snapshots:', requested)
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
                'requested_snapshots':requested,
                'synchronized_time_verified':False,
                'comparison_note':'Inspect retained metadata/header times and sampling windows before quantitative comparison.',
                'selected_fields':{label:[p.relative_to(project).as_posix() for p in paths] for label,paths in selected.items()},
                'files':[]}
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
    parser.add_argument('--deeponet-nout',type=int,default=98)
    parser.add_argument('--exact-nout',type=int,default=98)
    args=parser.parse_args()
    raise SystemExit(0 if collect(args.project,args.output,args.deeponet_nout,args.exact_nout) else 2)
