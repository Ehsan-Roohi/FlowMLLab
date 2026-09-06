"""Generate one reserved teaching test; never fit or evaluate a surrogate here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from generate_cylinder_cfd_dataset import SETTINGS, SPLIT, run_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reynolds = 115
    protocol = {
        'reynolds': reynolds, 'settings': SETTINGS,
        'purpose': 'Reserved independent teaching test; no model selection or scoring',
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'solver_sha256': hashlib.sha256(Path('flowmllab/cylinder_lbm.py').read_bytes()).hexdigest(),
        'release_rule': 'Freeze model, preprocessing, metrics and thresholds before scoring. Public bytes are not access-controlled; log first use and retire blind status after inspection.',
        'quality_scope': 'Array integrity only; grid/domain independence and statistical stationarity are not certified.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    SPLIT[reynolds] = 'reserved_test'
    row = run_case(reynolds, str(args.output))
    archive = args.output/row['file']
    full_sha = row['sha256']
    with np.load(archive, allow_pickle=False) as source:
        compact = {key: source[key] for key in source.files}
    for key in ('u', 'v', 'p', 'snapshot_time'):
        compact[key] = compact[key][::5]
    metadata = json.loads(str(compact['metadata_json']))
    metadata['export_snapshot_subsample_stride'] = 5
    compact['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(archive, **compact)
    row.update(bytes=archive.stat().st_size, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               snapshots=len(compact['snapshot_time']))
    with np.load(archive, allow_pickle=False) as data:
        for key in ('u','v','p','time','lift_coefficient','drag_coefficient'):
            if not np.isfinite(data[key]).all():
                raise ValueError('Nonfinite array: '+key)
        if not (np.diff(data['time']) > 0).all():
            raise ValueError('Nonmonotone time')
    manifest = {key:row[key] for key in ('file','bytes','sha256','snapshots')}
    manifest.update({'source_full_archive_sha256':full_sha, 'export_snapshot_subsample_stride':5, 'status':'reserved_unscored','reynolds':reynolds,'integrity_pass':True})
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
