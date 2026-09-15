"""Pin the compact, redistributable teaching evidence (not upstream source code)."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'results/week14_validation'
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())
       if p.suffix in ['.png','.npz','.json'] and p.name!='manifest.json'}
manifest={'source_url':'https://www.cfd-sweden.se/lada/pythons-rans-code-RANS-open.tar.gz',
 'source_archive_sha256':'7aae30d0e990e78ac008c03d008ddc7609ce3a4aae9cc55fada65f1be91d774f',
 'upstream_update_date':'2026-09-11',
 'files':files,'versions':{k:importlib.metadata.version(k) for k in
 ['numpy','scipy','matplotlib','scikit-learn','torch','nbformat','nbclient','pyamg']},
 'attribution':{'solver_and_workflow':'Lars Davidson','DNS':'Myoungkyu Lee and Robert D. Moser',
 'teaching_explanations_figures_audit':'Ehsan Roohi / FlowMLLab, AI-assisted'},
 'permission':'Instructor reports permission in shared conversation; exact written terms not supplied. No general relicensing of upstream material.',
 'reuse_boundary':'Attribution retained; publication authorized by instructor reporting permission. No upstream solver/checkpoints distributed and no blanket third-party relicensing.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(f'Pinned {len(files)} compact evidence files')
