"""Extract inspected upstream regular files into an isolated, short-name run tree."""
from pathlib import Path
import hashlib
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'tmp/week14-source/upstream.tar.gz'
expected='7aae30d0e990e78ac008c03d008ddc7609ce3a4aae9cc55fada65f1be91d774f'
if not archive.exists():
    archive.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve('https://www.cfd-sweden.se/lada/pythons-rans-code-RANS-open.tar.gz',archive)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:
    raise ValueError('Source archive changed: inspect the new release before executing it')
dest = ROOT / 'tmp/w14'
names = {
    'channel-5200-half-channel-yfac1.1': 'channel-5200-half-channel-yfac1.1',
    'PINN-NN': 'PINN-NN',
    'channel-5200-half-channel': 'baseline',
    'channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission': 'channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission',
    'channel-10000-half-channel-NN-PINN-from-channel-NN-vist-over-y-uv_tot-2nd-submission-t_int-3000': 'nn10000',
}
manifest = {'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': {}}
with tarfile.open(archive) as tar:
    for item in tar:
        parts = Path(item.name).parts
        if not item.isfile() or len(parts) < 2 or '\\' in item.name or '..' in parts:
            continue
        if len(parts) == 2:
            relative = Path(parts[1])
        elif parts[1] in names:
            relative = Path(names[parts[1]], *parts[2:])
        else:
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = tar.extractfile(item).read()
        # Extended path handles the author's long filenames on Windows.
        extended = Path('\\\\?\\' + str(target)) if __import__('os').name == 'nt' else target
        extended.write_bytes(data)
        manifest['files'][str(relative)] = hashlib.sha256(data).hexdigest()
(dest / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'Extracted {len(manifest["files"])} unchanged files; archive {manifest["archive_sha256"]}')
