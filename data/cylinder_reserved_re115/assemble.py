"""Reassemble distributed archive bytes and verify SHA-256 without reading fields."""
from pathlib import Path
import hashlib
import json
root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
target = root/manifest['file']
if target.exists():
    if hashlib.sha256(target.read_bytes()).hexdigest() != manifest['sha256']:
        raise SystemExit('Existing target checksum mismatch; refusing overwrite')
    print('Verified existing archive:', target.name)
else:
    chunks = []
    for part in manifest['parts']:
        data = (root/part['file']).read_bytes()
        if hashlib.sha256(data).hexdigest() != part['sha256']:
            raise SystemExit('Part checksum mismatch: '+part['file'])
        chunks.append(data)
    data = b''.join(chunks)
    if hashlib.sha256(data).hexdigest() != manifest['sha256']:
        raise SystemExit('Archive checksum mismatch')
    with target.open('xb') as stream:
        stream.write(data)
    print('Reassembled and verified:', target.name)
