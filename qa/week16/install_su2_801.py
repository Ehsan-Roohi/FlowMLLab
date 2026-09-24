"""Install the pinned official SU2 8.0.1 Linux binary for controlled version checks.

This separate installer never replaces the Week 16 SU2 8.5.0 installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://github.com/su2code/SU2/releases/download/v8.0.1/SU2-v8.0.1-linux64.zip'
ASSET_SHA256 = '844e3d1aa679431dad1e33622c95cd0774d7471a2ba955feb7650513864b4a38'
BINARY_SHA256 = '9d75d6b1b862e6d3470c80ecab2cfa09fac3d54943f6d6d610bc36cd7ef71351'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def install(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = root/'SU2-v8.0.1-linux64.zip'
    if not archive.exists():
        temporary = root/'download.part'
        urllib.request.urlretrieve(URL, temporary)
        if sha256(temporary) != ASSET_SHA256:
            raise RuntimeError('Official SU2 8.0.1 asset checksum mismatch')
        temporary.replace(archive)
    if sha256(archive) != ASSET_SHA256:
        raise RuntimeError('Retained SU2 8.0.1 archive checksum mismatch')
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            if not (root/name).resolve().is_relative_to(root):
                raise ValueError('Archive member escapes installation directory')
        if package.testzip() is not None:
            raise RuntimeError('Corrupt SU2 archive')
        package.extractall(root)
    binary = root/'bin/SU2_CFD'
    if sha256(binary) != BINARY_SHA256:
        raise RuntimeError('Unexpected SU2 8.0.1 executable checksum')
    binary.chmod(binary.stat().st_mode | 0o111)
    manifest = {'version': '8.0.1', 'asset_url': URL, 'asset_sha256': ASSET_SHA256,
                'binary_sha256': BINARY_SHA256, 'binary_relative_path': 'bin/SU2_CFD',
                'purpose': 'Controlled solver-version comparison; distinct from the original 8.5.0 campaign.'}
    (root/'installation.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'export SU2_CFD={binary}')
    print('Verified official SU2 8.0.1 Linux executable:', BINARY_SHA256)
    return binary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='.tools/week16_su2_801')
    install(parser.parse_args().directory)
