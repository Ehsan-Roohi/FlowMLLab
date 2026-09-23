"""Download the pinned official Linux OpenMP SU2 distribution without sudo."""
import argparse,hashlib,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',default='.tools/week16_su2');args=p.parse_args()
root=Path(args.directory).resolve();root.mkdir(parents=True,exist_ok=True)
url='https://github.com/su2code/SU2/releases/download/v8.5.0/SU2-v8.5.0-linux64-omp.zip'
archive=root/'release.zip'
if not archive.exists():urllib.request.urlretrieve(url,archive)
def extract(path,dest):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not (dest/name).resolve().is_relative_to(dest):raise ValueError('Invalid archive path')
        z.extractall(dest)
extract(archive,root)
inner=root/'linux64-omp.zip'
if inner.exists():extract(inner,root)
exe=root/'bin/SU2_CFD'
expected='0b8b343f41a4c441ab9900a597de6d3b56eca688157e3fdfef9d0f1f1ba01b17'
actual=hashlib.sha256(exe.read_bytes()).hexdigest()
if actual!=expected:raise RuntimeError(f'Unexpected SU2 binary SHA-256: {actual}')
exe.chmod(exe.stat().st_mode|0o111)
print(f'export SU2_CFD={exe}')
print('Verified SU2 8.5.0 official Linux OpenMP binary:',actual)
