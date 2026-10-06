"""Run in a fresh Colab cell: upload the supplied portable course archive."""
from pathlib import Path
import subprocess,sys,zipfile
from google.colab import files

uploaded=files.upload()
if len(uploaded)!=1:raise ValueError('Upload exactly one portable FlowMLLab course ZIP')
archive=next(iter(uploaded));destination=Path('/content/course').resolve()
destination.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for entry in z.infolist():
        target=(destination/entry.filename).resolve()
        if not target.is_relative_to(destination):raise ValueError('Unsafe archive path')
    z.extractall(destination)
roots=list(destination.rglob('pyproject.toml'))
if len(roots)!=1:raise ValueError('Expected one course project')
root=roots[0].parent
subprocess.run([sys.executable,'-m','pip','install','-e',str(root)+'[transformer]'],check=True)
print('Course root:',root,'; start notebooks from that directory.')
