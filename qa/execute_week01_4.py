"""Execute the standalone CPU notebook and preserve a compact verification receipt."""
from pathlib import Path
import os, sys, json, hashlib, time

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'tmp/week01_4'
TMP.mkdir(parents=True, exist_ok=True)
for key, part in [('MPLCONFIGDIR','mpl'),('IPYTHONDIR','ipython'),
                  ('JUPYTER_RUNTIME_DIR','jupyter-runtime'),('JUPYTER_CONFIG_DIR','jupyter-config')]:
    path = TMP / part
    path.mkdir(exist_ok=True)
    os.environ[key] = str(path)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['FLOWMLLAB_W14_OUTPUT'] = str(ROOT / 'results/week01_4_spectral_dg')
os.environ['JUPYTER_PATH'] = str(Path(sys.prefix) / 'share/jupyter')

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = ROOT / 'notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb'
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name='python3')
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
started = time.perf_counter()
client = NotebookClient(nb, timeout=180, km=km, resources={'metadata':{'path':str(path.parent)}})
client.execute()
elapsed = time.perf_counter() - started
nbformat.validate(nb)
errors = [o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type') == 'error']
assert not errors
nbformat.write(nb, path)
receipt = {
    'notebook': str(path.relative_to(ROOT)), 'executed_on':'2026-11-20',
    'python':sys.version, 'kernel_executable':sys.executable,
    'elapsed_run_all_seconds':elapsed,
    'code_cells':sum(c.cell_type=='code' for c in nb.cells),
    'markdown_cells':sum(c.cell_type=='markdown' for c in nb.cells),
    'error_outputs':len(errors),
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'executed_code_sha256':hashlib.sha256('\n'.join(c.source for c in nb.cells if c.cell_type=='code').encode()).hexdigest(),
    'claim':'Fresh local CPU execution; no hosted Colab execution claimed.'
}
(ROOT / 'qa/week01_4_execution.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
