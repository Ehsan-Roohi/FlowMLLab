"""Fail-fast checks for Week 14's evidence, notebook and source distinctions."""
from pathlib import Path
import hashlib
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'results/week14_validation'
manifest=json.loads((E/'manifest.json').read_text())
for name,sha in manifest['files'].items():
    assert hashlib.sha256((E/name).read_bytes()).hexdigest()==sha,name
s=json.loads((E/'summary.json').read_text())
assert s['runs']['train_ck']['epochs']==1000
assert not s['runs']['baseline']['converged']
assert s['runs']['pinn']['converged']
assert not s['runs']['nn10000']['converged']
assert not s['runs']['nn5200']['converged']
assert s['runs']['inverse']['epochs']==200000
assert s['runs']['inverse']['reported_final_total_loss'] > s['runs']['inverse']['reported_minimum_total_loss']
assert all(not v['matches_released_target'] for v in s['balance_regeneration'].values())
nb=json.loads((ROOT/'notebooks/week14/W14_pyCALC_RANS_PINN_NN.ipynb').read_text(encoding='utf-8'))
assert nb['nbformat']==4
assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
for cell in nb['cells']:
    if cell['cell_type']=='code':
        assert cell['execution_count'] is not None
        assert all(o['output_type']!='error' for o in cell['outputs'])
a=np.load(E/'teaching_data.npz',allow_pickle=False)
assert np.isfinite(a['baseline_archive']).all()
assert np.isfinite(a['corrected_archive']).all()
assert not np.array_equal(a['baseline_archive'],a['corrected_archive'])
print('PASS: hashes, executed cells, correct baseline identity, and retained negative gates')
