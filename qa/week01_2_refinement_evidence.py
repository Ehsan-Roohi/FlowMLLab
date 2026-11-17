"""Read the verified Week 1.2 refinement evidence without launching CFD."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

METHODS = ('simple', 'piso', 'pimple')


def load_cases(root):
    """Require all 12 matching FV configurations and check their native arrays."""
    root = Path(root)
    cases = []
    for re in (400, 1000):
        for method in METHODS:
            for cells in (32, 64):
                path = root / 'results/week01_2_formulations' / f'{method}-re{re}-n{cells}.json'
                record = json.loads(path.read_text(encoding='utf-8'))
                archive = path.with_suffix('.npz')
                if hashlib.sha256(archive.read_bytes()).hexdigest() != record['field_sha256']:
                    raise ValueError(f'Archive hash mismatch: {archive}')
                with np.load(archive, allow_pickle=False) as saved:
                    arrays = {k: saved[k] for k in saved.files}
                cfg = record['config']
                assert (cfg['reynolds'], cfg['method'], cfg['cells']) == (re, method, cells)
                assert cfg['convection'] == 'central'
                assert cfg['dt'] == .05 and cfg['pressure_correctors'] == 2
                assert cfg['outer_correctors'] == 3 and cfg['tolerance'] == 1e-6
                assert record['converged'] and record['status'] == 'converged'
                assert record['momentum_residual_linf'] < 1e-6
                assert record['continuity_linf'] < 1e-9
                assert all(np.isfinite(a).all() for a in arrays.values())
                cases.append({**record, **arrays})
    return cases


def teaching_tables(root):
    cases = load_cases(root)
    rows = []
    for r in cases:
        c = r['config']
        rows.append({
            'Re': c['reynolds'], 'method': c['method'].upper(), 'cells': c['cells'],
            'u error %': 100*r['ghia_u_relative_l2'],
            'v error %': 100*r.get('ghia_v_relative_l2_without_x09063', r['ghia_v_relative_l2']),
            'momentum Linf': r['momentum_residual_linf'], 'divergence Linf': r['continuity_linf'],
            'steps / iterations': r['iterations'], 'pressure solves': r['pressure_solves'],
            'seconds': r['runtime_seconds'],
            'timing group': 'Unity 2026-11' if r.get('slurm_job_id') else 'Local 2026-10',
        })
    table = pd.DataFrame(rows)
    pairs = []
    for (re, method), group in table.groupby(['Re', 'method'], sort=False):
        coarse = group.loc[group.cells == 32].iloc[0]
        fine = group.loc[group.cells == 64].iloc[0]
        r = next(r for r in cases if (r['config']['reynolds'], r['config']['method'].upper(), r['config']['cells']) == (re, method, 64))
        pairs.append({'Re': re, 'method': method,
                      'u: error64 / error32': fine['u error %']/coarse['u error %'],
                      'v: error64 / error32': fine['v error %']/coarse['v error %'],
                      '64-cell u and v below 5%': bool(max(fine['u error %'], fine['v error %']) < 5),
                      '64-cell vortex x/L': r['vortex_x'], '64-cell vortex y/L': r['vortex_y']})
    legacy = pd.read_csv(Path(root)/'results/week01_2_student_review/fresh_summary.csv')
    convection = legacy[(legacy.method == 'simple') & (legacy.reynolds == 400) & (legacy.cells == 32)].copy()
    assert set(convection.convection) == {'central', 'upwind'}
    convection['u error %'] = 100*convection.ghia_u_relative_l2
    convection['v error %'] = 100*convection.ghia_v_relative_l2
    convection = convection[['convection', 'u error %', 'v error %', 'momentum_residual_linf', 'continuity_linf']]
    return table, pd.DataFrame(pairs), convection
