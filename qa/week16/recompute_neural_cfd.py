"""Regenerate the retained checkpoint's eight CFD labels with complete run evidence.

These repeat known test geometries. They close the missing raw-log gap, not the
need for genuinely unseen geometries in a prospective generalization test.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
import cfd
from analyze import analyze, load_csv
from freeze_model import FrozenSurrogate
from independent_audit_report import metrics
from seeb_reference import check_su2_mesh

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'results/week16_lowboom'
R = E / 'reference'
SOURCES = ['cfd.py', 'analyze.py', 'freeze_model.py', 'independent_audit_report.py',
           'seeb_reference.py', 'recompute_neural_cfd.py']
RAW = ['mesh.su2', 'flow.cfg', 'metadata.json', 'history.csv', 'solver.log',
       'restart_flow.csv', 'surface_flow.csv', 'metrics.json', 'extracted.npz', 'prediction.json']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity():
    return {'dataset_sha256': digest(E/'dataset.npz'),
            'checkpoint_sha256': digest(R/'model_checkpoint.npz'),
            'source_sha256': {name: digest(ROOT/'qa/week16'/name) for name in SOURCES}}


def case_data(index):
    with np.load(E/'dataset.npz', allow_pickle=False) as data:
        ids = np.flatnonzero(data['splits'] == 'test')
        assert len(ids) == 8 and 0 <= index < 8
        i = ids[index]
        return str(data['names'][i]), data['parameters'][i].copy(), data['x'].copy()


def verify_raw(folder):
    """Recheck physical and convergence quantities from the actual CSV records."""
    m = json.loads((folder/'metrics.json').read_text())
    meta = json.loads((folder/'metadata.json').read_text())
    assert meta['returncode'] == 0 and m['returncode'] == 0
    assert meta['level'] == 2 and meta['mach'] == 1.8 and meta['axisymmetric']
    for key in ['mesh.su2', 'flow.cfg']:
        assert digest(folder/key) == meta[key+'_sha256']
    mesh = check_su2_mesh(folder/'mesh.su2', expected_cells=meta['cells'])
    history = load_csv(folder/'history.csv')
    residual = float(history['rms[Rho]'][-1])
    drop = float(history['rms[Rho]'][0] - residual)
    tail = history['CD'][-100:]
    assert len(tail) == 100 and np.isfinite(tail).all() and abs(tail.mean()) > 0
    stability = float(np.ptp(tail)/abs(tail.mean()))
    assert residual <= -9 and drop >= 5 and stability < 1e-4
    assert m['converged']
    for key, actual in [('density_residual_log10', residual), ('residual_drop', drop),
                        ('drag_tail_relative_range', stability)]:
        assert np.isclose(m[key], actual, atol=1e-12, rtol=1e-12)
    minima = {}
    for filename in ['restart_flow.csv', 'surface_flow.csv']:
        u = load_csv(folder/filename)
        rho = u['Density']
        assert np.isfinite(rho).all() and (rho > 0).all()
        p = .4*(u['Energy']-.5*(u['Momentum_x']**2+u['Momentum_y']**2)/rho)
        assert np.isfinite(p).all() and (p > 0).all()
        minima[filename] = {'min_density': float(rho.min()), 'min_pressure': float(p.min())}
    assert np.isfinite(m['cd_pressure']) and m['cd_pressure'] > 0
    return {'mesh_integrity': mesh, 'density_residual_log10': residual,
            'residual_drop': drop, 'drag_tail_relative_range': stability,
            'positive_fields': minima, 'iterations': len(history), 'passed': True}


def run_case(index):
    name, params, x = case_data(index)
    folder = E/'runs'/f'checkpoint_test_{index:03d}'
    # Exclusive creation prevents silent overwriting of either predictions or CFD.
    folder.mkdir(parents=True, exist_ok=False)
    wave, drag = FrozenSurrogate().predict(params)
    pred = {'case_index': index, 'test_name': name, 'parameters': params.tolist(),
            'x': x.tolist(), 'waveform': wave[0].tolist(), 'cd': float(drag[0]),
            'recorded_before_solver_utc': datetime.now(timezone.utc).isoformat(),
            'identity': identity(), 'scope': 'Repeated known test geometry; retrospective label recomputation.'}
    (folder/'prediction.json').write_text(json.dumps(pred, indent=2)+'\n')
    cfd.run(folder.name, a=float(params[0]), b=float(params[1]), level=2,
            mach=1.8, iterations=4000)
    analyze(folder)
    checks = verify_raw(folder)
    evidence = {'identity': identity(), 'case_index': index, 'test_name': name,
                'checks': checks, 'raw_sha256': {f: digest(folder/f) for f in RAW}}
    assert pred['identity'] == evidence['identity'], 'Inputs changed during run'
    (folder/'run_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'case': name, **checks}, indent=2))


def report(write=True):
    model = FrozenSurrogate()
    names, params, waves, actual, predicted_cd, actual_cd, evidence_rows = [], [], [], [], [], [], []
    for index in range(8):
        name, par, x = case_data(index)
        folder = E/'runs'/f'checkpoint_test_{index:03d}'
        evidence = json.loads((folder/'run_evidence.json').read_text())
        assert evidence['identity'] == identity()
        assert evidence['case_index'] == index and evidence['test_name'] == name
        assert set(evidence['raw_sha256']) == set(RAW)
        for f, sha in evidence['raw_sha256'].items():
            assert digest(folder/f) == sha, f'Changed raw evidence: {folder/f}'
        checks = verify_raw(folder)
        assert checks == evidence['checks']
        pred = json.loads((folder/'prediction.json').read_text())
        assert pred['identity'] == identity() and pred['test_name'] == name
        assert pred['case_index'] == index
        assert np.array_equal(par, pred['parameters']) and np.array_equal(x, pred['x'])
        w, d = model.predict(par)
        # Same stored weights, allowing only backend floating-point roundoff.
        np.testing.assert_allclose(w[0], pred['waveform'], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(d[0], pred['cd'], rtol=1e-12, atol=1e-14)
        with np.load(folder/'extracted.npz', allow_pickle=False) as a:
            assert np.array_equal(x, a['x']) and a['radii'][1] == .5
            target = a['cp'][1].copy()
            assert np.isfinite(target).all()
        m = json.loads((folder/'metrics.json').read_text())
        assert m['a'] == float(par[0]) and m['b'] == float(par[1])
        names.append(name);params.append(par);waves.append(pred['waveform']);actual.append(target)
        predicted_cd.append(pred['cd']);actual_cd.append(m['cd_pressure'])
        evidence_rows.append({'name': name, 'run': folder.name, 'checks': checks,
                              'run_evidence_sha256': digest(folder/'run_evidence.json'),
                              'raw_sha256': evidence['raw_sha256']})
    scores = metrics(np.array(waves), np.array(actual), np.array(predicted_cd), np.array(actual_cd))
    thresholds = {'wave_relative_l2': .1, 'peak_mean_relative_error': .1, 'drag_mean_relative_error': .1}
    checks = {key: scores[key] < limit for key, limit in thresholds.items()}
    result = {'identity': identity(), 'cases': 8, 'names': names, 'metrics': scores,
              'thresholds': thresholds, 'checks': checks, 'passed': all(checks.values()),
              'run_evidence': evidence_rows,
              'scope': 'Retained portable checkpoint versus eight newly rerun level-2 CFD solutions, with complete raw convergence/field evidence.',
              'limitations': ['These are the same eight known test geometries, not new unseen geometry tests.',
                             'Predictions were saved before each rerun, but earlier CFD labels were already available; this is retrospective repeated-label validation.',
                             'No training is performed, and neither the historical audit nor its predictions are replaced.',
                             'Passing aggregate criteria does not imply every case is below 10%.',
                             'No experimental neural validation, full-aircraft replication, atmospheric propagation or ground-noise result.']}
    arrays=dict(names=np.asarray(names), parameters=np.asarray(params), x=x,
                predicted=np.asarray(waves), actual=np.asarray(actual),
                predicted_cd=np.asarray(predicted_cd), actual_cd=np.asarray(actual_cd))
    if write:
        R.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(R/'recomputed_checkpoint_test.npz', **arrays)
    else:
        with np.load(R/'recomputed_checkpoint_test.npz', allow_pickle=False) as saved:
            for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    result['compact_arrays_sha256'] = digest(R/'recomputed_checkpoint_test.npz')
    if write:
        (R/'recomputed_checkpoint_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(4, 2, figsize=(11, 11), sharex=True)
        for i, ax in enumerate(axs.flat):
            ax.plot(x, actual[i], 'k', label='New SU2 level-2 CFD')
            ax.plot(x, waves[i], '--', label='Retained neural checkpoint')
            ax.set(title=f'{names[i]}: waveform error {100*scores["per_case_wave_relative_l2"][i]:.2f}%', ylabel='Cp')
            ax.grid(alpha=.2)
        for ax in axs[-1]:ax.set_xlabel('x/L at r/L = 0.5')
        axs[0,0].legend(fontsize=8)
        fig.tight_layout();fig.savefig(R/'recomputed_checkpoint_validation.png', dpi=160);plt.close(fig)
    print(json.dumps({'metrics': scores, 'checks': checks, 'passed': result['passed']}, indent=2))
    if not result['passed']:
        raise RuntimeError('Recomputed checkpoint comparison did not pass declared aggregate gates')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case-index', type=int, choices=range(8))
    group.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:report()
    else:run_case(args.case_index)
