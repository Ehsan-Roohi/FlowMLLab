"""Recompute the recovered eight-case audit; never fit or replace predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'results/week16_lowboom'
R = E / 'reference'
RECOVERED_BLOBS = {
    'frozen_test_predictions.json': '002791765d76c6357a27b9ce4b5e3060060ade44',
    'independent_test.npz': 'd3a595477981217e6878727045bbed7aa2146f76',
    'independent_neural_test.png': '7acc05116893f585d6673d5725056e16b64e16a5',
}


def metrics(predicted, actual, predicted_cd, actual_cd):
    assert predicted.shape == actual.shape and predicted_cd.shape == actual_cd.shape
    assert all(np.isfinite(a).all() for a in [predicted, actual, predicted_cd, actual_cd])
    norms = np.linalg.norm(actual, axis=1)
    peaks = actual.max(axis=1)
    assert (norms > 0).all() and (peaks > 0).all() and (actual_cd > 0).all()
    errors = np.linalg.norm(predicted - actual, axis=1) / norms
    return {
        'wave_relative_l2': float(np.linalg.norm(predicted - actual) / np.linalg.norm(actual)),
        'peak_mean_relative_error': float(np.mean(abs(predicted.max(axis=1) / peaks - 1))),
        'drag_mean_relative_error': float(np.mean(abs(predicted_cd / actual_cd - 1))),
        'per_case_wave_relative_l2': errors.tolist(),
        'worst_case_wave_relative_l2': float(errors.max()),
    }


def build_report(write=True):
    provenance = {}
    for name, expected in RECOVERED_BLOBS.items():
        payload = (R / name).read_bytes()
        blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
        assert blob == expected, f'Recovered evidence changed: {name}'
        provenance[name] = {'github_blob_sha': blob, 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
    frozen = json.loads((R / 'frozen_test_predictions.json').read_text())
    assert hashlib.sha256((E / 'dataset.npz').read_bytes()).hexdigest() == frozen['dataset_sha256']
    with np.load(R / 'independent_test.npz') as finer, np.load(E / 'dataset.npz') as coarse:
        names = list(map(str, coarse['names'][coarse['splits'] == 'test']))
        assert names == frozen['names'] == finer['names'].tolist()
        indices = np.array([np.flatnonzero(coarse['names'] == name).item() for name in names])
        assert np.array_equal(coarse['parameters'][indices], finer['parameters'])
        assert np.array_equal(finer['parameters'], np.asarray(frozen['parameters']))
        assert np.array_equal(coarse['x'], finer['x'])
        assert np.array_equal(finer['predicted'], np.asarray(frozen['predicted_waveforms']))
        assert np.array_equal(finer['predicted_cd'], np.asarray(frozen['predicted_cd']))
        fine_metrics = metrics(finer['predicted'], finer['finer'], finer['predicted_cd'], finer['actual_cd'])
        coarse_metrics = metrics(finer['predicted'], coarse['waveforms'][indices], finer['predicted_cd'], coarse['cd'][indices])
        mesh_change = float(np.linalg.norm(coarse['waveforms'][indices] - finer['finer']) / np.linalg.norm(finer['finer']))
    thresholds = frozen['thresholds']
    checks = {key: fine_metrics[key] < thresholds[key + '_max'] for key in ['wave_relative_l2', 'peak_mean_relative_error', 'drag_mean_relative_error']}
    report = {
        'description': 'Recomputed from recovered frozen audit outputs; no retraining or new CFD performed by this script.',
        'names': names, 'cases': len(names), 'finer_mesh': fine_metrics, 'paired_coarse_mesh': coarse_metrics,
        'coarse_to_finer_wave_relative_l2': mesh_change,
        'thresholds': thresholds, 'checks': checks, 'passed': all(checks.values()),
        'identity_checks': {'original_blob_hashes': True, 'original_dataset_hash': True, 'same_geometry_order_and_parameters': True, 'same_sampling_coordinates': True, 'frozen_predictions_exact': True},
        'provenance': provenance,
        'scope': 'Eight held-out geometries in the same two-parameter family; numerical CFD comparison, not experimental neural validation or full-aircraft reproduction.',
        'limitations': [
            'The audit uses a refit of the published architecture. It does not establish bitwise recovery of the original trained model.',
            'The original PCA automatic solver can introduce refit variability; this report uses only archived predictions.',
            'Recovered compact arrays establish the reported numerical differences. They do not independently establish solver convergence or mesh-cell counts without the original run logs.',
            'The archived protocol records frozen predictions and thresholds. File contents alone do not independently prove the chronological order of CFD generation and prediction freezing.',
            'Aggregate acceptance does not require every individual waveform error to be below 10%; the worst-case error is reported explicitly.',
        ],
    }
    if write:
        (R / 'neural_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    report = build_report()
    print(json.dumps({key: report[key] for key in ['cases', 'finer_mesh', 'paired_coarse_mesh', 'coarse_to_finer_wave_relative_l2', 'passed']}, indent=2))
