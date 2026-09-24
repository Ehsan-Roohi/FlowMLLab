"""Load and audit the retained portable checkpoint; this module never trains a model.

Run ``python qa/week16/freeze_model.py`` to reproduce checkpoint_audit.json.
The filename is historical: the checkpoint is already frozen, and is never rewritten.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from independent_audit_report import metrics

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'results/week16_lowboom'
R = E / 'reference'


class FrozenSurrogate:
    """NumPy-only inference: scale -> tanh MLP -> unscale -> POD/log-drag."""
    def __init__(self, path=R / 'model_checkpoint.npz'):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.arrays = {key: data[key].copy() for key in data.files}
        a = self.arrays
        assert int(a['format_version']) == 1 and int(a['layers']) == 3
        for key, shape in {'input_mean': (2,), 'input_scale': (2,),
                           'output_mean': (13,), 'output_scale': (13,),
                           'pca_mean': (561,), 'pca_components': (12, 561),
                           'weight_0': (2, 32), 'bias_0': (32,),
                           'weight_1': (32, 32), 'bias_1': (32,),
                           'weight_2': (32, 13), 'bias_2': (13,),
                           'x': (561,), 'training_names': (24,)}.items():
            assert a[key].shape == shape, (key, a[key].shape)
            if a[key].dtype.kind in 'fiu':
                assert np.isfinite(a[key]).all(), key
        assert (a['input_scale'] > 0).all() and (a['output_scale'] > 0).all()
        assert (np.diff(a['x']) > 0).all()

    def predict(self, parameters):
        a = self.arrays
        x = np.atleast_2d(np.asarray(parameters, dtype=float))
        if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
            raise ValueError('Expected finite geometry parameters with shape (N, 2).')
        z = (x - a['input_mean']) / a['input_scale']
        for i in range(3):
            z = z @ a[f'weight_{i}'] + a[f'bias_{i}']
            if i < 2:
                z = np.tanh(z)
        z = z * a['output_scale'] + a['output_mean']
        return z[:, :12] @ a['pca_components'] + a['pca_mean'], np.exp(z[:, 12])


def fingerprint(path):
    payload = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(payload).hexdigest(),
            'github_blob_sha': hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest(),
            'bytes': len(payload)}


def audit(write=True):
    model = FrozenSurrogate()
    a = model.arrays
    assert fingerprint(model.path)['github_blob_sha'] == '11337027584a29e44c78ef52e8dfd9c3cc034109', 'Unexpected checkpoint: declare and audit a new model identity.'
    with np.load(E / 'dataset.npz', allow_pickle=False) as data, np.load(R / 'independent_test.npz', allow_pickle=False) as fine:
        tr = data['splits'] == 'train'
        assert tr.sum() == 24
        assert np.array_equal(data['names'][tr], a['training_names'])
        assert np.array_equal(data['x'], a['x']) and np.array_equal(fine['x'], a['x'])
        for value, reference in [(a['input_mean'], data['parameters'][tr].mean(axis=0)),
                                 (a['input_scale'], data['parameters'][tr].std(axis=0)),
                                 (a['pca_mean'], data['waveforms'][tr].mean(axis=0))]:
            np.testing.assert_allclose(value, reference, atol=1e-13, rtol=1e-11)
        components = a['pca_components']
        np.testing.assert_allclose(components @ components.T, np.eye(12), atol=1e-12)
        # Verify the retained POD subspace against training data; no MLP fitting.
        centered = data['waveforms'][tr] - a['pca_mean']
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        subspace_error = float(np.linalg.norm(components.T @ components - vt[:12].T @ vt[:12]))
        assert subspace_error < 1e-9
        z_train = np.column_stack([centered @ components.T, np.log(data['cd'][tr])])
        np.testing.assert_allclose(a['output_mean'], z_train.mean(axis=0), atol=1e-13, rtol=1e-11)
        np.testing.assert_allclose(a['output_scale'], z_train.std(axis=0), atol=1e-13, rtol=1e-11)
        split_metrics = {}
        for split in ['train', 'validation', 'test', 'extrapolation']:
            mask = data['splits'] == split
            wave, drag = model.predict(data['parameters'][mask])
            single = [model.predict(x) for x in data['parameters'][mask]]
            np.testing.assert_allclose(wave, np.vstack([s[0] for s in single]), atol=1e-13, rtol=1e-11)
            np.testing.assert_allclose(drag, np.concatenate([s[1] for s in single]), atol=1e-13, rtol=1e-11)
            split_metrics[split] = {'cases': int(mask.sum()), **metrics(wave, data['waveforms'][mask], drag, data['cd'][mask])}
        test = data['splits'] == 'test'
        assert np.array_equal(data['names'][test], fine['names'])
        assert np.array_equal(data['parameters'][test], fine['parameters'])
        assert not set(a['training_names'].tolist()) & set(fine['names'].tolist())
        wave, drag = model.predict(fine['parameters'])
        finer_metrics = metrics(wave, fine['finer'], drag, fine['actual_cd'])
        prediction_difference = float(np.linalg.norm(wave - fine['predicted']) / np.linalg.norm(fine['predicted']))
    thresholds = {'wave_relative_l2': .10, 'peak_mean_relative_error': .10, 'drag_mean_relative_error': .10}
    checks = {key: finer_metrics[key] < limit for key, limit in thresholds.items()}
    report = {
        'model_identity': 'Retained portable full-SVD refit checkpoint; distinct from the historical frozen-prediction audit model.',
        'checkpoint': fingerprint(model.path),
        'dataset': fingerprint(E / 'dataset.npz'),
        'finer_reference': fingerprint(R / 'independent_test.npz'),
        'training_names': a['training_names'].tolist(),
        'architecture': {'input': 2, 'hidden': [32, 32], 'hidden_activation': 'tanh', 'output': 13,
                         'output_activation': 'identity', 'pod_modes': 12, 'waveform_samples': 561,
                         'drag_transform': 'log during fitting, exponential during inference'},
        'checks': {'array_shapes_and_finiteness': True, 'training_ids_match': True,
                   'training_only_scalers_and_pod_mean': True, 'training_pod_subspace_matches': True,
                   'batch_and_single_inference_agree': True, 'finer_geometry_ids_and_coordinates_match': True},
        'pod_training_subspace_difference_frobenius': subspace_error,
        'original_mesh': split_metrics, 'finer_mesh': finer_metrics,
        'thresholds': thresholds, 'numerical_accuracy_checks': checks,
        'passed': all(checks.values()),
        'relative_difference_from_historical_frozen_wave_predictions': prediction_difference,
        'scope': 'Reproducible inference and numerical evaluation on retained CFD arrays within the two-parameter teaching family.',
        'limitations': [
            'This checkpoint is a later refit. It is not the missing original historical checkpoint or a model from the Chinese paper.',
            'The finer CFD labels existed before this checkpoint was retained. This is a retrospective evaluation, not a new prospective blind test.',
            'Training scalers, POD and identifiers are verified against training data. Checkpoint arrays alone cannot independently establish the complete historical fitting procedure or absence of prior model selection.',
            'The 10% aggregate thresholds are inherited teaching criteria, not experimental uncertainty limits; individual cases may exceed 10%.',
            'Compact CFD arrays do not establish original solver convergence without the associated raw logs.',
            'No experimental neural validation, NASA-geometry generalization, full-aircraft replication or ground-noise prediction is established.',
            'Historical optimized designs belong to the model used to propose them; this refit does not retroactively validate its own optimized candidates.'
        ],
    }
    if write:
        (R / 'checkpoint_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    result = audit()
    print(json.dumps({k: result[k] for k in ['checkpoint', 'finer_mesh', 'passed']}, indent=2))
