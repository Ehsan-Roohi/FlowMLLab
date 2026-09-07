"""Run the frozen Week 7.2 POD-space Kalman-filter comparison."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import importlib.metadata
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from flowmllab.field_metrics import field_metrics
from flowmllab.modal_experiments import load_cases
from flowmllab.state_estimation import (fit_reduced_kalman, kalman_filter,
    reconstruct_states, open_loop, sensor_only, pointwise_interval_coverage)

PLAN = {
    'case': 110, 'field': 'v', 'train': [0, 160], 'validation': [160, 210],
    'test': [210, 281], 'ranks': [4, 6, 8], 'sensor_counts': [8, 16, 32],
    'noise_fraction_of_training_rms': 0.10, 'seeds': [31, 32, 33, 34, 35],
    'process_covariance_scales': [1, 10, 100, 1000],
    'selection': 'minimum mean validation Kalman relative L2; tie favors fewer sensors, lower rank, then lower inflation',
}


def score(pred, truth, case):
    area = float(np.diff(case['x'])[0] * np.diff(case['y'])[0])
    return field_metrics(pred, truth, np.full(truth.shape[1:], area))


def experiment(cases):
    case = cases[PLAN['case']]; field = case[PLAN['field']]
    flat = field.reshape(len(field), -1); train = flat[:160]
    sigma = PLAN['noise_fraction_of_training_rms'] * float(np.sqrt(np.mean(train**2)))
    candidates = []
    for rank in PLAN['ranks']:
        for count in PLAN['sensor_counts']:
            if count < rank: continue
            for scale in PLAN['process_covariance_scales']:
                model = fit_reduced_kalman(train, rank, count, sigma, scale)
                values = []
                for seed in PLAN['seeds']:
                    rng = np.random.default_rng(seed)
                    observations = flat[160:210, model.sensor_indices] + rng.normal(0, sigma, (50, count))
                    states, covs, _ = kalman_filter(model, observations)
                    values.append(score(reconstruct_states(model, states).reshape((-1, *field.shape[1:])), field[160:210], case)['relative_l2'])
                candidates.append({'rank': rank, 'sensor_count': count, 'process_covariance_scale': scale,
                    'validation_relative_l2_mean': float(np.mean(values)), 'validation_relative_l2_std': float(np.std(values, ddof=1))})
    chosen = min(candidates, key=lambda row: (row['validation_relative_l2_mean'], row['sensor_count'], row['rank'], row['process_covariance_scale']))
    model = fit_reduced_kalman(train, chosen['rank'], chosen['sensor_count'], sigma, chosen['process_covariance_scale'])
    horizon = len(flat) - 160; truth = field[160:]
    methods = {name: [] for name in ('kalman', 'open_loop', 'persistence', 'sensor_only')}
    coverage = []
    examples = None
    for seed in PLAN['seeds']:
        rng = np.random.default_rng(seed + 10000)
        observations = flat[160:, model.sensor_indices] + rng.normal(0, sigma, (horizon, chosen['sensor_count']))
        states, covs, innovations = kalman_filter(model, observations)
        predictions = {
            'kalman': reconstruct_states(model, states).reshape(truth.shape),
            'open_loop': open_loop(model, horizon).reshape(truth.shape),
            'persistence': np.repeat(field[159:160], horizon, axis=0),
            'sensor_only': sensor_only(model, observations).reshape(truth.shape),
        }
        for name, pred in predictions.items():
            methods[name].append({'seed': seed, 'validation': score(pred[:50], truth[:50], case),
                'test': score(pred[50:], truth[50:], case)})
        cov, width = pointwise_interval_coverage(model, states[50:], covs[50:], flat[210:])
        coverage.append({'seed': seed, 'test_pointwise_95_coverage': cov, 'mean_95_interval_width': width,
            'normalized_innovation_rms': float(np.sqrt(np.mean(innovations**2)) / sigma)})
        if examples is None: examples = predictions
    return {'plan': PLAN, 'selected': chosen, 'noise_sigma': sigma, 'candidates': candidates,
        'methods': methods, 'coverage': coverage,
        'claim_boundary': 'Previously inspected Re110 trajectory with synthetic sensor noise; causal filtering, not smoothing or a new blind CFD result.'}, model, examples


def make_figures(out, cases, result, model, examples):
    case = cases[110]; truth = case['v'][160:]; extent = [case['x'][0], case['x'][-1], case['y'][0], case['y'][-1]]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 220})
    names = ['LBM reference', 'Kalman filter', 'Open-loop DMD', 'Sensor-only']
    fields = [truth[-1], examples['kalman'][-1], examples['open_loop'][-1], examples['sensor_only'][-1]]
    vmax = float(np.max(np.abs(truth[-1])))
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 8), layout='compressed')
    for ax, name, values in zip(axes, names, fields):
        im = ax.contourf(case['x'], case['y'], values, levels=np.linspace(-vmax, vmax, 33), cmap='RdBu_r', extend='both')
        ax.set(title=name, xlabel='x/D', ylabel='y/D', aspect='equal')
        if name == 'Kalman filter':
            iy, ix = np.unravel_index(model.sensor_indices, truth.shape[1:]); ax.scatter(case['x'][ix], case['y'][iy], s=12, facecolors='none', edgecolors='black')
    fig.colorbar(im, ax=axes.tolist(), label='v/U', shrink=.75)
    fig.suptitle('Week 7.2: final test frame, common color scale')
    fig.savefig(out/'state_estimation_fields.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for name in ('kalman','open_loop','persistence','sensor_only'):
        mean = np.mean([100*r['test']['relative_l2'] for r in result['methods'][name]])
        std = np.std([100*r['test']['relative_l2'] for r in result['methods'][name]], ddof=1)
        ax.bar(name, mean, yerr=std, capsize=4)
    ax.set(ylabel='Test relative L2 (%)', title='Five sensor-noise seeds; mean +/- sample SD')
    fig.savefig(out/'state_estimation_scores.png'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    start = time.perf_counter(); cases = load_cases(ROOT/'data/modal_labs')
    result, model, examples = experiment(cases); result['elapsed_seconds'] = time.perf_counter() - start
    result['environment'] = {'python': platform.python_version(), **{p: importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','matplotlib')}}
    result['source_hashes'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ('flowmllab/state_estimation.py','qa/run_week72_state_estimation.py','data/modal_labs/manifest.json')}
    (args.output/'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    make_figures(args.output, cases, result, model, examples)
    print(json.dumps({'selected': result['selected'], 'test_mean': {name: float(np.mean([r['test']['relative_l2'] for r in rows])) for name, rows in result['methods'].items()}, 'coverage_mean': float(np.mean([r['test_pointwise_95_coverage'] for r in result['coverage']])), 'seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__': main()
