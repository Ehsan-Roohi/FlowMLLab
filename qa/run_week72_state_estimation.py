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
    """Equal physical x/y scales, vector companion, and an actual score table."""
    case = cases[110]; truth = case['v'][-1]
    names = ['(a) LBM reference', '(b) Kalman filter', '(c) Open-loop DMD', '(d) Sensor-only']
    fields = [truth, examples['kalman'][-1], examples['open_loop'][-1], examples['sensor_only'][-1]]
    vmax = max(float(np.max(np.abs(field))) for field in fields)
    errors = [np.abs(fields[1]-truth), np.abs(fields[3]-truth)]
    emax = max(float(error.max()) for error in errors)
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'axes.linewidth': .6, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'savefig.dpi': 320}):
        fig = plt.figure(figsize=(8, 5.5))
        grid = fig.add_gridspec(3, 2, left=.08, right=.84, bottom=.085,
                               top=.95, hspace=.48, wspace=.25)
        for index, (name, values) in enumerate(zip(names, fields)):
            ax = fig.add_subplot(grid[index//2, index%2])
            im = ax.contourf(case['x'], case['y'], values, levels=np.linspace(-vmax, vmax, 49), cmap='RdBu_r')
            ax.set(title=name, xlabel='x/D', ylabel='y/D')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([2, 4, 6, 8, 10, 12]); ax.set_yticks([-2, 0, 2])
        cax = fig.add_axes([.88, .45, .014, .43])
        fig.colorbar(im, cax=cax, label='v/U', ticks=np.linspace(-vmax, vmax, 5), format='%.2f')
        for index, (name, error) in enumerate(zip(['(e) Kalman absolute error', '(f) Sensor-only absolute error'], errors)):
            ax = fig.add_subplot(grid[2, index])
            err_im = ax.contourf(case['x'], case['y'], error, levels=np.linspace(0, emax, 33), cmap='magma')
            ax.set(title=name, xlabel='x/D', ylabel='y/D')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([2, 4, 6, 8, 10, 12]); ax.set_yticks([-2, 0, 2])
        cax = fig.add_axes([.88, .10, .014, .20])
        fig.colorbar(err_im, cax=cax, label='Absolute error in v/U', ticks=np.linspace(0, emax, 4), format='%.3f')
        for extension in ('png', 'svg'):
            fig.savefig(out/f'state_estimation_fields.{extension}', facecolor='white')
        plt.close(fig)
        # Sensor geometry is displayed separately so it cannot obscure the field.
        iy, ix = np.unravel_index(model.sensor_indices, truth.shape)
        sx, sy = case['x'][ix], case['y'][iy]
        fig, axes = plt.subplots(1, 2, figsize=(8, 2), gridspec_kw={'width_ratios': [1.45, 1]})
        dx, dy = float(np.diff(case['x'])[0]), float(np.diff(case['y'])[0])
        zoom = (float(sx.min()-2*dx), float(sx.max()+2*dx),
                float(sy.min()-2*dy), float(sy.max()+2*dy))
        for ax in axes:
            ax.scatter(sx, sy, s=5, c='#176b82', edgecolors='none', zorder=3)
            ax.set(xlabel='x/D', ylabel='y/D')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(color='#e3e9ed', linewidth=.4, zorder=0)
        axes[0].set(title='32 sensors in the wake region',
                    xlim=(case['x'][0],case['x'][-1]), ylim=(case['y'][0],case['y'][-1]))
        from matplotlib.patches import Rectangle
        axes[0].add_patch(Rectangle((zoom[0],zoom[2]),zoom[1]-zoom[0],zoom[3]-zoom[2],
                                   fill=False,edgecolor='#8795a1',linewidth=.7,linestyle='--'))
        axes[1].set(title='Sensor cluster - enlarged', xlim=zoom[:2], ylim=zoom[2:])
        axes[1].collections[0].set_sizes([8])
        fig.subplots_adjust(left=.07,right=.97,bottom=.24,top=.82,wspace=.3)
        for extension in ('png','svg'):
            fig.savefig(out/f'state_estimation_sensors.{extension}', facecolor='white')
        plt.close(fig)
        # Legacy filename remains valid for existing notebooks; its content is a table.
        labels = {'kalman': 'Kalman filter', 'sensor_only': 'Sensor-only POD',
                  'open_loop': 'Open-loop DMD', 'persistence': 'Persistence'}
        rows = []
        for name, label in labels.items():
            values = np.array([100*r['test']['relative_l2'] for r in result['methods'][name]])
            rows.append([label, f'{values.mean():.3f}', f'{values.std(ddof=1):.3f}'])
        fig, ax = plt.subplots(figsize=(8, 2.3)); ax.axis('off')
        table = ax.table(cellText=rows, colLabels=['Method', 'Mean relative L2 (%)', 'Sample SD (pp)'],
                         colWidths=[.42, .33, .25], cellLoc='left', bbox=[0, .09, 1, .83])
        table.auto_set_font_size(False); table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#dbe3e9'); cell.set_linewidth(.45)
            if row == 0:
                cell.set_facecolor('#173b56'); cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#eaf4f5' if row == 1 else ('#f3f6f8' if row%2 else 'white'))
                if col: cell.set_text_props(ha='right')
        fig.subplots_adjust(left=.02, right=.98, bottom=.09, top=.97)
        fig.text(.02, .035, 'Five measurement-noise seeds on one trajectory. SD in percentage points (pp).', fontsize=9, color='#506070')
        fig.savefig(out/'state_estimation_scores.png', facecolor='white'); plt.close(fig)


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
