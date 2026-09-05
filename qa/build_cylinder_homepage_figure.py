"""Plot retained Mach-8.5 fields without interpolating across omitted samples."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from flowmllab.hypersonic_cylinder import (
    load_cylinder_teaching_data, casewise_split_masks,
    case_interpolation_baseline, relative_l2,
)


def main():
    root = Path(__file__).resolve().parents[1]
    data = load_cylinder_teaching_data(root)
    masks = casewise_split_masks(data.mach_inf)
    selected = np.isclose(data.mach_inf, 8.5)
    truth = data.targets[selected]
    prediction = case_interpolation_baseline(data, masks['train'], selected)
    source_rows = data.source_row[selected]
    source_axis = np.linspace(0, 399, 50, dtype=int)
    row = np.searchsorted(source_axis, source_rows // 400)
    col = np.searchsorted(source_axis, source_rows % 400)
    x = np.array([np.median(data.x[selected][col == i]) for i in range(50)])
    y = np.array([np.median(data.y[selected][row == i]) for i in range(50)])
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    assert np.allclose(x[col], data.x[selected])
    assert np.allclose(y[row], data.y[selected])
    errors = relative_l2(truth, prediction)
    labels = ['Local Mach', 'Temperature (source TOV)', 'Pressure (source P)']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), layout='constrained')
    for j, label in enumerate(labels):
        for i, values in enumerate((truth[:, j], np.abs(prediction[:, j] - truth[:, j]))):
            field = np.full((50, 50), np.nan)
            field[row, col] = values
            cmap = plt.get_cmap('viridis' if i == 0 else 'magma').copy()
            cmap.set_bad('#cbd5e1')
            ax = axes[i, j]
            artist = ax.pcolormesh(x, y, np.ma.masked_invalid(field), shading='nearest', cmap=cmap, rasterized=True)
            fig.colorbar(artist, ax=ax, shrink=.78, pad=.02)
            ax.set(aspect='equal', xlabel='x (source coordinates)', ylabel='y (source coordinates)')
            ax.set_title(label if i == 0 else f'Absolute interpolation error\nRelative L2: {100*errors[j]:.3f}%', fontsize=10)
    fig.suptitle('Rarefied cylinder | freestream Mach 8.5\nDSMC teaching fields and whole-case interpolation', fontsize=17, fontweight='bold')
    fig.supxlabel('Gray: omitted solid/sentinel samples. Baseline uses Mach 8 and 9; no Mach-8.5 training data.', fontsize=10)
    output = root / 'results/hypersonic_cylinder_week7_1'
    fig.savefig(output / 'cylinder_homepage.png', dpi=170, facecolor='white')
    plt.close(fig)
    print(json.dumps({'case': 8.5, 'valid_points': len(truth), 'masked_points': 2500-len(truth),
                      'relative_l2': errors.tolist(), 'output': str(output/'cylinder_homepage.png')}))


if __name__ == '__main__':
    main()
