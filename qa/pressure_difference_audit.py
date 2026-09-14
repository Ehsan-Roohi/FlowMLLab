"""Area-weighted pressure differences; no pointwise relative division near zero."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def pressure_audit(x, y, weights, pressures, output):
    xx, yy = np.meshgrid(x, y)
    assert all(np.isfinite(p).all() for p in pressures)
    assert np.all(weights > 0)
    corners = (yy > 2.1) & ((xx < .1) | (xx > .9))
    masks = {'whole': np.ones(xx.shape, dtype=bool), 'top_corners_0p1W': corners,
             'outside_top_corners': ~corners, 'lower_y_lt_0p5W': yy < .5,
             'interior_0p1W_from_walls': (xx > .1) & (xx < .9) & (yy > .1) & (yy < 2.1)}
    pairs = [(2, 0, 'PINN - Nektar++'), (1, 0, 'OpenFOAM - Nektar++'),
             (2, 1, 'PINN - OpenFOAM')]
    result = {}
    differences = []
    for i, j, label in pairs:
        d = pressures[i] - pressures[j]
        differences.append(d)
        total = np.sum(weights * d*d)
        k = np.unravel_index(np.argmax(abs(d)), d.shape)
        result[label] = {'maximum_absolute': float(abs(d[k])),
                         'maximum_location': [float(xx[k]), float(yy[k])], 'regions': {}}
        for name, m in masks.items():
            e = np.sum(weights[m]*d[m]**2)
            denom = np.sum(weights[m]*pressures[j][m]**2)
            result[label]['regions'][name] = {
                'area_fraction': float(weights[m].sum()/weights.sum()),
                'rmse_p_over_rhoU2': float(np.sqrt(e/weights[m].sum())),
                'relative_L2': float(np.sqrt(e/denom)) if denom > 0 else None,
                'squared_difference_fraction': float(e/total) if total > 0 else 0.0}
    plt.rcParams.update({'font.size': 16, 'axes.titlesize': 16})
    limit = max(np.max(abs(d)) for d in differences)
    fig, axes = plt.subplots(1, 3, figsize=(14, 10), layout='constrained')
    for ax, d, (_, _, label) in zip(axes, differences, pairs):
        im = ax.pcolormesh(x, y, d, cmap='RdBu_r', vmin=-limit, vmax=limit, shading='auto')
        ax.set(title=label, xlabel='x/W', ylabel='y/W', aspect='equal', xlim=(0,1), ylim=(0,2.2))
    fig.colorbar(im, ax=axes, shrink=.7, label=r'$\Delta p/(\rho U^2)$ | linear, no clipping')
    fig.suptitle('Re = 1000 | D/W = 2.2 | retained PINN checkpoint 55118\nPressure differences: common area-weighted mean-zero gauge')
    fig.supxlabel('Differences are not certified PINN errors: lid profiles differ near corners.', fontsize=13)
    for ext in ('png', 'pdf'): fig.savefig(output/f'pressure_difference_linear.{ext}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    norm = SymLogNorm(linthresh=1e-4, vmin=-limit, vmax=limit)
    for col, (d, (_, _, label)) in enumerate(zip(differences, pairs)):
        for row, bounds in enumerate([(0, .1), (.9, 1)]):
            ax = axes[row, col]
            im = ax.pcolormesh(x, y, d, cmap='RdBu_r', norm=norm, shading='auto')
            ax.set(xlim=bounds, ylim=(2.1, 2.2), aspect='equal', xlabel='x/W', ylabel='y/W',
                   title=label + (' | upper left' if row == 0 else ' | upper right'))
    fig.colorbar(im, ax=axes, shrink=.8, label=r'$\Delta p/(\rho U^2)$ | symmetric log')
    fig.suptitle('Corner pressure differences | Re = 1000 | D/W = 2.2')
    fig.supxlabel('Common mean-zero gauge; shared scale, linear within +/-0.0001; no clipping. Lid profiles differ.', fontsize=12)
    for ext in ('png', 'pdf'): fig.savefig(output/f'pressure_difference_corner_zoom.{ext}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    for p, label in zip(pressures, ['Nektar++', 'OpenFOAM', 'PINN']):
        axes[0].plot(p[:, np.argmin(abs(x-.5))], y, label=label)
        axes[1].plot(x, p[np.argmin(abs(y-2.1)), :], label=label)
    axes[0].set(xlabel=r'$p/(\rho U^2)$ (mean removed)', ylabel='y/W', title='Near x/W = 0.5')
    axes[1].set(xlabel='x/W', ylabel=r'$p/(\rho U^2)$ (mean removed)', title='Near y/W = 2.1')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    for ext in ('png', 'pdf'): fig.savefig(output/f'pressure_difference_profiles.{ext}', dpi=220)
    plt.close(fig)
    (output/'pressure_difference_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
