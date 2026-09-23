"""Plot retained NASA measurements and LAVA submission without running CFD."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'cases/week16_lowboom/reference'
OUTPUT = ROOT / 'results/week16_lowboom/reference'


def load_reference():
    experiments = {}
    for label, filename, shift in [
        ('NASA run 195', 'shiftavg_195_219-221.39.out', 124.942 + 26.2945),
        ('NASA run 553', 'shiftavg_553_578-580.out', 124.748 + 26.2945),
    ]:
        data = np.loadtxt(SOURCE / filename)
        experiments[label] = {'x': data[:, 0] + shift, 'pressure': data[:, 1], 'uncertainty': data[:, 4]}
    rows = []
    active = False
    filename = 'Sozer_LAVA_AUSMPWplus_minmod_inviscid_axisymmetric_curvilinear_330K_SEEB.plt'
    for line in (SOURCE / filename).read_text().splitlines():
        if line.strip().startswith('ZONE'):
            if active:
                break
            assert 'H=21.2,PHI=0.0' in line
            active = True
        elif active and line.strip():
            rows.append([float(value) for value in line.split()])
    lava = np.asarray(rows)
    assert lava.ndim == 2 and lava.shape[1] == 4
    # Sort the source coordinates before interpolation; preserve original values.
    lava = lava[np.argsort(lava[:, 0])]
    # Repeated axial samples arise in the submitted extraction; average their pressure.
    x, inverse, count = np.unique(lava[:, 0], return_inverse=True, return_counts=True)
    pressure = np.bincount(inverse, weights=lava[:, 3]) / count
    return experiments, {'x': x, 'pressure': pressure}


def report(write=True):
    experiments, lava = load_reference()
    rows = {}
    for name, curve in experiments.items():
        mask = (curve['x'] >= 25) & (curve['x'] <= 46)
        x, actual = curve['x'][mask], curve['pressure'][mask]
        assert len(x) > 1 and x.min() >= lava['x'].min() and x.max() <= lava['x'].max()
        predicted = np.interp(x, lava['x'], lava['pressure'])
        rows[name] = {
            'points': len(x),
            'wave_relative_l2': float(np.linalg.norm(predicted - actual) / np.linalg.norm(actual)),
            'peak_relative_error': float(abs(predicted.max() / actual.max() - 1)),
        }
    result = {'comparison': 'Retained NASA-hosted LAVA CFD versus NASA wind-tunnel measurements; not a new SU2 validation.', 'window_inches': [25, 46], 'ordinate': 'delta_p/p_infinity', 'LAVA_zone': 'H=21.2,PHI=0.0', 'metrics': rows, 'alignment': 'Original NASA macro shifts only; no fitted amplitude or position.', 'interpolation': 'LAVA repeated axial coordinates averaged, then linear interpolation to experimental points.'}
    if write:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / 'nasa_retained_reference.json').write_text(json.dumps(result, indent=2) + '\n')
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, curve in experiments.items():
            line, = ax.plot(curve['x'], curve['pressure'], label=name, linewidth=1.4)
            ax.fill_between(curve['x'], curve['pressure'] - curve['uncertainty'], curve['pressure'] + curve['uncertainty'], color=line.get_color(), alpha=.15)
        ax.plot(lava['x'], lava['pressure'], 'k--', label='NASA-hosted LAVA CFD, 330k', linewidth=1.5)
        ax.set(xlim=(25, 46), xlabel='Source-aligned x [inches]', ylabel=r'$\Delta p/p_\infty$', title='SEEB-ALR, Mach 1.6, H=21.2 inches')
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(OUTPUT / 'seeb_comparison.png', dpi=180); plt.close(fig)
    return result


if __name__ == '__main__':
    print(json.dumps(report(), indent=2))
