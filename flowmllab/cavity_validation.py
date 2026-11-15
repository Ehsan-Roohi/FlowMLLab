"""Display retained CFD solutions against Ghia, with field identity and grid evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
from . import ghia

# These are actual retained solver runs, not fits to the reference points.
CFD_RUNS = {
    100: ('week01_2_pressure_velocity', 'simple', (32, 64)),
    400: ('week01_2_formulations', 'streamfunction-vorticity', (32, 64, 128)),
    1000: ('week01_2_formulations', 'streamfunction-vorticity', (32, 64, 128)),
}
COLORS = ('#0072B2', '#D55E00', '#009E73')


def load_evidence(root):
    """Hash-check native archives and extract profiles without fitting to Ghia.

    MAC u/v are sampled on their native faces, then the prescribed wall values
    are appended. Nodal FD fields are interpolated in both coordinates. No
    cell-centre extrapolation and no substitution of a neural prediction occur.
    """
    root = Path(root)
    cases = []
    for re, (folder, method, grids) in CFD_RUNS.items():
        for cells in grids:
            path = root / f'results/{folder}/{method}-re{re}-n{cells}.npz'
            meta_path = path.with_suffix('.json')
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != meta['field_sha256']:
                raise ValueError(f'Retained CFD archive changed: {path}')
            config = meta['config']
            if (config['reynolds'], config['cells'], config['method']) != (re, cells, method):
                raise ValueError(f'CFD configuration mismatch: {meta_path}')
            with np.load(path, allow_pickle=False) as z:
                field = {k: z[k] for k in z.files}
            if not all(np.isfinite(a).all() for a in field.values()):
                raise ValueError(f'Nonfinite retained CFD: {path}')
            x, y = field['x'], field['y']
            if field['u'].shape != (len(y), len(x)) or field['v'].shape != field['u'].shape:
                raise ValueError('Expected field arrays in [y, x] order')
            if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
                raise ValueError('CFD coordinates must increase')
            if method == 'simple':
                from .pressure_velocity import centerlines
                yp, up, xp, vp = centerlines(field)
                native = 'momentum_residual_linf'
            else:
                yp, xp = y, x
                up = np.array([np.interp(.5, x, row) for row in field['u']])
                vp = np.array([np.interp(.5, y, col) for col in field['v'].T])
                native = 'steady_vorticity_linf'
            scores = {c: ghia.profile_errors(re, c, t, val)
                      for c, t, val in [('u', yp, up), ('v', xp, vp)]}
            # Reproduce the original all-point metrics, including the flagged
            # Re=400 printed entry; the screened metric is reported separately.
            for c in ('u', 'v'):
                if not np.isclose(scores[c]['published_relative_l2'],
                                  meta[f'ghia_{c}_relative_l2'], rtol=2e-11, atol=1e-13):
                    raise ValueError(f'CFD/Ghia metric does not reproduce: {path}, {c}')
            if re == 400 and not np.isclose(scores['v']['screened_relative_l2'],
                    meta['ghia_v_relative_l2_without_x09063'], rtol=2e-11, atol=1e-13):
                raise ValueError('Re=400 reference-sensitivity metric changed')
            cases.append(dict(Re=re, cells=cells, method=method, metadata=meta,
                              field=field, profiles={'u': (yp, up), 'v': (xp, vp)},
                              scores=scores, source=path.relative_to(root).as_posix(),
                              sha256=digest, metadata_sha256_lf=hashlib.sha256(meta_path.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                              recorded_steady_defect=meta[native], native_equation=native))
    return cases


def score_rows(cases):
    rows = []
    for case in cases:
        row = {k: case[k] for k in ('Re', 'cells', 'method', 'source', 'sha256')}
        row.update(recorded_converged=case['metadata']['converged'],
                   native_equation=case['native_equation'],
                   recorded_steady_defect=case['recorded_steady_defect'],
                   recorded_continuity_linf=case['metadata']['continuity_linf'])
        for c in ('u', 'v'):
            row.update({f'{c}_{key}': value for key, value in case['scores'][c].items()})
        rows.append(row)
    return rows


def finest(cases):
    return [max((c for c in cases if c['Re'] == re), key=lambda c: c['cells'])
            for re in sorted({c['Re'] for c in cases})]


def _reference(ax, re, component, color='k'):
    ref = ghia.reference(re)
    t = ref['y' if component == 'u' else 'x']
    keep = ghia.valid_mask(re, component)
    ax.plot(t[keep], ref[component][keep], 'o', color=color, ms=4,
            mfc='white', mew=1, ls='none', label='Ghia samples')
    if (~keep).any():
        ax.plot(t[~keep], ref[component][~keep], 'x', color='#b62626', ms=8,
                mew=1.5, ls='none', label='Flagged printed entry')


def _labels(ax, component, error=False):
    ax.set(xlabel='y/L at x/L = 0.5' if component == 'u' else 'x/L at y/L = 0.5',
           ylabel=f'({component} CFD - {component} Ghia)/U' if error else f'{component}/U',
           xlim=(0, 1))
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    if error:
        ax.axhline(0, color='k', lw=.7)


def _legend(fig, axes, ncol=3, bottom=.02):
    handles, labels = [], []
    for ax in np.asarray(axes).flat:
        h, l = ax.get_legend_handles_labels()
        for item, label in zip(h, l):
            if label not in labels:
                handles.append(item); labels.append(label)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, bottom),
               ncol=ncol, frameon=False, fontsize=10)


def plot_profiles(cases, reynolds):
    """Two native-grid profiles and signed errors at the actual Ghia samples."""
    import matplotlib.pyplot as plt
    selected = [c for c in cases if c['Re'] == reynolds]
    if not selected:
        raise ValueError(f'No retained CFD field for Re={reynolds}')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.9))
    for col, component in enumerate(('u', 'v')):
        _reference(axes[0, col], reynolds, component)
        ref = ghia.reference(reynolds)
        t = ref['y' if component == 'u' else 'x']
        keep = ghia.valid_mask(reynolds, component)
        order = np.argsort(t)
        for case, color, style in zip(selected, COLORS, (':', '--', '-')):
            positions, values = case['profiles'][component]
            method = 'SIMPLE' if case['method'] == 'simple' else 'FD'
            axes[0, col].plot(positions, values, style, color=color, lw=1.65,
                              label=f'FlowMLLab {method}, {case["cells"]} cells')
            delta = np.interp(t, positions, values) - ref[component]
            valid = order[keep[order]]
            axes[1, col].plot(t[valid], delta[valid], style+'o', color=color, lw=1.2, ms=3)
            if (~keep).any():
                axes[1, col].plot(t[~keep], delta[~keep], 'x', color=color, ms=7, mew=1.5)
        _labels(axes[0, col], component)
        _labels(axes[1, col], component, error=True)
    _legend(fig, axes[0], ncol=3)
    fig.subplots_adjust(left=.09, right=.98, top=.97, bottom=.20, hspace=.46, wspace=.29)
    return fig


def plot_overview(cases):
    """Homepage comparison of actual finest retained CFD and reference samples."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    handles = []
    for case, color in zip(finest(cases), COLORS):
        re = case['Re']
        method = 'SIMPLE' if case['method'] == 'simple' else 'FD'
        handles.append(Line2D([], [], color=color, lw=1.7,
                              label=f'Re={re}: {method}, {case["cells"]} cells'))
        for ax, component in zip(axes, ('u', 'v')):
            pos, val = case['profiles'][component]
            ax.plot(pos, val, color=color, lw=1.7)
            _reference(ax, re, component, color)
    for ax, component in zip(axes, ('u', 'v')):
        _labels(ax, component)
    handles.extend([Line2D([], [], color='k', marker='o', mfc='white', ls='none', label='Ghia samples'),
                    Line2D([], [], color='#b62626', marker='x', ls='none', label='Flagged printed entry')])
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
               frameon=False, ncol=3, fontsize=10)
    fig.subplots_adjust(left=.085, right=.98, top=.96, bottom=.29, wspace=.28)
    return fig


def plot_streamlines(cases):
    """Black streamlines of the retained CFD; these are not Ghia data."""
    import matplotlib.pyplot as plt
    selected = finest(cases)
    fig, axes = plt.subplots(1, len(selected), figsize=(10, 3.5))
    for ax, case in zip(np.atleast_1d(axes), selected):
        f = case['field']
        ax.streamplot(f['x'], f['y'], f['u'], f['v'], color='black',
                      density=1.65, linewidth=.55, arrowsize=.7)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal',
               xlabel=f'x/L  (Re={case["Re"]})', ylabel='y/L')
    fig.subplots_adjust(left=.06, right=.975, bottom=.18, top=.98, wspace=.28)
    return fig
