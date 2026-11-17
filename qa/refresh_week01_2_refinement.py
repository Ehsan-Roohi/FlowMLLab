"""Audit and illustrate the complete 32/64-cell steady FV comparison.

No solver trajectories are launched. Recompute native residuals and Ghia
errors from hash-checked fields before writing any teaching artifact.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flowmllab.pressure_velocity import _residual, divergence, centerlines, GHIA_X, GHIA_Y
from flowmllab.pressure_velocity_experiments import ExperimentalConfig, REFERENCE, benchmark_valid_mask
from qa.week01_2_refinement_evidence import load_cases, teaching_tables


def source_bytes_match(data, digest):
    """Retain the executed raw hash while allowing Git's LF/CRLF checkout rule."""
    if hashlib.sha256(data).hexdigest() == digest:
        return 'exact bytes'
    lf = data.replace(b'\r\n', b'\n')
    if any(hashlib.sha256(candidate).hexdigest() == digest
           for candidate in (lf, lf.replace(b'\n', b'\r\n'))):
        return 'identical source after LF/CRLF conversion only'
    raise ValueError('Source differs from the executed source beyond line endings')


def audit(cases):
    expected={(re, method, n) for re in (400,1000) for method in ('simple','piso','pimple') for n in (32,64)}
    actual={(r['config']['reynolds'],r['config']['method'],r['config']['cells']) for r in cases}
    assert len(cases)==12 and actual==expected, 'The complete configuration-matched matrix is required.'
    checked = []
    for r in cases:
        cfg = ExperimentalConfig(**r['config'])
        # Native face velocities, not interpolated plotting fields.
        mom = _residual(r['u_faces'], r['v_faces'], r['p'], cfg)
        div = float(np.max(np.abs(divergence(r['u_faces'], r['v_faces'], 1/cfg.cells))))
        assert mom < 1e-6 and div < 1e-9
        assert np.isclose(mom, r['momentum_residual_linf'], rtol=1e-6, atol=1e-12)
        assert np.isclose(div, r['continuity_linf'], rtol=1e-4, atol=1e-12)
        source_checks = {}
        if r.get('slurm_job_id'):
            for path, digest in r['source_sha256'].items():
                source_checks[path] = source_bytes_match((ROOT/path).read_bytes(), digest)
        y, u, x, v = centerlines(r)
        scores = {}
        for component, locations, coordinate, values in [('u', GHIA_Y, y, u), ('v', GHIA_X, x, v)]:
            ref = np.asarray(REFERENCE[cfg.reynolds][component])
            mask = benchmark_valid_mask(cfg.reynolds, component, locations)
            diff = np.interp(locations, coordinate, values) - ref
            value = float(np.linalg.norm(diff[mask])/np.linalg.norm(ref[mask]))
            key = 'ghia_v_relative_l2_without_x09063' if cfg.reynolds == 400 and component == 'v' else f'ghia_{component}_relative_l2'
            assert np.isclose(value, r[key], rtol=1e-9, atol=1e-12)
            scores[component] = value
        checked.append({'case': f'{cfg.method}-re{cfg.reynolds}-n{cfg.cells}',
                        'momentum_linf_recomputed': mom, 'divergence_linf_recomputed': div,
                        'ghia_relative_l2_recomputed': scores, 'field_sha256': r['field_sha256'],
                        'executed_source_checks': source_checks})
    differences = []
    for re in (400, 1000):
        for n in (32, 64):
            rr = [r for r in cases if (r['config']['reynolds'], r['config']['cells']) == (re, n)]
            ref = next(r for r in rr if r['config']['method'] == 'simple')
            for r in rr:
                if r is ref:
                    continue
                relative = float(np.sqrt(sum(np.sum((r[k]-ref[k])**2) for k in ('u_faces','v_faces')) /
                                         sum(np.sum(ref[k]**2) for k in ('u_faces','v_faces'))))
                assert relative < 1e-4, (re, n, r['config']['method'], relative)
                differences.append({'Re': re, 'cells': n, 'method': r['config']['method'],
                                    'velocity_relative_l2_vs_simple': relative})
    return {'native_cases_checked': checked, 'fixed_mesh_method_differences': differences,
            'all_four_new_cases_from_rest_on_unity': True,
            'timing': 'Different hosts/groups retained; no cross-host speed ranking.',
            'reference': 'Ghia velocity samples; Re400 v at x=0.9063 omitted only from qualified score.'}


def error_figure(cases):
    from qa.refresh_week01_2_figures import plt, save, COLORS
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.7), sharey='row', layout='constrained')
    for col, re in enumerate((400, 1000)):
        for r in cases:
            c = r['config']
            if c['reynolds'] != re:
                continue
            y, u, x, v = centerlines(r)
            color = COLORS[('simple','piso','pimple').index(c['method'])]
            style = '-' if c['cells'] == 32 else '--'
            for row, (component, at, coord, values) in enumerate((('u',GHIA_Y,y,u),('v',GHIA_X,x,v))):
                ref = np.asarray(REFERENCE[re][component]); at = np.asarray(at)
                delta = np.interp(at, coord, values)-ref
                mask = benchmark_valid_mask(re, component, at)
                delta[~mask] = np.nan  # Keep a real gap at the excluded sample.
                order = np.argsort(at)
                axes[row,col].plot(at[order], delta[order], color=color, ls=style, lw=1.45,
                                   marker='o', ms=2.6, alpha=.85)
        axes[0,col].set_title(f'Re = {re}', fontsize=12)
        for row, component in enumerate(('u','v')):
            axes[row,col].axhline(0, c='.5', lw=.7)
            axes[row,col].set(xlim=(0,1), xlabel='y/L' if row == 0 else 'x/L',
                              ylabel=rf'$({component}_{{CFD}}-{component}_{{Ghia}})/U$')
            axes[row,col].grid(alpha=.17)
    handles = [Line2D([], [], c=color, lw=1.7, label=method) for color,method in
               zip(COLORS[:3], ('SIMPLE','PISO','PIMPLE'))]
    handles += [Line2D([], [], c='black', ls=style, lw=1.5, label=f'{n} cells/side')
                for style,n in (('-',32),('--',64))]
    fig.legend(handles=handles, loc='outside lower center', ncols=3, fontsize=11, frameon=False)
    save(fig, 'coupling_refinement_errors')


def update_report():
    table, pairs, _ = teaching_tables(ROOT)
    path = ROOT/'results/week01_2_formulations/README.md'
    text = path.read_text(encoding='utf-8').split('<!-- complete-refinement -->')[0].rstrip()
    notice='[Complete 32/64-cell FV comparison and new Unity cases](#complete-3264-cell-fv-refinement). The historical timing study is preserved below.'
    if notice not in text:
        title, rest=text.split('\n',1)
        text=title+'\n\n'+notice+'\n'+rest
    lines = ['<!-- complete-refinement -->', '## Complete 32/64-cell FV refinement', '',
             'The earlier tables and timing figures above describe the October local-host study. '
             'Four additional cases complete PISO/PIMPLE at Re=400 and 1000 on 64 cells. '
             'They were run from rest on Unity, using the unchanged spatial equations and steady gates. '
             'No model is fitted to Ghia in these solver runs.', '',
             '| Re | Method | Cells/side | u error % | Qualified v error % | Solver seconds | Timing group |',
             '|---:|---|---:|---:|---:|---:|---|']
    for _, r in table.iterrows():
        lines.append(f'| {r["Re"]} | {r["method"]} | {r["cells"]} | {r["u error %"]:.3f} | '
                     f'{r["v error %"]:.3f} | {r["seconds"]:.3f} | {r["timing group"]} |')
    lines += ['', 'Do not rank local and Unity timings as algorithm speedups. All are single samples; '
              'the Unity array may run cases concurrently on shared CPU nodes. Each JSON records the host, '
              'versions, source hashes, Slurm ID and full configuration. Native momentum and continuity '
              'defects are recomputed from face fields in [the verification record](refinement_verification.json).', '',
              'At fixed mesh, converged SIMPLE/PISO/PIMPLE velocities nearly coincide. Refinement changes '
              'the profile accuracy much more than changing the coupling algorithm. The Re400 qualified '
              'v score omits the predeclared printed anomaly; it remains in the raw reference and all-point metric. '
              'Two grids do not establish grid independence.', '',
              '![Signed centerline discrepancies at Re400 and Re1000](../../lectures/source/week01_2_assets/coupling_refinement_errors.png)', '',
              '[Case table](refinement_cases.csv) · [Refinement ratios and sampled vortex locations](refinement_pairs.csv) · '
              '[Controlled convection comparison](convection_comparison.csv).', '',
              'Reproduce this retained-data audit with `python qa/refresh_week01_2_refinement.py`, then rebuild '
              'the lecture and execute `python qa/validate_week01_2_figure_guide.py`. Fresh runs use '
              '`qa/run_week01_2_refinement.py --case 0 --output <new-directory>` for case IDs 0–3; '
              'the runner refuses to overwrite existing results. All normal notebook CFD switches remain off.',
              '<!-- /complete-refinement -->']
    path.write_text(text+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cases = load_cases(ROOT)
    report = audit(cases)
    from qa.refresh_week01_2_figures import archived_rows, profiles
    rows = archived_rows()
    for re in (400,1000):
        profiles(rows,re)
    error_figure(cases)
    dest = ROOT/'results/week01_2_formulations'
    for name, table in zip(('refinement_cases','refinement_pairs','convection_comparison'), teaching_tables(ROOT)):
        table.to_csv(dest/(name+'.csv'), index=False)
    table, _, _ = teaching_tables(ROOT)
    lines = [r'\begin{center}\small', r'\begin{tabular}{rlrrrr}\toprule',
             r'$Re$ & Method & $E_u(32)$ & $E_u(64)$ & $E_v(32)$ & $E_v(64)$\\',
             r' & & \multicolumn{4}{c}{Qualified profile error (\%)}\\\midrule']
    for (re, method), group in table.groupby(['Re','method'],sort=False):
        a = group.loc[group.cells == 32].iloc[0]
        b = group.loc[group.cells == 64].iloc[0]
        lines.append(f'{re} & {method} & {a["u error %"]:.3f} & {b["u error %"]:.3f} & '
                     f'{a["v error %"]:.3f} & {b["v error %"]:.3f}'+r'\\')
    lines.extend([r'\bottomrule\end{tabular}\end{center}'])
    (ROOT/'lectures/source/week01_2_refinement_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                              ['flowmllab/pressure_velocity.py','flowmllab/pressure_velocity_experiments.py',
                               'qa/week01_2_refinement_evidence.py','qa/refresh_week01_2_refinement.py']}
    (dest/'refinement_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    update_report()
    print(json.dumps({'verified_cases':len(cases),'maximum_fixed_mesh_method_difference':
                      max(r['velocity_relative_l2_vs_simple'] for r in report['fixed_mesh_method_differences'])}))


if __name__ == '__main__':
    main()
