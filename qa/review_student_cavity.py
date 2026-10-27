"""Recompute selected student-report cases without changing archived benchmarks.

Use --run to calculate fresh fields. The default only rebuilds figures from
hash-verified outputs of this script. Timings are single local observations.
"""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import platform
import os
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab.pressure_velocity import centerlines, GHIA_X, GHIA_Y
from flowmllab.pressure_velocity_experiments import (
    ExperimentalConfig, run_experiment, benchmark_valid_mask, REFERENCE,
)

OUT = ROOT / 'results/week01_2_student_review'
SOURCES = ['flowmllab/pressure_velocity.py', 'flowmllab/pressure_velocity_experiments.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases():
    for re in (400, 1000):
        for method in ('simple', 'piso', 'pimple'):
            yield ExperimentalConfig(method=method, reynolds=re, cells=32,
                velocity_relaxation=.5, pressure_relaxation=.2,
                tolerance=1e-6, max_iterations=4000)
        yield ExperimentalConfig(method='simple', reynolds=re, cells=64,
            velocity_relaxation=.5, pressure_relaxation=.2,
            tolerance=1e-6, max_iterations=4000)
    yield ExperimentalConfig(method='simple', reynolds=400, cells=32,
        convection='upwind', velocity_relaxation=.5, pressure_relaxation=.2,
        tolerance=1e-6, max_iterations=4000)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {p: digest(ROOT/p) for p in SOURCES}
    for config in selected_cases():
        stem = f'{config.method}-re{int(config.reynolds)}-n{config.cells}-{config.convection}'
        path = OUT/(stem+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['source_sha256'] != source_hashes or saved['config'] != asdict(config):
                raise ValueError('Existing run uses different solver sources: '+stem)
            if saved['field_sha256'] != digest(path.with_suffix('.npz')):
                raise ValueError('Existing field checksum mismatch: '+stem)
            print('Verified existing run', stem, flush=True)
            continue
        print('Running', stem, flush=True)
        result = run_experiment(config)
        arrays = {k:v for k,v in result.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(path.with_suffix('.npz'), **arrays)
        record = {k:v for k,v in result.items() if not isinstance(v, np.ndarray)}
        record.update(source_sha256=source_hashes,
            field_sha256=digest(path.with_suffix('.npz')),
            provenance='Fresh author-side calculation, 26 October 2026; distinct from submitted report measurements.',
            environment={'python':platform.python_version(), 'numpy':np.__version__,
                         'scipy':scipy.__version__, 'platform':platform.platform(),
                         'blas_thread_environment':{key:os.environ.get(key,'unset') for key in
                             ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}})
        path.write_text(json.dumps(record, indent=2)+'\n')
        print(stem, 'converged', result['converged'], 'iterations', result['iterations'],
              'seconds', round(result['runtime_seconds'], 2),
              'errors', result['ghia_u_relative_l2'], result['ghia_v_relative_l2'], flush=True)


def load():
    rows = []
    for path in sorted(OUT.glob('*-re*-n*-*.json')):
        record = json.loads(path.read_text())
        if digest(path.with_suffix('.npz')) != record['field_sha256']:
            raise ValueError('Field checksum mismatch: '+str(path))
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as archive:
            rows.append({**record, **{k:archive[k] for k in archive.files}})
    if len(rows) != 9:
        raise ValueError(f'Expected all nine selected cases, found {len(rows)}')
    return rows


def figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman', 'font.size':11,
        'mathtext.fontset':'stix', 'axes.spines.top':False, 'axes.spines.right':False})
    def finish(fig, name):
        fig.savefig(OUT/(name+'.png'), dpi=170, facecolor='white')
        plt.close(fig)
    colors = {'simple':'#174c90', 'piso':'#bc2424', 'pimple':'#167043'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for row, re in enumerate((400, 1000)):
        for result in rows:
            c = result['config']
            if c['reynolds'] != re or c['convection'] != 'central':
                continue
            y,u,x,v = centerlines(result)
            style = {'color':colors[c['method']], 'linestyle':'--' if c['cells']==64 else '-',
                     'lw':2 if c['cells']==64 else 1.2,
                     'label':f"{c['method'].upper()}, N={c['cells']}"}
            axes[row,0].plot(u,y,**style); axes[row,1].plot(x,v,**style)
        axes[row,0].plot(REFERENCE[re]['u'],GHIA_Y,'ko',ms=3,label='Ghia (1982)')
        mask=benchmark_valid_mask(re,'v',GHIA_X)
        axes[row,1].plot(np.asarray(GHIA_X)[mask],np.asarray(REFERENCE[re]['v'])[mask],
                         'ko',ms=3,label='Ghia: scored samples')
        if not mask.all():
            axes[row,1].plot(np.asarray(GHIA_X)[~mask],np.asarray(REFERENCE[re]['v'])[~mask],
                'rx',ms=8,label='Published anomaly: not scored')
        axes[row,0].set(xlabel='u / U',ylabel='y / L',title=f'Re={re}: vertical centerline')
        axes[row,1].set(xlabel='x / L',ylabel='v / U',title=f'Re={re}: horizontal centerline')
        for ax in axes[row]:ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Fresh calculations: changing the grid changes the resolved velocity profiles')
    finish(fig,'mesh_and_methods')
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,re in enumerate((400,1000)):
        for result in rows:
            c=result['config']
            if c['reynolds']!=re or c['cells']!=32 or c['convection']!='central':continue
            count=1 if c['method']=='simple' else c['pressure_correctors']*(c['outer_correctors'] if c['method']=='pimple' else 1)
            history=result['history'];work=count*history[:,0]
            axes[row,0].semilogy(work,history[:,3],color=colors[c['method']],label=c['method'].upper())
            axes[row,1].semilogy(work,np.maximum(history[:,4],1e-17),color=colors[c['method']],label=c['method'].upper())
        axes[row,0].axhline(1e-6,color='black',ls=':',label='Momentum target')
        axes[row,0].set(title=f'Re={re}, N=32',ylabel='Steady momentum defect, Linf')
        axes[row,1].set(title=f'Re={re}, N=32',ylabel='Cell divergence, Linf')
        for ax in axes[row]:ax.set_xlabel('Cumulative pressure solves');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Fresh equation histories; machine-precision divergence fluctuations are not mesh error')
    finish(fig,'coupling_histories')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    for result in rows:
        c=result['config']
        if c['method']!='simple' or c['reynolds']!=400:continue
        y,u,x,v=centerlines(result)
        label=f"N={c['cells']}, {c['convection']}"
        color='#bc2424' if c['convection']=='upwind' else ('#167043' if c['cells']==64 else '#174c90')
        for ax,a,b in [(axes[0],u,y),(axes[1],x,v)]:ax.plot(a,b,color=color,label=label)
    axes[0].plot(REFERENCE[400]['u'],GHIA_Y,'ko',ms=3,label='Ghia')
    mask=benchmark_valid_mask(400,'v',GHIA_X)
    axes[1].plot(np.asarray(GHIA_X)[mask],np.asarray(REFERENCE[400]['v'])[mask],'ko',ms=3,label='Ghia: scored samples')
    axes[0].set(xlabel='u / U',ylabel='y / L');axes[1].set(xlabel='x / L',ylabel='v / U')
    for ax in axes:ax.legend(fontsize=9);ax.grid(alpha=.15)
    fig.suptitle('Re=400: first-order upwind versus central convection, with the same SIMPLE solver')
    finish(fig,'convection_profiles')


def summary(rows):
    keys=['method','reynolds','cells','convection','iterations','pressure_solves','runtime_seconds',
          'momentum_residual_linf','continuity_linf','ghia_u_relative_l2','ghia_v_relative_l2']
    with (OUT/'fresh_summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for r in rows:writer.writerow({k:r['config'].get(k,r.get(k)) for k in keys})


def report(rows):
    lines=['# Student cavity study: selected fresh checks', '',
        "The student's October 2026 report supplied 12 central-convection cases.",
        'The table below contains **nine separate author-side calculations** with',
        'the current Python solver: eight corresponding central cases and one',
        'matched upwind case. No CFD package is used.', '',
        '[Review and corrected teaching explanations](../../docs/reviews/student_cavity_20261026.md)',
        ' | [Student-reported measurements](submitted_summary.csv)',
        ' | [Fresh measurements](fresh_summary.csv)',
        ' | [Source report hashes and provenance](provenance.json)',
        ' | [Validation](validation.json)', '',
        '## Configuration and evidence', '',
        'Unit-square two-dimensional laminar cavity, unit lid speed, viscosity 1/Re,',
        'impermeable walls and an unsmoothed moving-lid corner. The configurations',
        'retain dt=0.05, velocity relaxation 0.5, pressure relaxation 0.2, two PISO',
        'corrections and three PIMPLE outer passes. The steady momentum target is',
        '1e-6; divergence is separately checked below 1e-9. Every JSON records the',
        'full configuration, solver hashes and numerical-library versions.', '',
        'The original report did not include the raw source-run arrays, a complete',
        'configuration or machine specifications. Fresh checks therefore establish',
        'agreement of selected reported numerical results under the explicit settings',
        'above; they do not establish identical execution environments. Four submitted',
        '64-cell PISO/PIMPLE cases remain student-reported. The student report does',
        'not include a streamfunction-vorticity run; that independent formulation',
        'remains available in the existing Week 1.2 comparison.', '',
        '| Method | Re | N | Convection | Iterations | Pressure solves | Local seconds | u error (%) | v error (%) |',
        '| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in sorted(rows,key=lambda r:(r['config']['reynolds'],r['config']['cells'],r['config']['method'],r['config']['convection'])):
        c=r['config']
        lines.append(f"| {c['method'].upper()} | {int(c['reynolds'])} | {c['cells']} | {c['convection']} | {r['iterations']} | {r['pressure_solves']} | {r['runtime_seconds']:.2f} | {100*r['ghia_u_relative_l2']:.4f} | {100*r['ghia_v_relative_l2']:.4f} |")
    lines += ['', 'Runtime is a single sequential local wall-clock observation per case.',
        'It is not a repeat-averaged performance result or a reproduction of the',
        "student's timings. Pressure solves measure one part of the work; they do",
        'not include momentum assembly/factorization or equate costs across grids.', '',
        'Errors use relative L2 at Ghia Table I/II reference positions. At Re=400',
        'the anomalous v sample at x/L=0.9063 is preserved but excluded from the',
        'qualified norm; all-point diagnostics remain in each result JSON.', '',
        '## Velocity profiles: refinement and coupling', '',
        '![Fresh centerline velocities against the scored Ghia samples](mesh_and_methods.png)', '',
        'Solid lines are the three 32-cell coupling methods. The dashed SIMPLE',
        'line uses 64 cells. All curves are computed profiles, and the reference',
        'is shown only at its actual sample positions. The red cross is the',
        'unscored published anomaly. Close method curves at a fixed grid are',
        'expected when they converge to the same discrete steady equations.', '',
        '## Equation histories and coupling work', '',
        '![Momentum and divergence histories against pressure solves](coupling_histories.png)', '',
        'These plots use only the matched 32-cell central cases. Divergence is',
        'already near sparse-solve round-off while momentum continues to change.',
        'Fluctuations around machine precision do not indicate a loss of spatial',
        'accuracy. The final momentum and divergence values are checked directly',
        'from each saved face-velocity and pressure field.', '',
        '## A matched upwind comparison', '',
        '![SIMPLE velocity profiles with central and first-order upwind convection](convection_profiles.png)', '',
        'The central and upwind 32-cell cases share the same Reynolds number,',
        'coupling method, relaxation and stopping targets. The 64-cell central',
        'profile is a separately labeled refinement comparison. Both coarse',
        'calculations converge, so smaller equation defects do not explain the',
        'difference in profile accuracy. This measured case supports the numerical',
        'diffusion explanation; it does not rank every upwind scheme or establish',
        'a general stability advantage.', '',
        '## Reproduce', '',
        'From a complete checkout with the project dependencies installed:', '',
        '```bash',
        '# Rebuild figures and tables from hash-verified retained fields:',
        'python qa/review_student_cavity.py',
        '# Calculate any missing fresh cases (can take several minutes or longer):',
        'python qa/review_student_cavity.py --run',
        '# Check report arithmetic, fresh fields and executed teaching cells:',
        'python qa/validate_student_cavity_review.py',
        '```', '',
        'Existing results are reused only when solver hashes and full configurations',
        'match; field hashes are verified before reuse. To measure new timings,',
        'make a separate output directory and document the new run conditions.',
        'Retained fields and figures here are kept separate from the original course',
        'benchmark archives. The two new notebook code cells were executed separately;',
        'pre-existing notebook sources and outputs are preserved. This revision does',
        'not claim a fresh hosted Colab Run All.', '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    rows=load();summary(rows);figures(rows);report(rows)
