"""Beginner-facing controls and neutral descriptions of retained evidence."""
from pathlib import Path
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]

BASELINE_CODE='''from IPython.display import Markdown
from datetime import datetime, timezone
from uuid import uuid4
import hashlib, json

RUN_NEW = False
METHOD = "all"  # all three FV methods, or simple / piso / pimple
CELLS = 32
BASE_DT = 0.05                 # PISO/PIMPLE physical marching interval; ignored by SIMPLE
BASE_VELOCITY_RELAXATION = 0.7  # alpha_u: used by SIMPLE
BASE_PRESSURE_RELAXATION = 0.3  # alpha_p: used by SIMPLE
BASE_PRESSURE_CORRECTORS = 2   # N_i: PISO/PIMPLE inner corrections
BASE_OUTER_CORRECTORS = 3      # N_o: PIMPLE only
BASE_TOLERANCE = 1e-7          # native FV momentum AND divergence stopping threshold
BASE_MAX_ITERATIONS = 4000
BASE_CHECK_EVERY = 10          # evaluate native stopping tests every this many updates

if RUN_NEW:
    chosen = ["simple", "piso", "pimple"] if METHOD == "all" else [METHOD]
    fresh = [run_cavity(CavityConfig(method=m, cells=CELLS, dt=BASE_DT,
        velocity_relaxation=BASE_VELOCITY_RELAXATION,
        pressure_relaxation=BASE_PRESSURE_RELAXATION,
        pressure_correctors=BASE_PRESSURE_CORRECTORS,
        outer_correctors=BASE_OUTER_CORRECTORS, tolerance=BASE_TOLERANCE,
        max_iterations=BASE_MAX_ITERATIONS, check_every=BASE_CHECK_EVERY), progress=True)
        for m in chosen]
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    folder = ROOT / "output/week01_2_baseline" / tag
    folder.mkdir(parents=True, exist_ok=True)
    for result in fresh:
        stem = result['config']['method'] + f"-n{CELLS}"
        archive = folder / (stem + ".npz")
        np.savez_compressed(archive, **{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        metadata = {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        metadata['field_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
        (folder / (stem + ".json")).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    display(pd.DataFrame([{
        'method':r['config']['method'], 'native converged':r['converged'],
        'iterations':r['iterations'], 'seconds':r['runtime_seconds'],
        'momentum Linf':r['momentum_residual_linf'], 'divergence Linf':r['continuity_linf'],
        'report momentum gate':r['momentum_residual_linf']<1e-6,
        'report divergence gate':r['continuity_linf']<1e-9} for r in fresh]))
    active = [{**r, 'origin':'retained'} for r in recorded
              if r['config']['cells'] != CELLS or r['config']['method'] not in chosen]
    active += [{**r, 'origin':'fresh'} for r in fresh]
    order = ['simple','piso','pimple','streamfunction-vorticity']
    active.sort(key=lambda r:(r['config']['cells'],order.index(r['config']['method'])))
    same_mesh = [r for r in active if r['config']['cells']==CELLS]
    display(pd.DataFrame([{'origin':r['origin'], **r['config']} for r in same_mesh]))
    plain_names = {'simple':'SIMPLE','piso':'PISO','pimple':'PIMPLE',
                   'streamfunction-vorticity':'Streamfunction-vorticity'}
    plot_labels = {r['config']['method']:plain_names[r['config']['method']] +
                   ' [' + r['origin'] + ']' for r in same_mesh}
    provenance = '; '.join(plot_labels.values())
    for figure_i, (name,fig) in enumerate(figures(active,cells=CELLS,labels=plot_labels).items(),1):
        if name in {'runtime','grid_errors'}:
            plt.close(fig)
            continue
        display(fig)
        display(Markdown(f"**Experiment Figure E100-{figure_i}.** Re=100 {name}; h=1/{CELLS}. "
            f"Cases: {provenance}. Timings belong to their respective executions; rerun the "
            "compared methods on the same host before ranking wall time."))
        plt.close(fig)
    print('Saved full configurations and fields in', folder)
else:
    print('Retained results are shown above. Set RUN_NEW=True to launch a new FV solve.')
'''

HIGHER_CODE='''from IPython.display import Markdown
from datetime import datetime, timezone
from dataclasses import asdict
from uuid import uuid4
import hashlib, json

RUN_HIGHER_RE = False
HIGHER_RE = 400
HIGHER_CELLS = 32
HIGHER_METHOD = "all"  # all four formulations, or one method name
HIGHER_SCHEME = "central"  # central / upwind; independent FD supports central only
HIGHER_DT = 0.05           # FV PISO/PIMPLE interval; ignored by steady SIMPLE
HIGHER_VELOCITY_RELAXATION = 0.5  # SIMPLE alpha_u
HIGHER_PRESSURE_RELAXATION = 0.2  # SIMPLE alpha_p
HIGHER_PRESSURE_CORRECTORS = 2    # PISO/PIMPLE inner corrections
HIGHER_OUTER_CORRECTORS = 3       # PIMPLE outer passes
HIGHER_TOLERANCE = 1e-6           # native FV momentum AND divergence threshold
HIGHER_MAX_ITERATIONS = 4000
HIGHER_CHECK_EVERY = 10
HIGHER_FD_DT = 0.001              # independent FD physical time step
HIGHER_FD_MAX_STEPS = 400000

if RUN_HIGHER_RE:
    from flowmllab.pressure_velocity_experiments import ExperimentalConfig, run_experiment
    from flowmllab.streamfunction_experiments import run_streamfunction
    methods = ["simple","piso","pimple","streamfunction-vorticity"] if HIGHER_METHOD=="all" else [HIGHER_METHOD]
    higher_runs, higher_summary = [], []
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    save_root = ROOT / "output/week01_2_higher_re" / tag
    save_root.mkdir(parents=True,exist_ok=True)
    for m in methods:
        if m == 'streamfunction-vorticity':
            attempted = dict(method=m,reynolds=HIGHER_RE,cells=HIGHER_CELLS,
                dt=HIGHER_FD_DT,max_steps=HIGHER_FD_MAX_STEPS,convection=HIGHER_SCHEME,
                relative_update_tolerance=1e-12,check_every=500,consecutive_required=3)
        else:
            attempted = dict(method=m,reynolds=HIGHER_RE,cells=HIGHER_CELLS,
                convection=HIGHER_SCHEME,dt=HIGHER_DT,
                velocity_relaxation=HIGHER_VELOCITY_RELAXATION,
                pressure_relaxation=HIGHER_PRESSURE_RELAXATION,
                pressure_correctors=HIGHER_PRESSURE_CORRECTORS,
                outer_correctors=HIGHER_OUTER_CORRECTORS,tolerance=HIGHER_TOLERANCE,
                max_iterations=HIGHER_MAX_ITERATIONS,check_every=HIGHER_CHECK_EVERY)
        stem = f"{m}-re{HIGHER_RE}-n{HIGHER_CELLS}-{HIGHER_SCHEME}"
        try:
            if m == 'streamfunction-vorticity':
                if HIGHER_SCHEME != 'central':
                    raise ValueError('The independent FD formulation uses central differences; choose central for all four methods.')
                r = run_streamfunction(HIGHER_RE,HIGHER_CELLS,dt=HIGHER_FD_DT,
                                      max_steps=HIGHER_FD_MAX_STEPS,progress=True)
            else:
                config = ExperimentalConfig(**attempted)
                attempted = asdict(config)
                r = run_experiment(config,progress=True)
            higher_runs.append(r)
            summary = {**r['config'], **{k:r.get(k) for k in [
                'converged','iterations','steps','steady_vorticity_linf','runtime_seconds',
                'pressure_solves','momentum_residual_linf','continuity_linf',
                'ghia_u_relative_l2','ghia_v_relative_l2']}}
            summary['report divergence gate'] = r['continuity_linf'] < 1e-9
            if m == 'streamfunction-vorticity':
                summary['report native gate'] = r['update_converged'] and r['steady_vorticity_linf'] < 1e-6
            else:
                summary['report native gate'] = r['momentum_residual_linf'] < 1e-6
            higher_summary.append(summary)
            archive = save_root / (stem+'.npz')
            np.savez_compressed(archive, **{k:v for k,v in r.items() if isinstance(v,np.ndarray)})
            metadata = {k:v for k,v in r.items() if not isinstance(v,np.ndarray)}
            metadata['field_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
            (save_root/(stem+'.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        except Exception as error:
            failure = {'config':attempted,'converged':False,'error':str(error)}
            higher_summary.append({**attempted,'converged':False,'error':str(error)})
            (save_root/(stem+'-failed.json')).write_text(json.dumps(failure,indent=2),encoding='utf-8')
    higher_table = pd.DataFrame(higher_summary)
    display(higher_table)
    higher_table.to_csv(save_root/'summary.csv',index=False)
    print('Saved configurations, successful/nonconverged fields and failure records in',save_root)
    from flowmllab.pressure_velocity_experiment_plots import comparison_figures
    if higher_runs:
        for figure_i,(name,figure) in enumerate(comparison_figures(higher_runs).items(),1):
            display(figure)
            display(Markdown(f"**Experiment Figure EH-{figure_i}.** Fresh Re={HIGHER_RE} {name}; "
                f"h=1/{HIGHER_CELLS}, convection={HIGHER_SCHEME}. Settings and native stopping "
                "definitions are preserved in the case JSON and table; FD and FV norms differ."))
            plt.close(figure)
else:
    print('Retained higher-Re results are displayed above. Set RUN_HIGHER_RE=True to recompute.')
'''

METRICS=r'''### How the benchmark errors are measured

Let $s_j$ denote an actual Ghia reference position, $f_{\rm ref,j}$ its tabulated
velocity and $f_{\rm num}(s_j)$ the computed profile interpolated linearly to
that position. The profile component $f$ is either horizontal or vertical velocity.

$$
E_{{\rm rel},2}=\frac{\sqrt{\sum_j[f_{\rm num}(s_j)-f_{\rm ref,j}]^2}}
{\sqrt{\sum_j f_{\rm ref,j}^{\,2}}},\qquad
E_\infty=\max_j|f_{\rm num}(s_j)-f_{\rm ref,j}|.
$$

Percentage error is $100E_{{\rm rel},2}$. These compare with independent velocity
samples; momentum and divergence residuals instead measure satisfaction of the
discrete equations. The qualified Re=400 vertical-velocity ($v$) profile score omits the
separately marked published point at $x/L=0.9063$; the all-point result is retained.
Axis labels such as $u/U$ denote the dimensional ratios used in the original
nondimensionalization. Stored values already represent those ratios.
Recorded seconds belong to a specified solver call and host; compare accuracy,
spacing and stopping definitions before comparing cost.
'''

HIGHER_MD=r'''## Higher-Reynolds-number investigation

How do resolution, convection interpolation and coupling settings change accuracy
and work at $Re=400$ and $1000$? `ExperimentalConfig` supports those values;
the baseline `CavityConfig` is qualified for Re=100.

**Cost and hardware.** Finer and higher-Re cases can take substantially longer
and require more memory and a stronger CPU. Doubling cells per side quadruples
the number of 2-D cells, and solve cost can grow faster. Begin with 32 cells,
one method and measured runtime before refining. Default figures load retained
results and launch no new CFD.

### An independent formulation

The streamfunction-vorticity method evolves vorticity, obtains streamfunction
from a Poisson equation, reconstructs velocity and recovers pressure afterward.
It provides an independent 2-D incompressible formulation. It uses different
unknown locations, equations and native stopping norms from the FV methods.
Compare physical spacing and centerline errors rather than treating its native
residual as an FV momentum defect.

**Reading the fields.** Black curves follow the computed velocity, even when
the color represents pressure or vorticity. Signed-field colors use symmetric
logarithmic normalization; speed is nonnegative and uses a linear scale.
Negative vorticity means clockwise local rotation. Seeds reveal resolved
recirculation but cannot create an unresolved vortex.

**Required sequence and deliverables**

1. Reproduce Re=100 first. Start Re=400 and 1000 on 32 cells and record settings,
   failed/nonconverged attempts and stopping reasons.
2. Refine to 64 cells; justify the cost before choosing 128. Compare both centerlines
   with the reference for the same Reynolds number.
3. Compare SIMPLE velocity relaxation (0.3/0.5/0.7), PISO inner corrections (1/2/4),
   or PIMPLE outer passes (1/2/3), changing one setting at a time. Keep spatial
   equations and final acceptance requirements fixed; record solve counts.
4. Compare central and first-order upwind at the same spacing and coupling method.
   The independent FD implementation supports central differences only.
5. Plot velocity structure, signed vorticity and zero-mean pressure; locate primary
   and resolved secondary vortices. A smoother field must still pass accuracy checks.

**Separate acceptance checks**

- FV: original steady momentum Linf below $10^{-6}$ and cell divergence below $10^{-9}$.
- FD: relative vorticity update below $10^{-12}$ at three checks, plus recomputed
  steady-vorticity RHS Linf below $10^{-6}$. Also report its independently computed divergence.
- All: finite fields, near-zero mean pressure, both reference-profile errors and
  spatial refinement evidence. The classroom 5% profile target is a chosen criterion.

The FV `tolerance` knob is used for both native stopping tests; it is not a separate
mass-tolerance setting. Report-level acceptance is checked again on returned fields.
`HIGHER_DT` applies to FV PISO/PIMPLE; `HIGHER_FD_DT` controls the independent FD step.

For Re=400, retain the published $v$-profile value at $x/L=0.9063$ but omit
that marked sample from the qualified error norm; report the all-point norm as well.
Ghia velocity tables do not supply an independent pressure benchmark.

The next cell verifies saved field hashes and displays benchmark diagnostics.
The retained field figures follow. The later opt-in cell exposes relaxation,
corrector counts and stopping controls and saves complete returned configurations.

[Measured higher-Re report](https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/results/week01_2_formulations/README.md)
'''

def improve_practical_cells(book,guide,markdown):
    byid={c.id:c for c in book.cells}
    def source(id,value):
        cell=byid[id]
        if cell.source!=value:
            cell.source=value
            if cell.cell_type=='code':cell.outputs=[];cell.execution_count=None
    if '**How to use this notebook.**' not in byid['829339c3'].source:
        byid['829339c3'].source+='\n\n**How to use this notebook.** Run cells sequentially. The default executes small algebra examples and loads previously executed (retained) CFD results. Set a `RUN_*` switch to `True` only when ready to launch the corresponding new solve. When opening an updated lesson, start with a fresh Colab runtime: setup reuses an existing checkout. Download any outputs you want to keep before ending a temporary runtime.\n'
    source('pressure-recorded-evidence','## Recorded cavity evidence\n\nAll four formulations are included. The following tables distinguish reference-profile error, equation defects and recorded solver-call time. Matching spacing does not make the FV and FD spatial equations identical.')
    source('f5e083c3','''## Run selected FV methods again

`RUN_NEW=False` loads retained results. Set it to `True` to compute one or all
three FV methods. The independent FD reference remains retained; a retained peer
appears only where matching spacing is available. Each new plot labels fresh
and retained cases separately.

The controls below map to the preceding equations: `velocity_relaxation` and
`pressure_relaxation` are SIMPLE's alpha factors; `pressure_correctors` is PISO/PIMPLE's
inner count; `outer_correctors` is PIMPLE's outer count. Relaxation settings are
ignored by this transient PISO/PIMPLE implementation, outer count is ignored
outside PIMPLE, and `dt` is ignored by steady SIMPLE. A new output folder preserves
each run's exact configuration and fields. Compare runtimes on the same host.
''')
    source('327cdaea',BASELINE_CODE)
    source('01ef9b14',HIGHER_MD)
    source('1a08e6d5',HIGHER_CODE)
    c=byid['0e4e49ab'];s=c.source
    start=s.index('The study below actually runs') if 'The study below actually runs' in s else -1
    if start>=0:
        end=s.index('The two-dimensional formulation uses',start)
        s=s[:start]+'''The retained study contains independently executed runs from rest for all
four formulations, using central differences at matched spacing h=1/32.
FD uses 33x33 boundary-inclusive nodes; FV uses 32x32 pressure cells.
The default loads those records. The optional cell below recomputes an FD case.
Pressure in FD is recovered after velocity convergence.

'''+s[end:]
    source(c.id,s)
    c=byid['student-review-refinement-study'];s=c.source.replace('Separate fresh calculations check selected cases with','Separate retained reproduction runs check selected cases with').replace('The fresh suite includes','The retained reproduction suite includes')
    source(c.id,s)
    c=byid['student-review-refinement-evidence'];source(c.id,c.source.replace('Fresh author-side calculations (9 cases; independent timings):','Retained independent reproduction measurements (9 cases; separate timings):').replace('Shared cases compare rounded reported errors with fresh field-derived errors.','Shared cases compare rounded reported errors with field-derived reproduction errors.'))
    # The practical route uses named headings; theory section references remain unique.
    tail=False
    for c in book.cells:
        if c.id=='pressure-recorded-evidence':tail=True
        if tail and c.cell_type=='markdown':c.source=re.sub(r'(?m)^(#{2,3}) \d+\. ',r'\1 ',c.source)
        if c.cell_type=='markdown':
            pattern=r'(!\[[^\]]*\]\(https://raw\.githubusercontent\.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/([^/)]+)\.png\))\n\n\*\*Figure [^\n]+'
            def caption(m):
                e=guide[m.group(2)]
                return m.group(1)+'\n\n**Figure '+e['number']+'.** '+e['caption']
            c.source=re.sub(pattern,caption,c.source)
    book.cells=[c for c in book.cells if c.id not in {'beginner-recorded-metrics','beginner-fd-pressure-recovery'}]
    for anchor,id,text in [('51fefcd5','beginner-recorded-metrics',METRICS),
        ('ee106a2c','beginner-fd-pressure-recovery',r'''### FD pressure-gradient discrepancy

The FD formulation recovers pressure after the velocity solve. This quantity is
the RMS vector mismatch between the gradient of recovered pressure and the
pressure-gradient field inferred from steady momentum, divided by the latter's
RMS magnitude. RMS is the square root of the mean squared vector magnitude over
the grid. It tests recovery consistency; it is not a Ghia pressure error.
''')]:
        index=next(i for i,c in enumerate(book.cells) if c.id==anchor)
        book.cells.insert(index,nbformat.v4.new_markdown_cell(text,id=id))
    return book
