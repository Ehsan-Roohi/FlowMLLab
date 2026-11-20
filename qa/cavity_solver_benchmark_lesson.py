"""Append the measured linear-solver case study without rewriting earlier cells.

Run directly to replace only this tagged appendix in the two current notebooks.
The lesson builders import benchmark_cells() so later rebuilds retain it.
Run All reads frozen measurements; it never launches a solver or HPC job.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
TAG = 'flowmllab_solver_benchmark_v1'


def _cell(kind, text, label):
    source = textwrap.dedent(text).strip()
    cell = nb.v4.new_markdown_cell(source) if kind == 'md' else nb.v4.new_code_cell(source)
    cell.id = 'solver-benchmark-' + label
    cell.metadata[TAG] = True
    return cell


LOAD = r'''
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image

# Independent of neural-model imports; also runs from a notebook subdirectory.
bench_candidates = [Path(p) for p in [globals().get('ROOT'), Path.cwd(),
                    *Path.cwd().parents, Path('/content/FlowMLLab')] if p is not None]
bench_root = next((p for p in bench_candidates
                   if (p/'results/cavity_solver_benchmark/manifest.json').is_file()), None)
if bench_root is None:
    raise FileNotFoundError('Use the complete updated FlowMLLab checkout.')
BENCH = bench_root/'results/cavity_solver_benchmark'
bench_manifest = json.loads((BENCH/'manifest.json').read_text(encoding='utf-8-sig'))
for name, expected in bench_manifest['sha256'].items():
    assert hashlib.sha256((BENCH/name).read_bytes()).hexdigest() == expected, name
bench_evidence = json.loads((BENCH/'evidence.json').read_text(encoding='utf-8-sig'))
bench_trials = bench_evidence['trials']
production_trials = [e['result'] for e in bench_trials if e['path'].startswith('production/')]
assert len(production_trials) == 26

# Match inputs and numerical policy BEFORE calculating a performance ratio.
identity_keys = ['seed_sha256', 'reference_sha256', 'packed_initial_state_sha256',
                 'host', 'slurm_job_id', 'environment', 'thread_environment',
                 'schedule', 'nonlinear_tolerance', 'source_sha256_lf', 'candidate_config']
paired_runs = {}
for run in production_trials:
    key = (run['cells'], run['scenario'], run['repeat'])
    paired_runs.setdefault(key, {})[run['method']] = run
assert len(paired_runs) == 13
for key, pair in paired_runs.items():
    assert set(pair) == {'direct', 'gmres_mg'}, key
    direct, mg = pair['direct'], pair['gmres_mg']
    for name in identity_keys:
        assert direct[name] == mg[name], (key, name)
    for run in (direct, mg):
        assert run['success'] and run['final_residual_linf'] < 1e-8
        assert set(run['thread_environment'].values()) == {'1'}
        assert run['reference_discrepancies']['velocity_relative_l2'] < 1e-8
        assert run['reference_discrepancies']['omega']['relative_l2'] < 1e-7
        assert all(s['success'] and s['original_relative_defect'] <= 1e-5
                   for s in run['linear_solves'])

# One row per actual run; never put failed pilot runs in successful timing medians.
trial_timing = pd.DataFrame([{
    'nodes': run['nodes'], 'scenario': run['scenario'], 'method': run['method'],
    'repeat': run['repeat'], 'wall_s': run['solve_wall_seconds'],
    'setup_s': run['profiling']['preconditioner_setup_seconds'],
    'newton_steps': sum(h['newton_iterations'] for h in run['history']),
    'gmres_iterations': run['profiling']['krylov_iterations'],
    'peak_rss_MiB': run['peak_rss_KiB']/1024,
} for run in production_trials])
bench_timing = trial_timing.groupby(['nodes', 'scenario', 'method']).agg(
    repetitions=('wall_s', 'size'), median_s=('wall_s', 'median'),
    minimum_s=('wall_s', 'min'), maximum_s=('wall_s', 'max'),
    setup_median_s=('setup_s', 'median'),
    newton_steps_median=('newton_steps', 'median'),
    gmres_iterations_median=('gmres_iterations', 'median'),
    peak_rss_MiB=('peak_rss_MiB', 'max')).reset_index()
bench_comparison = bench_timing.pivot(index=['nodes', 'scenario'], columns='method', values='median_s')
bench_comparison['MG_over_direct'] = bench_comparison.gmres_mg / bench_comparison.direct
bench_comparison.columns.name = None
print('Verified evidence-file hashes and recorded input hashes/pair identity for 13 pairs (26 successful main trials).')
display(bench_timing[['nodes', 'scenario', 'method', 'repetitions', 'median_s', 'minimum_s', 'maximum_s']])
display(bench_comparison)
'''

FULL_INTRO = r'''
## 8. A measured solver experiment: fewer iterations, more time

**Engineering decision: retain our existing sparse-direct Newton solver for these cases.**
We tested a more elaborate linear solver rather than assuming that it would be faster.
Its warm-case time was **1.95-2.17 times the direct solver time**, while the accepted
fields agreed to numerical precision. This is a useful negative result: it prevents
repeating an expensive unsuccessful configuration and identifies what must change
before another attempt is worthwhile.

This appendix uses the completed, frozen experiment of 15 November 2026.
**Run All reanalyzes stored data on CPU; it does not run CFD, submit jobs or train a network.**
The [full evidence and report](../../results/cavity_solver_benchmark/README.md),
[experimental solver](../../flowmllab/cavity_krylov.py), and
[trial driver](../../qa/benchmark_cavity_solvers.py) preserve the successful and failed trials.
The corresponding discussion is at the end of the
[Week 4 lecture](../../lectures/week04_cavity_surrogates_deeponet.pdf), in the solver-benchmark evidence appendix.

### 8.1 What changed, and what stayed fixed?

The discrete cavity equations define a residual $F(q)=0$, where
$q=[\psi,\omega]^T$ contains the interior streamfunction and vorticity unknowns.
At Newton step $k$ we solve

$$J(q_k)\,\Delta q_k=-F(q_k),\qquad q_{k+1}=q_k+\alpha_k\Delta q_k.$$

Here $J=\partial F/\partial q$ is the same analytical Jacobian, $\Delta q_k$ the
Newton correction, and $0<\alpha_k\le1$ the existing line-search step.
The baseline factors the sparse $J$ directly. The candidate uses GMRES
(a Krylov iterative linear solver) with a coupled geometric Galerkin multigrid
V-cycle as preconditioner. A preconditioner is an approximate inverse used to
make the linear iteration easier; it also costs time to construct and apply.

The V-cycle uses bilinear prolongation, scaled-transpose restriction, and ILUT
(incomplete LU with threshold dropping) before and after coarse-grid correction.
The coarsest problem is solved directly. Invertible variable scaling and row
scaling balance variable and equation scales before the iterative solve; acceptance is checked again
in the **original, unscaled equations**. This is a CFD linear-solver experiment,
not a neural-network optimizer experiment.

The PDE, walls, grids, seed fields, precision, Jacobian, outer Newton algorithm
and final acceptance checks are unchanged. The final residual must be below
$10^{-8}$ in maximum norm, and velocity/vorticity must agree with the retained
same-grid CFD within relative $L_2$ limits $10^{-8}$ and $10^{-7}$.
**Inner tolerances are not identical:** the unchanged outer linear-defect gate is
$10^{-5}$; the candidate additionally requires an original-equation linear defect
at most $10^{-7}$, with internal GMRES relative tolerance $10^{-9}$.
We did not loosen the final criterion to make a method appear faster.

### 8.2 A matched timing protocol

Each trial used one CPU thread on the same Intel Xeon Gold 5220 node, with
Python 3.12.3, NumPy 2.2.6 and SciPy 1.15.3. Paired methods ran sequentially;
the two grid studies occupied separate cores on the same allocation.
There were three warm repetitions with method order **AB, BA, AB**, where A is
direct and B is GMRES/multigrid. Both methods start from the same stored
$Re=6500$ field for the $Re=7500$ target, or the same $Re=9000$ field for $Re=10000$.
These are **warm continuation steps, not cold-start solves**.

The separate 257-by-257-node continuation comparison starts from one retained
129-by-129-node $Re=1000$ field, interpolates it identically, and follows the same
20-root path from $Re=1000$ to $10000$. It has **one repetition per method**.
No full 513-by-513-node continuation comparison is claimed.

Solve wall time includes Jacobian assembly and all preconditioner construction.
It excludes queuing, imports, initial I/O, seed generation, verification and
checkpoint writing. The following cells recompute medians and ratios from the
individual run records rather than trusting a transcribed table.
'''

DIAGNOSE = r'''
### 8.3 Why did the candidate lose?

Both methods required four Newton steps in every warm case. The successful
candidate used only eight GMRES iterations in total, but rebuilding its strong
ILUT hierarchy at every Newton step dominated the cost. The ratio of median
setup time to median solve time was **89.3-95.2%**. Few Krylov iterations therefore
did not imply a fast solve. Peak process memory was also somewhat higher for
the candidate; it includes Python and all arrays, not just factors.

The first three lighter ILUT configurations failed both 129-by-129-node test
cases: six failures in total, comprising three singular incomplete
factorizations and three failures of the original-equation linear-defect check.
A bounded stronger pilot tested fill/drop settings $(20,10^{-6})$ and
$(40,10^{-8})$. Both passed both cases. The faster setting, $(40,10^{-8})$, was
**frozen before the 257- and 513-node evaluations**. This made the solver robust,
but not competitive in this experiment. Do not omit failed configurations from
the scientific record or tune the fine-grid result after seeing its timings.

The timing field `linear_total_seconds` already includes preconditioner setup.
**Do not add setup to that timer a second time.** We divide the independently
recorded setup median by the complete solve median below.
'''

PROFILE = r'''
### 8.4 First verify the physical answer

The velocity and signed-vorticity profiles below use the same native 513-by-513
grid, repeat 0, for both methods. The curves overlap because they solve the
same accepted discrete problem. The Newton-residual plots show the outer
convergence; they do not display physical time or quantify solve cost.

Across five representative complete-field pairs, the largest relative $L_2$
difference was $2.11\times10^{-14}$ for velocity and $1.71\times10^{-14}$ for
vorticity. For velocity this comparison is

$$E_{\rm pair}=\sqrt{\frac{\sum_i[(u_{{\rm MG},i}-u_{{\rm D},i})^2+
(v_{{\rm MG},i}-v_{{\rm D},i})^2]}{\sum_i(u_{{\rm D},i}^2+v_{{\rm D},i}^2)}}.$$

Here $i$ enumerates every common grid node, D denotes the direct result and MG
the candidate. Vorticity uses the analogous scalar norm. The compact audit
below contains the recorded full-field checks; the plotted centreline curves
are not substituted for that full-grid calculation.
**Solver agreement is not error against Ghia, a continuum exact solution,
mesh independence, or evidence of temporal stability.**
'''

EXERCISES = r'''
### 8.5 Engineering exercises: use the evidence before spending more compute

**F - Make the performance decision numerically.** From `trial_timing`, reproduce
all four warm-case medians and min/max ranges for both methods. Report
$R=T_{\rm MG}/T_{\rm D}$ and the number of repetitions. Explain why the
continuation row is separate and why its single repetition cannot supply a
credible repeat-to-repeat variability estimate. Report the six failed pilot
trials separately; do not average them into successful solve times.

**G - Diagnose the cost and state a testable next idea.** For each warm case,
calculate the ratio of median setup to median total time. Then investigate

$$T_{\rm hyp}(r)=T_{\rm MG}-T_{\rm setup}+rT_{\rm setup},\qquad 0\le r\le1.$$

Here $r$ is the **retained fraction** of setup cost; $1-r$ is the reduction.
Solve for the largest $r$ that would make $T_{\rm hyp}\le T_{\rm D}$ and report
the required percentage reduction for all four cases. Assume other costs stay
fixed only for this calculation. Propose a preconditioner-reuse experiment,
including when to rebuild and how to check the original linear residual.
Reusing factors can increase Krylov work or cause failure: your calculated
headroom is **not a measured speedup** and reuse has not been tested here.

**H - Separate numerical agreement from physical accuracy.** Plot signed
direct-minus-candidate velocity and vorticity differences for the two 513-node
warm cases. Select only entries whose `path` starts with `production/n512/`,
whose `result['repeat'] == 0`, whose `result['scenario']` is `warm7500` or
`warm10000`, and which contain `profiles`. This gives four entries: two methods
for each case. In each `profiles` dictionary, use the stored `y` with `u_center`,
and the stored `x` with `v_center` and `omega_horizontal_center`. Pair the methods
at identical coordinates. Make one three-panel signed-difference figure per Re;
retain the native sample points and the velocity/vorticity units. Compare these
discrepancies with the Lab 7 mesh-sensitivity and Ghia comparisons. Explain why
a difference near machine precision does not make either CFD solution exact
or establish stable high-Re physical flow.

Submit one CSV with timings, ratios, setup fractions, required setup reductions
and sample counts; two signed physical-profile comparisons; and a short
recommendation tied to the measured cost and acceptance gates. Include the
evidence manifest hash. **No additional CFD or training run is required.**

### 8.6 What this experience establishes

For the recorded grids, seeds, precision and hardware, the existing direct solver
is the better production choice. The candidate is retained as a reproducible
experimental implementation, not enabled by default. This result does not rule
out multigrid generally, other smoothers, safe hierarchy reuse, or larger problems.
It is **not** Ghia et al.'s coupled strongly implicit multigrid algorithm. Historical
Ghia timings and hardware prices do not form a controlled speed comparison.
An improved candidate must keep the original acceptance gates, preserve failure
records, freeze tuning on a small pilot, and beat the complete solve cost on fresh
matched cases before replacing the existing method.
'''

DIAGNOSTIC_CODE = r'''
candidate_warm = bench_timing[(bench_timing.method == 'gmres_mg') &
                               bench_timing.scenario.str.startswith('warm')].copy()
candidate_warm['setup_fraction_of_medians'] = candidate_warm.setup_median_s / candidate_warm.median_s
display(candidate_warm[['nodes', 'scenario', 'newton_steps_median',
                       'gmres_iterations_median', 'setup_median_s', 'median_s',
                       'setup_fraction_of_medians', 'peak_rss_MiB']])
pilot_rows = []
for entry in bench_trials:
    if entry['path'].startswith(('pilot/', 'pilot_strong/')):
        run = entry['result']
        config = run.get('candidate_config') or {}
        pilot_rows.append({'phase': entry['path'].split('/')[0], 'scenario': run['scenario'],
            'method': run['method'], 'fill': config.get('ilu_fill_factor'),
            'drop': config.get('ilu_drop_tol'), 'passed': run['success'],
            'wall_s_if_passed': run.get('solve_wall_seconds') if run['success'] else np.nan,
            'failure': run.get('error', '')})
pilot_table = pd.DataFrame(pilot_rows)
assert ((pilot_table.method == 'gmres_mg') & ~pilot_table.passed).sum() == 6
display(pilot_table)
'''

FIGURE_CODE = r'''
paired_field_audit = json.loads((BENCH/'paired_fields_audit.json').read_text(encoding='utf-8-sig'))
paired_field_table = pd.DataFrame(paired_field_audit['pairs'])
display(paired_field_table[['nodes', 'scenario', 'repeat',
                           'paired_velocity_relative_l2', 'paired_omega_relative_l2']])
display(Image(filename=str(BENCH/'physical_profiles.png')))
display(Image(filename=str(BENCH/'nonlinear_convergence.png')))
print('Manifest SHA256:', hashlib.sha256((BENCH/'manifest.json').read_bytes()).hexdigest())
'''

BRIDGE = r'''
## 9. Measured CFD cost: a solver change that did not pay off

The [new final section of Lab 7](W4_Lab7_High_Re_CFD_Ghia.ipynb) records an actual
matched experiment: keep the steady cavity equations, grids, seeds and outer
Newton iteration, but replace sparse direct factorization with GMRES and a
multigrid preconditioner. The candidate took **1.95-2.17 times as long** in four
warm-case/grid comparisons. Rebuilding the strong ILUT hierarchy consumed
**89.3-95.2%** of solve time (ratios of medians), despite only eight total GMRES
iterations and the same four Newton steps. Our production CFD solver is retained.

These are the costs of **generating discrete CFD labels**, not training times,
neural inference times, or neural accuracy gains. No network was retrained in
this experiment, no train/validation/test split changed, and no new optimizer
was tested on the neural loss. A faster label generator would not itself lower
the neural prediction error or resolve the mesh uncertainty discussed above.

Three warm repetitions per method start from matched $Re=6500$ or $9000$
fields, targeting $Re=7500$ or $10000$. The 257-node full continuation path has
one repetition per method. All used one CPU thread on an Intel Xeon Gold 5220
node. Timing includes setup but excludes seed generation, queues, I/O and
verification; it is not an end-to-end dataset-production invoice. The candidate
has a stricter additional inner residual check, while final acceptance is shared.

The following CPU-only cell verifies the evidence-file hashes and the recorded
input hashes/pair identity, then recalculates the table. It does not reload the
original seed NPZ files from the full benchmark archive. See the [full report](../../results/cavity_solver_benchmark/README.md)
and the solver-benchmark evidence appendix in the [Week 4 lecture](../../lectures/week04_cavity_surrogates_deeponet.pdf)
for failures, equations, tolerances and limitations. This is one candidate
implementation; it is not a reproduction of Ghia's strongly implicit multigrid.
'''

BRIDGE_END = r'''
### Engineering exercise H - Account for the entire surrogate workflow

Use the measured `bench_comparison` table to explain why replacing the label
solver with this candidate would increase the measured solve cost. Report the
four warm ratios and distinguish them from the single continuation ratio.
Keep these matched November 15 timings separate from the earlier historical
comparison in Section 7, which used different runs and initialization schedules.

For $M$ future queries, write a cost ledger

$$T_{\rm surrogate}(M)=T_{\rm labels}+T_{\rm train}+M\,t_{\rm infer},\qquad
T_{\rm CFD}(M)=M\,t_{\rm CFD}.$$

Here $T_{\rm labels}$ is total label-generation cost, $T_{\rm train}$ training and
selection cost, $t_{\rm infer}$ cost per neural query, and $t_{\rm CFD}$ a comparable
CFD query cost. If $t_{\rm CFD}>t_{\rm infer}$, the break-even condition is
$M>(T_{\rm labels}+T_{\rm train})/(t_{\rm CFD}-t_{\rm infer})$.
List which terms are measured, missing, or incomplete in this notebook. Explain
why a warm continuation timer cannot automatically stand in for an arbitrary
new CFD query. Do not invent an inference time or numerical break-even count;
state what a fair future measurement must include.

Finally, compare solver-to-solver agreement (maximum full-grid velocity relative
$L_2$ discrepancy $2.11\times10^{-14}$ over five representative field pairs) with
the actual held-out neural error and mesh sensitivity above. These are different
quantities. Submit a one-page cost ledger and a numerical timing table, with the
manifest hash. No rerun or training is required; further model selection must
still keep the disclosed test cases out of fitting and tuning.
'''


def benchmark_cells(kind='cfd'):
    """Return fresh tagged cells, suitable for new builds or append-only updates."""
    if kind == 'cfd':
        return [_cell('md', FULL_INTRO, 'cfd-intro'), _cell('code', LOAD, 'cfd-load'),
                _cell('md', DIAGNOSE, 'cfd-diagnosis'), _cell('code', DIAGNOSTIC_CODE, 'cfd-costs'),
                _cell('md', PROFILE, 'cfd-accuracy'), _cell('code', FIGURE_CODE, 'cfd-figures'),
                _cell('md', EXERCISES, 'cfd-exercises')]
    if kind == 'neural':
        return [_cell('md', BRIDGE, 'neural-intro'), _cell('code', LOAD, 'neural-load'),
                _cell('md', BRIDGE_END, 'neural-exercise')]
    raise ValueError(kind)


def append_current(path: Path, kind: str):
    """Execute only the new suffix, preserving all prior cells and stored outputs."""
    sys.path.insert(0, str(ROOT/'qa'))
    from execute_cavity_lesson import execute
    doc = nb.read(path, as_version=4)
    tagged = [i for i,c in enumerate(doc.cells) if c.metadata.get(TAG)]
    prefix_count = min(tagged) if tagged else len(doc.cells)
    if tagged and tagged != list(range(prefix_count, len(doc.cells))):
        raise ValueError('Benchmark cells must form a contiguous final suffix.')
    prefix = copy.deepcopy(doc.cells[:prefix_count])
    prefix_digest = hashlib.sha256(json.dumps(prefix,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    suffix = nb.v4.new_notebook(cells=benchmark_cells(kind))
    execute(suffix, ROOT)
    offset = max((c.get('execution_count') or 0 for c in prefix), default=0)
    for cell in suffix.cells:
        if cell.cell_type == 'code':
            cell.execution_count += offset
    doc.cells = prefix + suffix.cells
    doc.metadata['solver_benchmark_appendix'] = {
        'tag': TAG, 'base_cells_preserved': prefix_count, 'base_cells_sha256': prefix_digest,
        'new_cells':len(suffix.cells), 'execution':suffix.metadata['execution_audit'],
        'scope':'Frozen benchmark reanalysis only; no CFD run, submission or model training'}
    nb.validate(doc)
    nb.write(doc, path)
    saved = nb.read(path, as_version=4)
    assert saved.cells[:prefix_count] == prefix
    return {'notebook':str(path.relative_to(ROOT)), 'base_cells_preserved':prefix_count,
            'base_cells_sha256':prefix_digest, 'appendix_cells':len(suffix.cells),
            'new_code_cells_executed':suffix.metadata['execution_audit']['code_cells'],
            'prior_cells_and_outputs_unchanged':True,
            'execution_method':'Fresh namespace, sequential socket-free Python; not hosted Colab/Jupyter certification'}


if __name__ == '__main__':
    reports = [append_current(ROOT/'notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb','cfd'),
               append_current(ROOT/'notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb','neural')]
    print(json.dumps(reports,indent=2))
