"""Update and execute Week 4's physical-output teaching supplements.

Lab 2 and Lab 5 have separate owners/builders.  No frozen numerical evidence or
model is rewritten here.  Use --update for the idempotent additive source
revision, --complete-lab4-gallery for its native seven-method gallery, and
--execute NAME to verify a specific notebook on CPU. The notebook generators
call the same updater for their own notebook after rebuilding its source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks/week04"
RESULTS = ROOT / "results/week04_teaching_outputs"

FILES = {
    "lab1": "W4_Lab1_CFD_Data_Production_Student.ipynb",
    "lab3": "W4_Lab3_DeepONet_Cavity_Student.ipynb",
    "lab4": "W4_Lab4_Stokes_to_Navier_Stokes.ipynb",
    "grid51": "W4_Lab4_Grid51_Validation.ipynb",
    "lab6": "W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb",
    "lab7": "W4_Lab7_High_Re_CFD_Ghia.ipynb",
    "lab8": "W4_Lab8_High_Re_Field_DeepONet.ipynb",
    "lab9": "W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb",
    "rom": "W4_1_Classical_ROM_Cavity.ipynb",
}

AIMS = {
"lab1": r"""## Aim and route through this lab

**Aim.** Produce a numerical cavity field that another student can inspect and
qualify before using it as a learning label. The input is Reynolds number and a
declared grid/time-step policy; the output is the velocity, streamfunction and
vorticity together with convergence history and a quality card. This lab runs
the CFD solver. It does not train a network.

The solver advances the vorticity transport equation, solves the streamfunction
Poisson equation, obtains velocity by differentiating streamfunction, and
reapplies the moving-lid and stationary-wall conditions. A small change between
iterations is only one check: it does not establish mesh independence or
accuracy against Ghia. The fixed number of Poisson iterations is also part of
the numerical policy, not an exact Poisson solve.

| Stage | What you do | Visible evidence and interpretation |
|---|---|---|
| Common reproduction | Run the declared Re=100 case | Speed, streamlines, signed streamfunction and vorticity show the physical flow; the residual shows iteration history. |
| Independent check | Compare both centreline components at the printed Ghia positions | Signed discrepancies reveal where the CFD differs; flagged printed entries remain visible. |
| Assigned production | Change only the assigned ID and solve its specified Re | Compare vortex position/strength and check the quality card before export. |
| Export and report | Save fields plus metadata | A saved file is not accepted merely because a solver stopped. Quote each gate and keep grid/time studies separate. |

**Reading.** Use the Lab 1 section of [Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
**Output contract.** Each live solve produces a physical-field figure and a
separate solver/reference diagnostic figure. Legends sit outside the data;
all native extrema are retained. Vorticity uses a symmetric-log scale because
wall values are much larger than the weak interior circulation. Its scale is
labelled explicitly; velocity discrepancies remain linear. Ghia supplies
velocity profiles, not a full reference pressure or vorticity field.
""",
"lab3": r"""## Aim and route through this lab

**Aim.** Learn a map from Reynolds number to an entire cavity velocity and
pressure field using a fixed POD spatial basis and small neural branch heads.
This extends Lab 2's pointwise coordinate network: the trunk here is obtained
from an SVD of development CFD fields, so each learned coefficient controls a
coherent spatial pattern. A separate pressure trunk avoids mixing quantities
with different scales in a single SVD.

| Stage | Numerical operation | What the output should let you judge |
|---|---|---|
| CFD/data qualification | Check centreline physics, array coordinates and whole-case splits | A small network loss cannot repair mislabeled locations or incorrect CFD. |
| Representation | Subtract a development mean and compute the POD modes | Mode plots and singular values show the structures available to the regressor. Energy alone is not a vortex-accuracy certificate. |
| Branch selection | Fit coefficients using complete training cases and select on Re=225 | The validation table separates rank, width, input transform and seed effects. |
| Public held-out assessment | Reconstruct all three cases, Re=175,275,375 | Each case gets velocity contours, overlaid streamlines, signed centreline discrepancies and a pressure comparison. |
| Cost and limits | Read field errors alongside wall/divergence/gauge checks | Fast inference is useful only after label quality and accuracy have been established. |

**Default mode** re-plots the retained numerical predictions and recomputes
their errors; it does not silently train a new model. In `fresh_experiment`
mode, the same plots consume the current in-memory fit. The already disclosed
cases remain public held-out regression tests, not new blind tests. Read the
Lab 3 section of [Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
""",
"lab4": r"""## Aim and route through this lab

**Aim.** Ask whether learning the nonlinear correction to a known Stokes
solution is more effective than learning the complete Navier–Stokes field.
The Stokes baseline omits inertial convection; the network learns the remaining
streamfunction correction. Differentiation supplies velocity, and the loss
can include vorticity so that a small velocity error does not hide weak flow
structures. Changing the lid profile changes the physical boundary problem,
not merely the network input format.

Work through sections 1–3 to identify the Reynolds/lid families and accepted
CFD grids. Sections 4–6 explain the correction representation, derivative
weights and optional regeneration. The historical pilot gallery compares all
seven retained methods. Sections 7–9 compare the original POD and correction
stages using velocity, recovered pressure, streamlines and vortices. Section 11 links
the separate 51-by-51 refinement addendum. This notebook is the **Lab 4**
correction extension, also listed as **Week 4.2** in the course map.

**What to read from the figures.** Match the named lid family and Re before
comparing methods. A streamfunction contour is a physical flow structure;
low global velocity error alone does not guarantee that its secondary eddies
are captured. Compare the same-grid model discrepancy first, then the change
in the CFD label under refinement. Pressure was reconstructed and its gradient
consistency is a separate audit; Ghia does not validate pressure.

Run All reanalyses retained evidence on CPU. The optional regeneration switch
is explicit. Keep the existing exercises and add to your report one example
where a local derivative/vortex check changes the conclusion suggested by the
global velocity score. Read the Lab 4 section of
[Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
""",
"grid51": r"""## Aim of this historical refinement companion

**Aim.** Separate the error made by the surrogate on a fixed CFD grid from the
change in the CFD reference when that grid is refined. This companion supports
the Lab 4 / Week 4.2 Stokes-correction experiment; it is not another neural
architecture and not a newly generated reference solution.

Read the 25-to-51-node comparison alongside the same-grid surrogate errors.
Inspect the retained velocity/streamline/vorticity images, identify the lid
family and Re in each panel, and state whether the field-label change is larger
than the improvement attributed to a model. Refinement sensitivity is evidence
about the numerical label, not an exact continuum error estimate. The tables,
original figures and reproduction commands below are all retained.
""",
"lab6": r"""## Aim and route through this lab

**Aim.** Isolate how network depth, optimizer and data weighting affect
interpolation between *whole Reynolds cases* of Ghia centreline velocities.
The model receives position, log Reynolds number and a velocity-component tag.
It returns one velocity sample. It cannot produce a two-dimensional vortex,
pressure contour or CFD field from those two one-dimensional profiles.

First verify the frozen cases and seeds; next compare every retained model
with training-case interpolation at all seven Re. Read the signed discrepancy
panels as carefully as the physical profiles. The five-layer model has more
parameters than the two-layer model, and wall weighting changes the training
objective: neither is a pure, parameter-matched depth comparison. The displayed
scores average per-seed errors, whereas the curves show the seed-mean
prediction; these are different statistics.

The optional last cell is the student experiment: declare architecture,
optimizer and evaluation budget, fit only the allowed training cases, and
inspect **Re=400 validation profiles, signed errors and training history**.
Freeze a decision before assessing the public test cases. The loss-history
horizontal axis counts joint objective/gradient evaluations; it is not the
number of epochs and does not equate computational cost between optimizers.

**Required evidence:** an all-model comparison table, two physical profiles,
signed discrepancies, seed spread, parameter counts and measured time. The
retained interpolation baseline wins both disclosed test cases; a polished
plot must not imply that every neural fit is better. Read the Lab 6 section of
[Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
""",
"lab7": r"""## Aim and route through this lab

**Aim.** Qualify the high-Reynolds CFD fields before using them as neural labels.
Re=3200,5000,7500,10000 are represented on 128,256,512-cell grids. The notebook
loads native arrays and verifies hashes, equations, walls and centreline
comparisons. Run All audits completed numerical evidence on CPU; it does not
launch those large solves again.

Sections 2–3 show the actual flow and the published Ghia samples. Section 4
compares nested native grids: signed velocity changes locate where refinement
matters, and signed streamfunction contours expose weak corner vortices that
a whole-domain view may miss. Section 5 is a distinct short-time perturbation
and time-step check. A steady Newton residual does not prove that a steady
solution is dynamically stable at high Re.

Sections 6–7 explain deliberate reproduction and specific engineering tasks.
The final solver appendix compares a sparse-direct Newton solve with a
GMRES/multigrid candidate at matched accepted discrete solutions. Fewer linear
iterations did not imply lower total time because setup dominated. Keep this
CFD solver experiment separate from a neural optimizer experiment.

**Required evidence:** all/screened Ghia errors, signed mesh differences,
corner-flow contours, short-time sensitivity, and an explicit decision about
which uncertainty remains. Read the Lab 7 and solver-cost sections of
[Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
""",
"lab8": r"""## Aim and route through this lab

**Aim.** Predict complete high-Re cavity velocity fields from Reynolds number
and compare a frozen POD-DeepONet with interpolation using the same training
information. Unlike Lab 6, this is a two-dimensional field experiment; unlike
Lab 9, its training labels are CFD fields.

The first three sections establish the whole-Re split, SVD representation,
validation-based architecture choice, and projection error that no branch
optimizer can remove. Sections 4–6 then compare retained test predictions:
inspect overlaid streamlines, signed velocity profiles, vorticity and recovered
pressure, with special attention to weak corner circulation. A smaller global
velocity norm does not guarantee a smaller derivative or corner error.

The optional CPU-fit cell now plots the **current validation** predictions and
their profiles, not just a numeric table. It does not alter the frozen result
package or fit on test targets. Its rank remains the frozen trunk rank; deeper
branches cannot create spatial modes absent from that trunk. New architecture
choices informed by the already published test images need fresh tests for a
new research claim.

Run All re-plots verified evidence on CPU. The historical-computing and solver
cost sections distinguish a measured present-day experiment from historical
context. Read the Lab 8 section of
[Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf), and include both
gains and deteriorations in your report.
""",
"lab9": r"""## Aim and route through this advanced audit

**Aim.** Determine whether a label-free PINN trained from the equations and
boundary conditions actually reproduces cavity flow. Complete Week 13 Part A
before this audit. The CFD/Ghia fields are evaluation references, not training
labels. Exact boundary transforms and an exact continuity identity do not
establish correct momentum balance or vortex structure.

Sections 1–3 define the equations, corner collar and residual-guided sampling.
The finite-pool example now plots the uniform and adaptive sampling densities
alongside the residual-like integrand: importance weights preserve the stated
uniform objective in expectation, while unweighted adaptive sampling changes
it. This is an illustrative quadrature example, not a trained cavity result.

Then choose Re and seed for history/physical-profile inspection. The completed
baseline gallery retains every Reynolds case and predeclared representative
fields, followed by the matched-work uniform-versus-stratified comparison.
Use its separate case/seed/collar metadata; do not compare unrelated runs by
their best-looking panel. A rejected result remains rejected in the figures.

Run All reads verified evidence on CPU and never starts training or cluster
jobs. Report PDE audit, signed reference discrepancies, corner coverage and
seed variation alongside training loss. Read the Lab 9 section of
[Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf).
""",
"rom": r"""## Aim and route through this optional Lab 4.1

**Aim.** Distinguish a compact spatial representation from a dynamical reduced
solver, and determine which work DEIM actually removes. Both POD-Galerkin and
POD-DEIM evolve coefficients from the governing cavity equations; neither
learns a static Re-to-field map as the Lab 3 neural branch does.

Validate the full-order model (FOM), inspect grid/time sensitivity, then read
the validation-based rank decision. The fresh Re=275 calculation integrates
FOM, Galerkin and DEIM to the **same final physical time t*=5**. Both reduced
methods now receive their own physical-field/error figure, followed by shared
centreline comparisons and velocity-error histories. Those curves use current
in-memory runs, not a stored training figure. Subsequent cells reproduce all
three public held-out trajectories.

Separate snapshot projection error, dynamical reduction error, nonlinear DEIM
error and CFD discretization error. Full Poisson/nonlinear work can dominate a
Galerkin solver even when its state has only 16 coefficients. Speed should be
reported with offline cost and repeated-query break-even count. These are
time-matched transient trajectories, not a new claim of steady Ghia accuracy.
The original evidence, equations and exercises below remain part of the lab.
""",
}


def source(cell):
    return cell.source


def replace_code(nb, needle, replacement):
    matches = [c for c in nb.cells if c.cell_type == "code" and needle in c.source]
    if len(matches) != 1:
        raise ValueError(f"Expected one cell containing {needle!r}; found {len(matches)}")
    matches[0].source = replacement
    matches[0].outputs = []
    matches[0].execution_count = None


def update(keys=None):
    import nbformat
    selected = list(FILES) if keys is None else list(keys)
    unknown = set(selected) - set(FILES)
    if unknown:
        raise ValueError(f"Unknown notebook keys: {sorted(unknown)}")
    for key in selected:
        filename = FILES[key]
        path = NOTEBOOKS / filename
        nb = nbformat.read(path, as_version=4)
        if any(c.get("metadata", {}).get("week04_output_revision") for c in nb.cells):
            # Do not duplicate imports, reset output, or alter an executed file.
            continue
        aim = nbformat.v4.new_markdown_cell(AIMS[key], metadata={"week04_output_revision": 1})
        nb.cells.insert(1, aim)
        # Put the root on sys.path for local Run All as well as hosted Colab.
        first_code = next(c for c in nb.cells if c.cell_type == "code")
        first_code.source += '''\n\n# Shared, array-driven figures; student outputs stay separate from frozen evidence.
from pathlib import Path
import os, sys
_w4_root = next(p for p in [Path.cwd(), *Path.cwd().parents, Path('/content/FlowMLLab')]
                if (p/'flowmllab/week04_teaching_plots.py').is_file())
if str(_w4_root) not in sys.path: sys.path.insert(0, str(_w4_root))
from flowmllab import week04_teaching_plots as w4plots
W4_OUTPUT = Path(os.environ.get('FLOWMLLAB_W4_OUTPUT', str(_w4_root/'student_outputs/week04_current')))
W4_OUTPUT.mkdir(parents=True, exist_ok=True)
def save_current_figure(fig, name):
    fig.savefig(W4_OUTPUT/(name+'.png'), dpi=150, bbox_inches='tight')
    return fig
'''
        if key == "lab1":
            replace_code(nb, "def diagnostic_figure(result):", '''def diagnostic_figure(result, label="reproduction"):
    figures = w4plots.cfd_diagnostics(result)
    for suffix, fig in zip(('fields', 'diagnostics'), figures):
        save_current_figure(fig, f"lab1_{label}_Re{int(result['Re'])}_{suffix}")
        plt.show()
        plt.close(fig)
    return figures

diagnostic_figure(reproduction)
''')
            for c in nb.cells:
                if c.cell_type == "code":
                    c.source = c.source.replace('diagnostic_figure(production); plt.show()',
                                                'diagnostic_figure(production, "production")')
                    c.source = c.source.replace('"reproduction", "case_outputs"',
                                                '"reproduction", W4_OUTPUT/"lab1_reproduction"')
                    c.source = c.source.replace('cfg["split"], "case_outputs"',
                                                'cfg["split"], W4_OUTPUT/"lab1_production"')
        elif key == "lab3":
            first_code.source = first_code.source.replace('sys.path.insert(0, str(ROOT / "common"))',
                'if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\nsys.path.insert(0, str(ROOT / "common"))')
            replace_code(nb, "re_plot = float(blind", '''# Use the current fresh ensemble OR the retained numerical predictions explicitly.
if RUN_REFIT:
    plotted_predictions = fresh_predictions
    plot_context = 'fresh in-memory ensemble'
else:
    archive = np.load(RESULTS/'deeponet_predictions.npz', allow_pickle=False)
    plotted_predictions = {float(re): tuple(archive[q][j] for q in ('u','v','p'))
                           for j,re in enumerate(archive['Re'])}
    plot_context = 'retained public held-out ensemble'

recomputed = []
for re_plot, (up, vp, pp) in plotted_predictions.items():
    idx = poddon.case_index(data, re_plot)
    truth_u, truth_v = data['u'][idx], data['v'][idx]
    prediction = {'POD-DeepONet': (up, vp)}
    figs = w4plots.velocity_comparison(data['x'], data['y'], truth_u, truth_v,
                                       prediction, re_plot, plot_context)
    figs += [w4plots.velocity_profiles(data['x'], data['y'], truth_u, truth_v,
                                       prediction, re_plot, plot_context)]
    figs += [w4plots.pressure_comparison(data['x'], data['y'], data['p'][idx], pp, re_plot)]
    for suffix, fig in zip(('velocity', 'profiles', 'pressure'), figs):
        save_current_figure(fig, f'lab3_Re{re_plot:g}_{suffix}')
        plt.show(); plt.close(fig)
    record = {'Re': re_plot, **poddon.field_report(data, re_plot, (up, vp, pp))}
    recomputed.append(record)
recomputed = pd.DataFrame(recomputed)
display(recomputed[['Re','relative_L2_uv','relative_L2_p','wall_rms_error','div_l2_pred']])
# Check retained labels against the original numeric prediction package, not a screenshot.
if not RUN_REFIT:
    for re_plot in plotted_predictions:
        a = recomputed.set_index('Re').loc[re_plot]
        b = blind.set_index('Re').loc[re_plot]
        np.testing.assert_allclose(a[['relative_L2_uv','relative_L2_p']].to_numpy(float),
                                   b[['relative_L2_uv','relative_L2_p']].to_numpy(float), rtol=1e-9)
''')
            idx = next(i for i,c in enumerate(nb.cells) if c.cell_type == "code" and
                       "_, singular_values, spatial_modes" in c.source)
            nb.cells.insert(idx+1, nbformat.v4.new_code_cell('''# A mode is a signed correction pattern, not an independent cavity solution.
fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout='constrained')
npts = len(data['x'])*len(data['y'])
for k in range(3):
    for row, component in enumerate(('u','v')):
        mode = spatial_modes[k, row*npts:(row+1)*npts].reshape(len(data['y']), len(data['x']))
        scale = max(np.abs(mode).max(), 1e-15)
        im = axes[row,k].contourf(data['x'], data['y'], mode,
                                  levels=np.linspace(-scale, scale, 31), cmap='turbo')
        axes[row,k].set(xlabel='x/L', ylabel='y/L', aspect='equal',
                        title=f'Mode {k+1}: {component} component')
        fig.colorbar(im, ax=axes[row,k], shrink=.82)
fig.suptitle('Signed POD velocity modes from development fields', fontsize=12)
save_current_figure(fig, 'lab3_pod_velocity_modes'); plt.show(); plt.close(fig)
'''))
            for c in nb.cells:
                if c.cell_type == "code" and 'for ax in axes: ax.grid(alpha=.25); ax.legend()' in c.source:
                    c.source = c.source.replace('for ax in axes: ax.grid(alpha=.25); ax.legend()\nplt.tight_layout()',
                        "for ax in axes: ax.grid(alpha=.25)\nfig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False)\nfig.suptitle('Re = 400; educational CFD on 65 × 65 nodes', y=1.05)\nfig.tight_layout(rect=(0,0,1,.88)); plt.show()")
        elif key == "rom":
            for c in nb.cells:
                c.source = c.source.replace('65Ã—65','65×65').replace('33Ã—33','33×33').replace('25â†’33','25→33').replace('33â†’49','33→49').replace('.004â†’.002','.004→.002').replace('.002â†’.001','.002→.001')
            cell = next(c for c in nb.cells if c.cell_type == 'code' and 'model = cavity_rom.load_deim_model' in c.source)
            prefix = cell.source[:cell.source.index('truth = cavity_rom.state_to_fields')]
            cell.source = prefix + '''truth = cavity_rom.state_to_fields(fom['states'][-1], 33)
pg_field = cavity_rom.state_to_fields(galerkin['states'][-1], 33)
deim_field = cavity_rom.state_to_fields(deim['states'][-1], 33)
x = np.linspace(0, 1, 33)
predictions = {'POD-Galerkin': (pg_field['u'], pg_field['v']),
               'POD-DEIM': (deim_field['u'], deim_field['v'])}
figures = w4plots.velocity_comparison(x, x, truth['u'], truth['v'], predictions,
                                       275, 'time-matched t*=5; FOM reference')
for name, fig in zip(('galerkin', 'deim'), figures):
    save_current_figure(fig, 'rom_'+name+'_physical'); plt.show(); plt.close(fig)
fig = w4plots.velocity_profiles(x, x, truth['u'], truth['v'], predictions, 275, 't*=5')
save_current_figure(fig, 'rom_physical_profiles'); plt.show(); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 3.8), layout='constrained')
ax.plot(fom['time'], 100*pg_error, label='POD-Galerkin', color='#0077bb')
ax.plot(fom['time'], 100*deim_error, label='POD-DEIM', color='#d62728', ls='--')
ax.set(xlabel='Physical time t*', ylabel='Velocity relative L2 discrepancy (%)',
       title='Re = 275; matched FOM/ROM time samples')
ax.grid(alpha=.2)
fig.legend(*ax.get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
save_current_figure(fig, 'rom_time_error'); plt.show(); plt.close(fig)
'''
        elif key == "lab6":
            cell = next(c for c in nb.cells if c.cell_type == 'code' and 'TRAIN_NEW_PROFILE_MODEL = False' in c.source)
            cell.source = cell.source.replace("else:\n    print('Showing retained runs; no test-informed retraining.')", '''    fig = w4plots.profile_trial(trial, 400)
    save_current_figure(fig, 'lab6_current_validation_profiles'); plt.show(); plt.close(fig)
    hist = pd.DataFrame(trial['history'])
    fig, axes = plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    for ax, coordinate in zip(axes, ('evaluation','seconds')):
        for phase, values in hist.groupby('phase',sort=False):
            ax.semilogy(values[coordinate], values['loss'], label=phase)
        ax.set(xlabel=coordinate, ylabel='Training objective'); ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    save_current_figure(fig, 'lab6_current_training_history'); plt.show(); plt.close(fig)
else:
    print('Showing retained runs; no test-informed retraining.')''')
        elif key == "lab7":
            index = next(i for i,c in enumerate(nb.cells) if c.cell_type == 'code' and 'sample_table =' in c.source)
            nb.cells.insert(index+1, nbformat.v4.new_markdown_cell('''### Locate the mesh discrepancy in physical space

This figure implements the comparison needed before declaring a grid sufficient.
The finest retained CFD field is the reference for this **mesh-sensitivity**
check, not an exact solution. The upper-left region is not inferred from a
global error: inspect its signed streamfunction levels on each native grid.
The whole-domain and corner panels use identical physical contour levels for
all grids; solid, dashed and dotted lines distinguish the grids.
'''))
            nb.cells.insert(index+2, nbformat.v4.new_code_cell('''subset = sorted([c for c in cases if c['Re']==RE_TO_INSPECT],key=lambda c:c['cells'])
fine = subset[-1]
fig, axes = plt.subplots(2,2,figsize=(10,7.8),layout='constrained')
for case, style, color in zip(subset[:-1], ('--','-.'), ('#0077bb','#d62728')):
    for ax, component in zip(axes[0], ('u','v')):
        position, coarse = case['profiles'][component]
        reference = np.interp(position,*fine['profiles'][component])
        ax.plot(position, coarse-reference,ls=style,color=color,
                label=f"{case['cells']} minus {fine['cells']} cells")
        ax.set(xlabel='y/L' if component=='u' else 'x/L',ylabel=f'Signed {component} discrepancy / U_lid')
for ax in axes[0]: ax.axhline(0,color='.5',lw=.7); ax.grid(alpha=.2)
levels = np.r_[-np.geomspace(.00001,.12,15)[::-1], np.geomspace(.000001,.003,12)]
for case, style, color in zip(subset, ('--','-.','solid'), ('#0077bb','#d62728','black')):
    f=case['field']
    for ax in axes[1]:
        ax.contour(f['x'],f['y'],f['psi'],levels=levels,colors=color,linestyles=style,linewidths=.8)
        ax.set(xlabel='x/L',ylabel='y/L',aspect='equal')
axes[1,0].set(xlim=(0,1),ylim=(0,1),title='Common signed streamfunction levels')
axes[1,1].set(xlim=(0,.3),ylim=(.7,1),title='Upper-left corner: identical levels')
from matplotlib.lines import Line2D
handles = [Line2D([],[],color=color,ls=style,label=f"{c['cells']} cells")
           for c,style,color in zip(subset,('--','-.','solid'),('#0077bb','#d62728','black'))]
fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False)
fig.suptitle(f'Re = {RE_TO_INSPECT}; mesh sensitivity of profiles and flow structures')
save_current_figure(fig, f'lab7_Re{RE_TO_INSPECT}_mesh_profiles_corners');plt.show();plt.close(fig)
'''))
        elif key == "lab8":
            cell = next(c for c in nb.cells if c.cell_type == 'code' and 'RUN_NEW_CPU_EXPERIMENT = False' in c.source)
            cell.source = cell.source.replace("        rows.append({'Re':reynolds,**op.velocity_error(prediction[j],truth)})", """        rows.append({'Re':reynolds,**op.velocity_error(prediction[j],truth)})
        nn = len(data['x']); shape = (len(data['y']), nn); size = shape[0]*shape[1]
        up, vp = prediction[j,:size].reshape(shape), prediction[j,size:].reshape(shape)
        preds = {'Current student POD-DeepONet': (up,vp)}
        figures = w4plots.velocity_comparison(data['x'],data['y'],data['u'],data['v'],
                                               preds,reynolds,'new validation fit')
        figures += [w4plots.velocity_profiles(data['x'],data['y'],data['u'],data['v'],
                                               preds,reynolds,'new validation fit')]
        for suffix,fig in zip(('velocity','profiles'),figures):
            save_current_figure(fig,f'lab8_current_validation_Re{reynolds}_{suffix}')
            plt.show();plt.close(fig)""")
        elif key == "lab9":
            cell = next(c for c in nb.cells if c.cell_type == 'code' and "values={'Uniform objective'" in c.source)
            cell.source += '''\n\nfig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
axes[0].plot(x,f,color='black');axes[0].set(xlabel='Pool position x',ylabel='Illustrative integrand f(x)')
axes[1].plot(x,np.full_like(x,1/N),color='black',label='Uniform probability')
axes[1].plot(x,q,color='#d62728',ls='--',label='Adaptive probability')
axes[1].set(xlabel='Pool position x',ylabel='Probability per native pool point')
for ax in axes:ax.grid(alpha=.2)
fig.legend(*axes[1].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Sampling example: same uniform objective requires importance weights')
save_current_figure(fig,'lab9_sampling_objective');plt.show();plt.close(fig)
'''
        for c in nb.cells:
            if c.cell_type == "markdown":
                c.source = re.sub(r"(week04_cavity_surrogates_deeponet\.pdf)#page=\d+", r"\1", c.source)
        nbformat.validate(nb)
        nbformat.write(nb, path)
        print("updated", filename, flush=True)


def execute(key, timeout=900):
    temp = ROOT / "tmp/week04_outputs"
    for env, part in [("MPLCONFIGDIR", "mpl"), ("IPYTHONDIR", "ipython"),
                      ("JUPYTER_RUNTIME_DIR", "jupyter"), ("JUPYTER_CONFIG_DIR", "config")]:
        folder = temp / part; folder.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(folder)
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                      FLOWMLLAB_W4_OUTPUT=str(RESULTS), JUPYTER_PATH=str(Path(sys.prefix)/"share/jupyter"))
    RESULTS.mkdir(parents=True, exist_ok=True)
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    path = NOTEBOOKS / FILES[key]
    nb = nbformat.read(path, as_version=4)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    start = time.perf_counter()
    client = NotebookClient(nb, timeout=timeout, km=km, resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except Exception:
        nbformat.write(nb, temp / (path.stem + "_failed.ipynb"))
        raise
    nbformat.validate(nb)
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
    nbformat.write(nb, path)
    record = dict(notebook=path.relative_to(ROOT).as_posix(), elapsed_seconds=time.perf_counter()-start,
                  python=sys.version, executed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  code_cells=sum(c.cell_type == "code" for c in nb.cells),
                  png_outputs=sum("image/png" in o.get("data", {}) for c in nb.cells for o in c.get("outputs", [])),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256("\n".join(c.source for c in nb.cells if c.cell_type == "code").encode()).hexdigest(),
                  errors=0, scope="Fresh local CPU Run All; retained evidence explicitly distinguished from new solves/fits; no hosted Colab execution claimed")
    receipt = RESULTS / "execution.json"
    records = json.loads(receipt.read_text()) if receipt.exists() else {}
    records[key] = record
    receipt.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


def refresh_lab4_text(nb):
    for c in nb.cells:
        if c.cell_type == 'markdown':
            c.source = c.source.replace('The refined field plots use display interpolation only',
                                        'The archived field plots used display interpolation only')
            if c.source.startswith('The following fields are linearly interpolated for display only.'):
                c.source = ('The following comparisons use the retained 25-by-25 native fields. '
                            'The constant and diverse lid families are evaluated separately at the '
                            'printed matched case. Filled CFD fields and model contour lines use '
                            'identical physical levels; no upsampling or smoothing is applied. '
                            'The pressure is recovered from velocity and shifted to a common '
                            'zero-mean gauge. Signed pressure errors use a linear scale.')
            if c.source.startswith('## 9. Streamlines and vortex capture'):
                c.source = ('## 9. Streamlines and vortex capture\n\n'
                            'The retained all-case table measures interior vorticity on original '
                            'nodes. The new figures compare signed streamfunction contours for '
                            'CFD, original POD and both correction stages on the same native '
                            '25-by-25 grid. The whole-domain and lower-right corner panels use '
                            'identical physical contour levels. Their symmetric-log color scale '
                            'retains the primary-flow extrema while showing weak recirculation. '
                            'The lower panels show signed interior vorticity and its discrepancy '
                            'on the horizontal centreline, computed with the same discrete '
                            'negative Laplacian of streamfunction. No interpolated extrema or '
                            'decorative vortex-core markers are introduced. A small corner '
                            'feature on this coarse grid is not mesh-independent vortex validation.')
            c.source = c.source.replace(
                'Therefore, the comparison isolates the effect of optimization and additional training data on the same inspected regression set; it is not a new blind benchmark.',
                'The stages change representation, optimization and then training data on the same inspected regression set. The original POD uses one seed and the correction models use three-seed ensembles. These are combined-method comparisons, not an isolated optimizer benchmark or a new blind test.')


def complete_lab4_gallery():
    """Replace displayed bars/smoothed pixels with matched native numerical plots."""
    import nbformat
    path = NOTEBOOKS / FILES['lab4']
    nb = nbformat.read(path, as_version=4)
    if any(c.metadata.get('week04_native_gallery') for c in nb.cells):
        return
    section = nbformat.v4.new_markdown_cell(r'''## Historical seven-method pilot: inspect every algorithm

This gallery re-plots **saved pilot predictions**, not newly trained models.
The original pilot used 25-by-25 fields; 16 training, 4 validation and 6 test
cases per in-distribution lid family; and one declared neural seed (7). Its
120-epoch budget does not establish a universal ranking of architectures.
Nearest-case and RBF predictions use their stored deterministic baseline fits.

| Method | What it predicts and how |
|---|---|
| Nearest case | Reuses the most similar training-case streamfunction in the stored parameter/lid representation. |
| RBF | A radial-basis-function interpolator combines training cases with a distance kernel and polynomial term; it is not a CFD algorithm. |
| Coordinate MLP | A feed-forward network receives parameters and spatial coordinates and returns streamfunction. |
| POD-DeepONet | A parameter branch predicts coefficients of a fixed POD spatial basis. |
| U-Net | An encoder-decoder convolutional network uses local stencils and skip connections to reconstruct a field. |
| FNO | A Fourier neural operator mixes retained Fourier modes and local channels. |
| Physics-weighted FNO | The same operator family adds a physics-related training penalty; the resulting objective and its weight differ. |

For this gallery, every stored model predicts streamfunction. The same discrete
derivatives and prescribed lid give its plotted velocity. Therefore exact wall
velocities do not prove that the interior prediction is accurate. Error maps
retain all native extrema on a linear scale. Shared physical contour levels,
overlaid streamlines and signed profiles expose both agreement and failure.
Error ranges differ between figures, so equal colors do not imply equal errors
and cannot rank models. Compare the numerical maxima, relative norms and
signed physical profiles in addition to each labelled colorbar.

Choose `TRAIN_FAMILY`, `TEST_FAMILY` and `CASE_INDEX` below. The printed table
provides the exact case ID, Re, length, viscosity, reference lid speed and lid
coefficients. `CASE_INDEX=0` is the first retained case, not a best-case choice.
The `ood_shape` and `ood_re` families are extrapolation tests. Inspecting these
public results is useful learning, but choosing a new model after seeing them
requires new independent tests for a new research claim.
''', metadata={'week04_native_gallery': 1})
    code = nbformat.v4.new_code_cell('''# Run All loads retained numeric arrays; this cell does not train seven new models.
TRAIN_FAMILY = 'constant'  # 'constant' or 'diverse'
TEST_FAMILY = 'constant'   # 'constant', 'diverse', 'ood_shape', or 'ood_re'
CASE_INDEX = 0            # 0,...,5 within that test family
pilot_manifest = json.loads((ROOT/'results/cavity_diversity_pilot/manifest.json').read_text())
case_choices = [c for c in pilot_manifest['cases'] if c['family']==TEST_FAMILY and c['split']=='test']
display(pd.DataFrame(case_choices)[['id','Re','U_ref','L','nu','coefficients']])
pilot_case, pilot_reference, pilot_models = w4plots.historical_pilot(
    ROOT, TRAIN_FAMILY, TEST_FAMILY, CASE_INDEX)
print('Exact retained pairing:',pilot_case,'; training family:',TRAIN_FAMILY)
pilot_rows = []
for name, values in pilot_models.items():
    u, v = pilot_reference['u'], pilot_reference['v']
    error = np.hypot(values['u']-u, values['v']-v)
    pilot_rows.append({'method':name,'case_id':pilot_case['id'],'Re':pilot_case['Re'],
                      'relative_velocity_L2':np.linalg.norm(error)/np.linalg.norm(np.hypot(u,v)),
                      'max_vector_error_Uref':error.max()})
    figures = w4plots.velocity_comparison(pilot_reference['x'],pilot_reference['y'],u,v,
        {name:(values['u'],values['v'])},pilot_case['Re'],
        f"retained pilot; {TRAIN_FAMILY} to {TEST_FAMILY}; case {CASE_INDEX}")
    for fig in figures:
        w4plots.reference_speed_labels(fig)
        stem=name.lower().replace(' ','_').replace('-','_')
        save_current_figure(fig, f'lab4_pilot_{TRAIN_FAMILY}_{TEST_FAMILY}_{CASE_INDEX}_{stem}')
        plt.show();plt.close(fig)
display(pd.DataFrame(pilot_rows))
method_names = list(pilot_models)
for start in (0,3,6):
    group={name:(pilot_models[name]['u'],pilot_models[name]['v']) for name in method_names[start:start+3]}
    fig=w4plots.velocity_profiles(pilot_reference['x'],pilot_reference['y'],
        pilot_reference['u'],pilot_reference['v'],group,pilot_case['Re'],
        f'retained pilot; {TRAIN_FAMILY} to {TEST_FAMILY}')
    w4plots.reference_speed_labels(fig)
    save_current_figure(fig,f'lab4_pilot_profiles_{start//3+1}');plt.show();plt.close(fig)
''')
    index = next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'R = ROOT /' in c.source)
    nb.cells[index+1:index+1] = [section, code]
    replace_code(nb, "display(Image(filename=str(R/'week04_2_error_matrix.png')))", '''# The numerical table above retains every training/test combination.
# Compare the physical signed profiles below instead of repeating this table as bars.
display(main[['train','test','model','velocity_percent','worst_percent']])
print('Original POD uses one pilot seed; correction predictions are three-seed ensembles.')
print('The stage comparison changes optimization/representation and then training data; it is not an isolated optimizer benchmark.')
''')
    replace_code(nb, "display(Image(filename=str(ROOT/'figures/Cavity_constant_velocity_pressure.png')))", '''# Native retained fields, no upsampling or smoothing. Change this shared case index.
CORRECTION_CASE_INDEX = 0
correction_cases = {}
for family in ('constant','diverse'):
    case, reference, models = w4plots.stokes_refinement(ROOT, family, CORRECTION_CASE_INDEX)
    correction_cases[family] = (case, reference, models)
    print('Matched correction case:', case)
    predictions = {name:(values['u'],values['v']) for name,values in models.items()}
    figures = w4plots.velocity_comparison(reference['x'],reference['y'],
        reference['u'],reference['v'],predictions,case['Re'],f'{family} lid; retained correction study')
    for i,fig in enumerate(figures):
        w4plots.reference_speed_labels(fig)
        save_current_figure(fig,f'lab4_{family}_correction_method{i+1}');plt.show();plt.close(fig)
    fig=w4plots.velocity_profiles(reference['x'],reference['y'],reference['u'],reference['v'],
                                  predictions,case['Re'],family+' lid')
    w4plots.reference_speed_labels(fig)
    save_current_figure(fig,f'lab4_{family}_correction_profiles');plt.show();plt.close(fig)
    fig=w4plots.pressure_comparison(reference['x'],reference['y'],reference['p'],
        models['Correction / 64 train']['p'],case['Re'],'Correction / 64 train')
    w4plots.reference_speed_labels(fig)
    save_current_figure(fig,f'lab4_{family}_native_pressure');plt.show();plt.close(fig)
''')
    cell = next(c for c in nb.cells if c.cell_type == 'code' and "vortices = pd.read_csv(R/'vortex_comparison.csv')" in c.source)
    # Preserve the all-case vortex audit; replace only the interpolated images.
    prefix = cell.source[:cell.source.index('display(Image')] if 'display(Image' in cell.source else cell.source
    cell.source = prefix + '''\nfor family, (case, reference, models) in correction_cases.items():
    fig = w4plots.streamfunction_vorticity_comparison(reference,models,case['Re'],family+' lid')
    save_current_figure(fig,f'lab4_{family}_native_vortices');plt.show();plt.close(fig)
'''
    cell.outputs = []; cell.execution_count = None
    refresh_lab4_text(nb)
    nbformat.validate(nb)
    nbformat.write(nb, path)
    print('Native gallery added; historical arrays and image files are unchanged.', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--execute", choices=list(FILES))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--complete-lab4-gallery", action="store_true")
    args = parser.parse_args()
    if args.update: update()
    if args.complete_lab4_gallery: complete_lab4_gallery()
    if args.execute: execute(args.execute, args.timeout)
