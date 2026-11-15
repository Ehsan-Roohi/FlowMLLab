"""Build three separate cavity labs and append navigation to earlier lessons.

Only cells bearing this extension's metadata tag are replaced. Earlier cells,
outputs, exercises and historical experiments remain intact. Execute the new
sections separately; this is not a fresh hosted Run All of the older lessons.
"""
from pathlib import Path
import csv, hashlib, json, sys
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
from flowmllab import ghia
from execute_cavity_lesson import execute

TAG='cavity_expansion_20261115'
BOOT='''from pathlib import Path
import sys, subprocess
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents, Path('/content/FlowMLLab')]
             if (p/'flowmllab/ghia.py').is_file()), None)
if ROOT is None:
    candidate = Path('/content/FlowMLLab') if Path('/content').is_dir() else Path.cwd()/'FlowMLLab'
    if not candidate.exists():
        subprocess.run(['git','clone','--depth','1','https://github.com/Ehsan-Roohi/FlowMLLab.git',str(candidate)],check=True)
    if not (candidate/'flowmllab/ghia.py').is_file():
        raise RuntimeError('This checkout predates the cavity extension. Use an updated FlowMLLab checkout after saving your student work; do not overwrite your edited notebook.')
    ROOT = candidate
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from flowmllab import ghia, cavity_course as course
'''


def md(s): return nb.v4.new_markdown_cell(s.strip().replace('\\\\','\\'),metadata={TAG:True})
def code(s): return nb.v4.new_code_cell(s.strip(),metadata={TAG:True})

INTRO=r'''# Lab 5: our CFD against Ghia and seven-Re coverage

This standalone CPU lab follows the original Week 1/1.2 and Week 4 CFD lessons.
It widens the **reference and experimental parameter range** to $Re=100,400,1000,3200,5000,7500,10000$.
Ghia's Tables I and II give 17 samples on each centreline, not a full flow field.
The square cavity has width $L$, lid speed $U$, $Re=UL/\nu$, and a discontinuous
moving lid at its upper corners. A smooth-lid or deep cavity is a different problem.

[Benchmark source and anomaly policy](../../references/ghia/README.md) ·
[Experiment, results and limitations](../../results/cavity_ghia_allre/README.md) ·
[Setup](../../START_HERE.md).

Three suspicious **printed** entries remain in the raw tables. We show both the
unfiltered score and a predeclared screened sensitivity score; we never silently
replace them with a plausible number. A train/test split by individual points
would leak the same Reynolds number into both sets.

For $N$ retained samples, report
$$E_u=\frac{\sqrt{\sum_i(\hat u_i-u_i)^2}}{\sqrt{\sum_i u_i^2}},\qquad
e_{\mathrm{RMS},u}/U=\sqrt{\frac1N\sum_i[(\hat u_i-u_i)/U]^2}.$$
Use the analogous definitions for $v$. The norm denominator is shared over the
profile, so a reference zero crossing causes no pointwise division by zero.
The code stores velocities already divided by $U$.
'''

REFERENCE=[md(INTRO + """

**Start with our computed solutions below.** This notebook now opens the native
FlowMLLab CFD archives, recomputes their errors, and plots CFD curves against
Ghia markers by default. It does not fit a curve through Ghia and call it CFD.
Retained CFD is available at Re=100, 400 and 1000; the four higher-Re fields are
still missing. The later neural section has a different data source and purpose.
"""),code(BOOT),md("""### 1. Our CFD solutions, mesh changes and Ghia validation

Re=100 uses the existing staggered-grid SIMPLE runs on 32/64 cells. Re=400 and
1000 use the existing independent finite-difference streamfunction-vorticity
runs on 32/64/128 cells. Every archive is checked against its recorded SHA256.
For SIMPLE, centreline velocities come from the native MAC faces with the
prescribed wall values. For FD, we interpolate the actual two-dimensional field.

Table A recomputes the profile errors from these fields. Both the original
printed-data score and the predeclared sensitivity score are shown. Native
momentum and vorticity defects belong to different equations; their magnitudes
cannot rank the methods. A passed iteration gate does not establish mesh or
physical-time convergence.
"""),code("""if not (ROOT/'flowmllab/cavity_validation.py').is_file():
    raise RuntimeError('Use FlowMLLab v1.12.1 or later. Save your work, then update the checkout; do not overwrite your edited notebook.')
from flowmllab import cavity_validation as validation
cfd = validation.load_evidence(ROOT)
cfd_scores = pd.DataFrame(validation.score_rows(cfd))
print('Table A: measured errors of our retained CFD against Ghia')
columns = ['Re', 'method', 'cells', 'recorded_converged',
           'u_published_relative_l2', 'v_published_relative_l2',
           'v_screened_relative_l2', 'u_screened_rmse_U', 'v_screened_rmse_U',
           'recorded_steady_defect', 'native_equation']
display(cfd_scores[columns])
print('The relative-L2 columns are fractions (multiply by 100 for percent).')
print('Source archives and hashes:')
display(cfd_scores[['Re','cells','source','sha256']])"""),code("""print('Actual FlowMLLab CFD streamlines: SIMPLE at Re=100; FD at Re=400 and 1000.')
validation.plot_streamlines(cfd)
plt.show()
for re in (100, 400, 1000):
    print(f'Re={re}: our CFD on retained grids versus Ghia, then signed sample errors')
    validation.plot_profiles(cfd, re)
    plt.show()"""),md("""Read the changes in the velocity profiles before reading the scalar score.
At Re=400, the marked printed v value near x/L=0.9063 strongly affects the
all-point norm. Its signed discrepancy remains visible; screening does not
replace the published value. At Re=100, these two SIMPLE grids do not give a
monotone Ghia-error reduction. Report that outcome rather than claiming a
successful convergence study. The original source and histories remain in
[Re=100 evidence](../../results/week01_2_pressure_velocity/README.md) and
[Re=400/1000 evidence](../../results/week01_2_formulations/README.md).

### 2. What we have for neural learning on the Ghia cases

There are two different retained experiments:

* **Full fields:** the existing POD-DeepONet learns 65 by 65 velocity and pressure
  fields from our CFD over Re=100--400. Its blind cases are Re=175, 275 and 375.
  Ghia checks at Re=100 and 400 are not unseen-Re tests of that model.
  [Full-field results and protocol](../../results/pod_deeponet/README.md).
* **All-seven-Re profile study:** a small MLP learns u(y) and v(x) directly from
  Ghia centreline samples. Training uses Re=100,1000,5000,10000; validation uses
  400; complete test cases are 3200 and 7500. Adam, L-BFGS, their sequence, network
  depth and wall-data weights are compared across three seeds. These results do
  not supply the missing high-Re CFD fields or demonstrate a high-Re PINN solve.

The figures below show the frozen baseline, selected network and interpolation
against the held-out Ghia samples. Their lower panels show signed discrepancies
at measured reference locations. Curves are three-seed means; Table B averages
individual-seed errors. The original printed-data sensitivities remain in the
CSV. [Lab 6](W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) gives the complete training
protocol, all seven plots, source models and architecture/optimizer exercises.
"""),code("""course.verify_frozen_models(ROOT)
neural_folder = ROOT/'results/cavity_ghia_allre'
measured = pd.read_csv(neural_folder/'scores.csv')
per_seed = measured.groupby(['model','Re','role','seed'])['rmse_U'].mean().reset_index()
test = per_seed[(per_seed.role == 'test') & per_seed.model.isin(
    ['adam_2','hybrid_5_wall','linear_logRe'])]
print('Table B: mean u/v RMSE divided by U, averaged over individual seeds')
display(test.groupby(['model','Re']).rmse_U.mean().unstack('Re'))
for re in (3200, 7500):
    print(f'Re={re}: held-out Ghia-profile prediction; these curves are not a CFD solve')
    display(Image(filename=str(neural_folder/f're{re}_profiles.png'), width=800))
print('The selected neural model improves over Adam, but log-Re interpolation wins both tests.')"""),md("""### 3. Coverage and the remaining high-Re CFD work

Table C gives all seven reference cases and explicitly identifies the missing
CFD fields. A Ghia table is not a full field, and a profile-regression prediction
does not fill that gap. For each additional CFD case retain grid, time step,
convection, residual history, wall treatment and both velocity profiles. Compare
three grids and a halved time step, and inspect stationarity, mass balance and
vortex locations before using the field for machine learning. Pressure requires
an independent validation: Ghia's velocity samples do not provide one.
"""),code("""print('Table C: reference and retained CFD coverage')
display(pd.DataFrame(course.coverage(ROOT)))
# Reference-only figures are optional: the default plots above contain OUR solutions.
SHOW_REFERENCE_ONLY = False
if SHOW_REFERENCE_ONLY:
    for re in ghia.REYNOLDS:
        print(f'Re={re}: published samples only')
        course.plot_reference(re)
        plt.show()"""),code("""# Optional new CFD: choose one case and audit it before treating it as a field label.
RUN_NEW_CFD = False
RE_TO_SOLVE = 3200
CELLS = 128
if RUN_NEW_CFD:
    from flowmllab.streamfunction_experiments import run_streamfunction
    field = run_streamfunction(reynolds=RE_TO_SOLVE, cells=CELLS,
                              dt=0.0005, max_steps=400000, progress=False)
    display({k:field[k] for k in ['status','steps','steady_vorticity_linf',
                                'continuity_linf','ghia_u_relative_l2','ghia_v_relative_l2']})
    display(course.compare_field(RE_TO_SOLVE,field['x'],field['y'],field['u'],field['v']))
else:
    print('Retained CFD and neural results above were loaded and evaluated; no new CFD run requested.')"""),md(r"""### Specific student tasks

1. **Table A and the three CFD figures:** report the u/v errors for every retained
   grid. Locate the largest signed discrepancy and explain the nonmonotone
   Re=100 result. For Re=400, explain the difference between raw and screened v
   norms using the marked sample. Do not choose a mask from your observed error.
2. **Spatial evidence:** identify the primary vortex in each actual CFD field.
   Describe which corner structures the present grids resolve. Ghia's two
   centreline tables alone cannot validate a complete vortex map or pressure.
3. **Table B and the two test figures:** compare Adam, the selected network and
   interpolation separately at Re=3200 and 7500. State where the neural model
   gains and loses, and distinguish seed-mean error from error of the mean curve.
4. **Information and test sets:** list the different training labels and blind
   Re values of the full-field POD-DeepONet and the centreline MLP. Explain why
   a successful profile fit is not independent validation of a flow solver.
5. **Table C:** write an admission decision for each Re. Design a three-grid and
   two-time-step CFD experiment for one missing high-Re field. Keep missing,
   nonstationary and failed cases explicitly marked.
6. **Algorithm comparison:** use the original Week 1.2 matched SIMPLE/PISO/PIMPLE
   runs to compare pressure solves and elapsed time at equal achieved accuracy.
   SIMPLE iterations are not physical time. Do not infer an algorithm ranking
   by comparing the different Re/method combinations in this notebook.
""")]

MLINTRO=r'''# Lab 6: whole-Re generalization, network depth and optimization

This additional CPU experiment learns **centreline profiles** from Ghia samples;
it does not solve a PDE, learn pressure, or validate a 2D field. Complete
[Lab 5: Ghia validation](W4_Lab5_All_Re_Ghia_Validation.ipynb) first. The earlier
CFD/field-surrogate notebooks retain their existing data and Reynolds range.

Inputs are normalized wall position, $\log_{10} Re-3$ and a u/v component tag.
Train on Re=100,1000,5000,10000; select architecture/optimizer/weights using only
Re=400; freeze the choice before scoring Re=3200 and 7500. All points from a Re
belong to one split. These public teaching tests are now disclosed: further tuning
against them is exploratory and requires a new independent test for a research claim.

We ran three initializations (7,23,61), five configurations, and 1200 joint
function/gradient evaluations per fit. Width is 24; compare 2 versus 5 hidden
layers (721 versus 2521 parameters), Adam, L-BFGS, and Adam then L-BFGS.
The last variant triples training weight at positions below 0.12 or above 0.88,
renormalizing each case/component to equal total weight. These are **regression
data weights**, not PINN PDE/boundary loss weights. Depth is not parameter matched.
Training uses Xavier initialization and tanh. Hard endpoint lifts enforce
$\hat u(t)=t+t(1-t)N_\theta$ and $\hat v(t)=t(1-t)N_\theta$.

Adam rescales each coordinate's first moment by its second moment and is a useful
initial stage. L-BFGS builds curvature information from parameter/gradient
differences; its line search needs a fixed deterministic objective. More layers
can represent sharper structure but also make optimization harder. Weighting
can prioritize boundary layers while worsening the core. **None is guaranteed
to beat interpolation.** The retained results below show both gains and losses.
'''
ML=[md(MLINTRO),code(BOOT),code('''frozen = course.verify_frozen_models(ROOT)
folder = ROOT/'results/cavity_ghia_allre'
scores = pd.read_csv(folder/'scores.csv')
timing = pd.read_csv(folder/'timing.csv')
display(pd.DataFrame([{'Re':int(k),'role':v} for k,v in frozen['roles'].items()]))
print('Frozen selection:', frozen['selected'], '|', frozen['selection'])
# First average u/v within a seed, then report seed mean and spread.
per_seed = scores.groupby(['model','Re','role','seed'])[['rmse_U','relative_l2']].mean().reset_index()
summary = per_seed.groupby(['model','Re','role']).agg(
    rmse_mean=('rmse_U','mean'), rmse_seed_sd=('rmse_U','std'),
    relative_l2_mean=('relative_l2','mean')).reset_index()
display(summary[summary.role=='test'])
display(timing.groupby('model').agg(parameters=('parameters','first'),
       evaluations=('evaluations','max'),median_seconds=('seconds','median')))
print('Timing is local and load-sensitive; equal evaluations are not equal FLOPs or time.')'''),code('''for re in ghia.REYNOLDS:
    print(f'Re={re}, {ghia.case_roles()[re]}: physical profiles and signed discrepancies')
    display(Image(filename=str(folder/f're{re}_profiles.png'), width=800))'''),md(r'''### What the retained experiment found

Re=400 selected the five-layer, wall-weighted Adam-to-L-BFGS variant. On the two
test cases its mean u/v RMSE is about 0.01974 U (Re=3200) and 0.00563 U (Re=7500).
The two-layer Adam baseline gives 0.03395 U and 0.03800 U. However, piecewise linear
interpolation in **log Re**, using only training cases and the same screened
samples, gives 0.01629 U and 0.00518 U: it wins both tests. The network improves
on its neural baseline but does not justify replacing this simple method.
These are averages of the per-seed, per-component errors, not errors of an ensemble.
The plotted neural curves are three-seed means. Raw printed-data sensitivity
scores, including the suspicious Re=3200 entry, remain in `scores.csv`.

### Specific student tasks: architecture, optimization and weights

1. Reproduce the **test table above** from the CSV and show the standard deviation
   across seeds. Compare Adam-2 with hybrid-2, hybrid-2 with hybrid-5, and hybrid-5
   with hybrid-5-wall. State what changes in each comparison and what is confounded.
2. Use the u/v plots to locate the largest signed error in each held-out case.
   Explain why a lower training loss need not imply a better withheld-Re profile.
3. Before running a new model, declare width, depth, seed list, learning rate,
   weighting and budget. Tune only on Re=400. Try widths chosen to approximately
   match parameter count; also compare wall-clock cost with warm-up and repeats.
4. Vary the wall weight from 1 to 3 or 5 and report wall and core errors separately.
   Explain why a wall-weighted training loss must not replace an unweighted audit.
5. Compare interpolation in Re with interpolation in log Re. Choose that convention
   using validation only, then freeze it. Do not claim either is a CFD solver.
6. For a future full-field DeepONet, first obtain independently admitted fields
   for every train/validation/test Re. Keep all grid points from one Re together;
   fit normalization and the POD basis on training fields only. Compare against
   field interpolation and POD regression before interpreting network depth.
'''),code('''# A small optional new training run, kept separate from the frozen comparison.
TRAIN_NEW_PROFILE_MODEL = False
if TRAIN_NEW_PROFILE_MODEL:
    from flowmllab.ghia_surrogate import ModelConfig, fit, predict
    trial_config = ModelConfig(name='student_trial',depth=3,width=20,
                               optimizer='hybrid',weighting='uniform',evaluations=1200)
    trial = fit(trial_config,seed=17)
    reference = ghia.reference(400)
    for component in ('u','v'):
        t=reference['y' if component=='u' else 'x']; keep=ghia.valid_mask(400,component)
        error=predict(trial,400,t,component)-reference[component]
        print(component,'validation RMSE/U',np.sqrt(np.mean(error[keep]**2)))
else:
    print('Showing retained runs; no test-informed retraining.')''')]


def append_section(path,cells):
    document=nb.read(path,as_version=4)
    document.cells=[c for c in document.cells if not c.metadata.get(TAG)]
    section=nb.v4.new_notebook(cells=[nb.from_dict(json.loads(json.dumps(c))) for c in cells])
    execute(section,ROOT)
    document.cells.extend(section.cells)
    document.metadata[TAG]={'new_section_executed':True,'older_cells_rerun':False,
                           'execution':'sequential plain-Python; not a hosted kernel certification'}
    nb.validate(document);nb.write(document,path)


def deep_notebook():
    cells=[md(r'''# Week 13 extension: regional accuracy in a D/W=5 cavity

**Authors' research evidence:** Ehsan Roohi and Christopher J. McDevitt,
[Regional accuracy of PINNs for recirculation in a deep lid-driven cavity](https://arxiv.org/abs/2610.38814v2).
This separate CPU notebook complements the existing Week 13 square/D=2/D=2.2
notebook. It loads retained CFD/PINN fields, recomputes errors, and studies weak
recirculation, pressure and derivatives. It does not retrain the paper's networks.

**Geometric depth D/W=5 is not five network layers.** The four cases are Re=100,
250,500,1000; their Gaussian-regularized lid is not Ghia's square-cavity lid.
Width W, lid scale U, density rho, and height h measured from the floor set the
units. Arrays have shape [h,x], 605 by 121, and exclude the exact walls by 0.002 W.

[Lecture](../../lectures/week13_rectangular_cavity_pinn.pdf) ·
[Data and provenance](../../data/week13_d5/README.md) ·
[Results](../../results/week13_d5/README.md) · [Setup](../../START_HERE.md).
Prerequisites: Week 1 validation, Week 1.3 optimization, Week 13 PINN basics.
The CPU route needs NumPy, pandas and Matplotlib; no GPU or TensorFlow is required.
'''),code(BOOT),md(r'''## 1. Verify the field identity before comparing

The loader checks the compressed bundle hash. The original arrays have been
retained exactly; only redundant speed arrays were omitted. The common-grid
Re=1000 field is the **control** trajectory. It is not the refined result from
the separate 161 by 801 assessment. CFD times are 120,160,160,317 respectively;
these are reference-solver times, not neural optimizer iterations.
'''),code('''from flowmllab.deep_cavity_audit import load_bundle, evaluate, masks, metrics, horizontal_profile
fields, provenance = load_bundle(ROOT/'data/week13_d5')
print('Geometry:',provenance['aspect_ratio_D_W'],'| shape [h,x]:',provenance['grid_shape_h_x'])
print('CFD times:',provenance['CFD_times'])
print('Pressure convention:',provenance['pressure'])
print('Bundle SHA256:',provenance['data_sha256'])
table=pd.DataFrame(evaluate(fields))
display(table[['Re','region','count','velocity_RMS_U','CFD_velocity_RMS_U',
               'velocity_NRMSE_percent','pressure_RMS_rhoU2','vorticity_NRMSE_percent']])'''),md(r'''## 2. The physics and the error definitions

A streamfunction imposes $u=\partial\psi/\partial h$ and
$v=-\partial\psi/\partial x$, so continuity is satisfied analytically.
Momentum residuals and wall conditions still need to be satisfied; differentiating
velocity to obtain $\omega=\partial v/\partial x-\partial u/\partial h$
can amplify a small field error. The supplied vorticity is sampled finite
differences on the common grid, not a comparison of native AD/spectral derivatives.

For a fixed region R with N equally weighted retained nodes:
$$e_{u,R}=\sqrt{N^{-1}\sum_{i\in R}[(\hat u_i-u_i)^2+(\hat v_i-v_i)^2]},\quad
E_{u,R}=100\,\frac{e_{u,R}}{\sqrt{N^{-1}\sum_{i\in R}(u_i^2+v_i^2)}}.$$
All velocities in the file are normalized by U. A tiny absolute floor error
can coexist with a measurable relative error because the reference flow is tiny.
Neither quantity is a statistical uncertainty estimate or a numerical noise floor.

Pressure is fixed to one whole-interior gauge before regional comparisons.
Let $\delta p_i=\hat p_i-p_i$, $b_R=\langle\delta p\rangle_R$ and
$s_R=\sqrt{\langle(\delta p-b_R)^2\rangle_R}$. Then
$$e_{p,R}^2=b_R^2+s_R^2.$$
Report RMS, signed offset and local shape together. Re-centering each region
silently would remove the offset and change the stated metric.
'''),code('''identity = table.pressure_RMS_rhoU2**2 - (
    table.pressure_signed_mean_offset_rhoU2**2 + table.pressure_local_shape_RMS_rhoU2**2)
print('Largest pressure decomposition residual:', abs(identity).max())
display(table[['Re','region','pressure_RMS_rhoU2','pressure_signed_mean_offset_rhoU2',
               'pressure_local_shape_RMS_rhoU2']])'''),md(r'''## 3. Inspect every retained Reynolds number

For each Re, CFD and PINN speed have the **same** full-range symmetric-log color
scale so weak lower vortices remain visible. The signed u discrepancy uses a
separate linear scale. Profiles compare u, globally gauged p and sampled vorticity
at three actual retained heights; no curve outside the retained mesh is invented.
Do not infer a smaller error from a different color transform.
'''),code('''for re in (100,250,500,1000):
    print(f'Re = {re}: matched geometry/lid, common sampled grid')
    display(Image(filename=str(ROOT/f'results/week13_d5/re{re}_fields.png'),width=650))
    display(Image(filename=str(ROOT/f'results/week13_d5/re{re}_profiles.png'),width=900))'''),md(r'''## 4. Optimization history: what improved, and what was not isolated

The paper compares accumulated workflows, including transferred initialization,
regional collocation, residual scaling/envelope choices and a SOAP-to-self-scaled
Broyden refinement stage. Equal final refinement steps do not erase unequal prior
training histories. The pressure and vorticity rankings can differ from velocity.
Read the source and provenance before attributing gains to one optimizer.

In the separate 161 by 801 Re=1000 assessment, the control-to-refined interior
velocity discrepancy decreases from 5.624% to 2.658%, and pressure RMS from
0.002793 to 0.001167 rho U squared. This is retained research evidence, not a new
controlled optimizer ablation. The common-grid results above use a different
sampling grid and must not be mixed into the same ranking.

The public [paper ancillary analysis code](https://arxiv.org/abs/2610.38814v2)
and [DeepPlasma cavity implementation](https://github.com/cmcdevitt2/DeepPlasma/tree/fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b/LDC)
provide the original context. FlowMLLab's existing `qa/WEEK13_PINN_MATRIX_PROTOCOL.md`
and runners provide its separate Adam/SSBroyden2 training exercise. This notebook's
portable code is `flowmllab/deep_cavity_audit.py`; it reproduces field assessment,
not the full optimizer/RNG checkpoint history of the paper.
'''),code('''workflow = pd.read_csv(ROOT/'results/week13_d5/workflow_metrics.csv')
display(workflow[workflow.region=='Interior'][['workflow','velocity_NRMSE_percent',
    'pressure_RMS_rhoU2','vorticity_RMS_UoverW']])
print('Frozen author assessment on 161x801; one trajectory per workflow, unequal histories.')'''),md(r'''## 5. Engineering exercises and required submission

Submit your notebook, the two tables below, four physical-profile figures and a
one-page interpretation. Keep signs, units, sample counts and reference identity.

1. **Table A: weak-flow audit.** For every Re, report Interior and Floor_0.4 sample
   count, absolute vector RMS, reference RMS and relative error. Verify their
   algebraic relation. Explain why $10^{-9}U$ alone does not establish nine-digit accuracy.
2. **Region sensitivity.** Change the floor cutoff from 0.4 to 0.65 W using `masks`
   and compare the rankings without changing the fields. Plot u and v at a
   retained height inside each region. Report the actual height returned by the code.
3. **Pressure gauge.** Add a constant 0.001 to a copy of the predicted pressure.
   Show how global RMS, regional offset and local shape respond. Then restore the
   original arrays; do not modify the supplied NPZ.
4. **Derivative sensitivity.** Recompute vorticity with second-order differences
   and compare with the retained sampled vorticity. Repeat after subsampling the
   common grid by two. This tests postprocessing resolution, not native CFD mesh convergence.
5. **Table B: workflow trade-offs.** From `workflow_metrics.csv`, rank all four
   workflows by velocity, pressure RMS and vorticity separately in Interior,
   Upper-left and Floor_0.4. Identify where a workflow improves one variable and
   loses on another. Do not hide a weak corner behind a full-domain norm.
6. **Design an isolated experiment.** Fix lid, Re, geometry, initialization,
   collocation, precision and audit samples. Compare depth 2/5 at matched parameter
   counts, Adam/L-BFGS/SSBroyden2 with counted evaluations, and uniform/regional
   weights one at a time. Use at least three seeds and an untouched validation
   reference. Freeze collocation and weights during each quasi-Newton stage;
   otherwise the line-search objective changes. No new paper-quality gain is
   claimed until this experiment and the reference checks are actually run.
'''),code('''# Student starting point: an actual retained profile, not an interpolated floor point.
RE = 1000
HEIGHT = 0.3
fig,ax=plt.subplots(figsize=(7,3.8))
for method,style in [('cfd','k-'),('pinn','b--')]:
    x,values,actual_h=horizontal_profile(fields,RE,method,'u',HEIGHT)
    ax.plot(x,values,style,label=method.upper())
ax.set(xlabel='x/W',ylabel='u/U');ax.grid(alpha=.2)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
fig.subplots_adjust(top=.84,bottom=.17)
print('Actual retained h/W:',actual_h)
plt.show()''')]
    doc=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},TAG:True})
    doc.cells[0].source += '\n\n[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb)'
    execute(doc,ROOT);nb.write(doc,ROOT/'notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb')


def main():
    ref=ROOT/'references/ghia';ref.mkdir(parents=True,exist_ok=True)
    with (ref/'published_samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['Re','component','coordinate','value','flagged']);writer.writeheader()
        for re in ghia.REYNOLDS:
            for c in ('u','v'):
                t=ghia.GHIA[re]['y' if c=='u' else 'x'];m=ghia.valid_mask(re,c)
                for i,(x,v) in enumerate(zip(t,ghia.GHIA[re][c])):
                    writer.writerow(dict(Re=re,component=c,coordinate=x,value=v,flagged=not m[i]))
    for filename,cells in [
        ('W4_Lab5_All_Re_Ghia_Validation.ipynb',REFERENCE),
        ('W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb',ML)]:
        doc=nb.v4.new_notebook(cells=[nb.from_dict(json.loads(json.dumps(c))) for c in cells],
            metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                      'language_info':{'name':'python'},TAG:True})
        doc.cells[0].source += f'\n\n[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/{filename})'
        execute(doc,ROOT);nb.write(doc,ROOT/'notebooks/week04'/filename)
        print('Built separate lab',filename,flush=True)
    navigation=[md('## Optional continuation: all seven Ghia Reynolds numbers\n\nThe original cells, outputs and worked-example scope above are preserved. New material lives in two separate notebooks: [Lab 5: seven-Re validation and CFD qualification](../week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) and [Lab 6: CPU profile surrogates, network depth, optimizers and weights](../week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb). The profile study does not imply that missing high-Re full fields have been validated. [Retained results](../../results/cavity_ghia_allre/README.md).')]
    for name in ['week01/03_cavity_ghia.ipynb','week01_2/W1_2_Cavity_Pressure_Velocity.ipynb',
                 'week04/W4_Lab1_CFD_Data_Production_Student.ipynb',
                 'week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb',
                 'week04/W4_Lab3_DeepONet_Cavity_Student.ipynb','week04/W4_1_Classical_ROM_Cavity.ipynb',
                 'week04/W4_Lab4_Grid51_Validation.ipynb','week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb']:
        append_section(ROOT/'notebooks'/name,navigation)
    deep_notebook()
    path=ROOT/'notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb'
    doc=nb.read(path,as_version=4);doc.cells=[c for c in doc.cells if not c.metadata.get(TAG)]
    doc.cells.append(md('## Continue to the authors\' D/W=5 research cavity\n\nThe separate [D/W=5 regional-accuracy notebook](W13_Deep_Cavity_D5_Regional_Accuracy.ipynb) now supplies matched CFD/PINN fields for Re=100,250,500,1000, signed profiles, regional velocity/pressure/vorticity metrics and the retained optimization-workflow comparison. It complements the D=1/2 matrix and the near-matched D=2.2 case above. Geometric depth and network depth are separate variables.'))
    nb.write(doc,path)
    print('Built D/W=5 notebook and preserved earlier lessons',flush=True)


if __name__=='__main__':
    if '--only-validation' in sys.argv:
        filename='W4_Lab5_All_Re_Ghia_Validation.ipynb'
        doc=nb.v4.new_notebook(cells=[nb.from_dict(json.loads(json.dumps(c))) for c in REFERENCE],
            metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                      'language_info':{'name':'python'},TAG:True})
        doc.cells[0].source += f'\n\n[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/{filename})'
        execute(doc,ROOT)
        with (ROOT/'notebooks/week04'/filename).open('w',encoding='utf-8',newline='\n') as handle:
            nb.write(doc,handle)
        print('Rebuilt the validation lab only; other notebooks unchanged.',flush=True)
    else:
        main()
