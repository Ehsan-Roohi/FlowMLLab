"""Create the new field-neural lab, preserving previous notebooks."""
from pathlib import Path
import sys
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'qa')]
from build_cavity_expansion_lessons import BOOT
from execute_cavity_lesson import execute

def build():
    cells=[]
    md=lambda s:cells.append(nb.v4.new_markdown_cell(s.strip().replace(chr(92)*2,chr(92))))
    code=lambda s:cells.append(nb.v4.new_code_cell(s.strip()))
    md(r"""# Lab 8: full-field cavity learning across Reynolds number

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb)

Predict complete velocity fields and inspect their physical discrepancies.
This new companion follows Lab 7. Earlier notebooks and the historical
Ghia-profile MLP are preserved. Run All audits saved models on CPU; it does not
start training or HPC jobs.

[Protocol](../../docs/CAVITY_FIELD_NEURAL_PROTOCOL.md) |
[Results](../../results/cavity_field_deeponet/README.md) |
[CFD audit](../../results/cavity_highre_ghia/README.md) |
[Related studies](../../docs/CAVITY_ALL_RE_NEURAL_LITERATURE.md)

## 1. What is being learned?

The target is our steady 2-D square-cavity velocity on a 513-by-513 grid,
with a constant moving lid and its corner discontinuity. This is a specified
numerical solution, not an exact mesh-independent field. Pressure is not predicted.

Ghia supplies sparse centreline observations, used here only for comparison.
Training labels are our full CFD fields. A branch maps a scalar Re to coefficients
of a fixed POD trunk. This restricted POD-DeepONet is not a data-free PINN or
an operator over arbitrary boundary functions. Every location of each held-out
Re stays outside fitting and model selection.""")
    code(BOOT+"""
import json
from flowmllab import cavity_field_operator as op
from flowmllab import cavity_field_evidence as evidence
OUT = ROOT/'results/cavity_field_deeponet'
selection = json.loads((OUT/'frozen_selection.json').read_text())
display(pd.DataFrame([
    {'Role':'Training','Re':selection['train_re']},
    {'Role':'Validation','Re':selection['validation_re']},
    {'Role':'Test','Re':selection['test_re']}]))
assert not set(selection['train_re']) & set(selection['test_re'])
assert not set(selection['validation_re']) & set(selection['test_re'])
print('Selected:',selection['selected_configuration'],'POD rank:',selection['rank'])
cases, info = evidence.load(ROOT)
""")
    md(r"""## 2. Spatial representation, architecture and optimization

Stack both velocity components into $\mathbf q=[u_1,\ldots,u_N,v_1,\ldots,v_N]^T$.
The training mean $\overline{\mathbf q}$ and orthonormal POD modes form the trunk:

$$\widehat{\mathbf q}(Re)=\overline{\mathbf q}
 +\sum_{k=1}^{r}a_k(Re)\boldsymbol\phi_k,\qquad
 \mathbf a(Re)=\mathbf s\odot f_\theta(\log_{10}Re-3).$$

Here $N=513^2$, $r$ is the training-only rank, $\theta$ the tanh branch parameters,
and $\mathbf s$ a training-only coefficient scale. The training loss is

$$J(\theta)=\frac{1}{n_{\rm tr}r}\sum_{j=1}^{n_{\rm tr}}\sum_{k=1}^{r}
 [f_{\theta,k}(z_j)-a_{jk}/s_k]^2.$$

Global scaling uses one RMS for all coefficients. Per-mode scaling uses each
mode's RMS, emphasizing low-energy structures. This can improve details or
harm generalization; it is a hypothesis to test.

Adam adapts updates using gradient moments. L-BFGS approximates curvature from
recent parameter and gradient changes. The hybrid warms up with Adam then
refines one fixed full-batch objective. Line-search evaluations count toward
the same budget. Five layers do not automatically outperform two.""")
    code("""
validation = pd.read_csv(OUT/'validation_scores.csv')
display(validation[['method','seed','Re','relative_l2','parameters','evaluations']])
display(validation.groupby(['method','Re'],sort=False).relative_l2.agg(['mean','min','max']))
display(pd.read_csv(OUT/'validation_baselines.csv'))
""")
    md(r"""## 3. Diagnose the basis before blaming the optimizer

The first nine-training-case round is retained in selection_round_1.
At validation Re=400, its optimal POD projection already had 3.2510% error.
No branch optimizer can remove that spatial-basis floor. Independent training
cases at Re=500 and 750 filled the low-Re gap. Validation/test cases and the
candidate budget remained unchanged. This amendment used validation, before
any neural test scoring; the benchmark physics was already disclosed.

The target projection is
$$\mathbf q_{\rm proj}=\overline{\mathbf q}+\Phi\Phi^T
(\mathbf q_{\rm true}-\overline{\mathbf q}).$$
It uses the true field and is a representation lower bound, not a predictor.""")
    code("""
decision = json.loads((OUT/'selection_round_1/ROUND_DECISION.json').read_text())
display(pd.DataFrame([decision]))
display(pd.read_csv(OUT/'projection_lower_bound.csv'))
""")
    md(r"""## 4. Matched-information tests and physical profiles

Both baselines interpolate exactly the training fields, linearly in Re or
log(Re). The convention is selected on validation; both test scores are shown.
Neural curves average the three predeclared seeds of the selected configuration,
rather than choosing the best test seed.

$$E_{\mathbf u}=\sqrt{\frac{\sum_i[(\widehat u_i-u_i)^2+
(\widehat v_i-v_i)^2]}{\sum_i(u_i^2+v_i^2)}}.$$

This measures vector discrepancy, not merely speed discrepancy, and avoids
pointwise division by zero velocity. The field plots share speed limits and
a linear vector-error scale. Signed centreline errors and actual Ghia samples
are retained; a cross marks the predeclared suspicious printed entry.""")
    code("""
test_scores = pd.read_csv(OUT/'test_scores.csv')
reported = test_scores[test_scores.method.isin([
    'selected_three_seed_mean','linear_in_Re','linear_in_log_Re'])]
display(reported[['method','Re','relative_l2','rmse_U','linf_U',
                  'u_relative_l2','v_relative_l2']])
for reynolds, fields in cases.items():
    evidence.plot_fields(fields,reynolds)
    plt.show()
    evidence.plot_profiles(fields,reynolds)
    plt.show()
""")
    md(r"""## 5. Physical constraints and numerical limits

Linear combinations preserve matched wall velocities and discrete continuity.
They generally do not satisfy nonlinear Navier-Stokes at a new Re.
Re-evaluate the vorticity-transport, Poisson, wall and continuity defects below.

Also compare surrogate error with 256-to-512 mesh sensitivity on common native
points. The mesh difference is not an exact-error bound or confidence interval.
It can exceed neural error; excellent agreement with one discretization does
not establish physical accuracy. A steady root does not establish stability.""")
    code("""
physical = pd.DataFrame(evidence.diagnostics(cases))
display(physical[['Re','method','relative_velocity_l2','steady_vorticity_linf',
                  'poisson_linf','continuity_linf','wall_linf']])
mesh = pd.read_csv(ROOT/'results/cavity_highre_ghia/mesh_sensitivity.csv')
display(mesh[(mesh.cells==256)&mesh.Re.isin(op.TEST)][[
    'Re','cells','reference_cells','velocity_relative_l2','u_linf_U','v_linf_U']])
""")
    md(r"""## 6. Overlaid flow structures, vorticity and reconstructed pressure

The next six comparisons use Re=1000 (a training-case reconstruction) and
Re=7500 (the held-out test). Black solid curves are CFD and orange dashed
curves are the frozen three-seed POD-DeepONet mean. Streamlines use matched
seeds; all contour levels are identical between methods and between Re.

Vorticity is derived from each velocity field using the same centered curl,

$$\omega=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y},\qquad
E_\omega=\frac{\|\widehat\omega-\omega_{\rm CFD}\|_2}{\|\omega_{\rm CFD}\|_2}.$$

Here $u,v$ are velocity components, $x,y$ are positions, $L$ is the cavity
width and $U$ the lid speed. Plotted vorticity is nondimensionalized by $U/L$.
Only the common 511-by-511 interior nodes are used; there is no wall extrapolation.
Vorticity is not a separately trained output. At Re7500 its discrepancy is
0.2931%, larger than the 0.0844% velocity discrepancy: differentiation exposes
small-scale errors. Inspect signed extrema and their locations as well as L2.

Pressure is reconstructed from each velocity by integrating its steady momentum
pressure gradient in a least-squares sense. One zero-interior-mean gauge is used
per entire field. Pressure is **postprocessed, not learned or independently
pressure-validated**. At Re7500 its neural/CFD discrepancy is 0.05373%, while
256-to-512 CFD pressure sensitivity is 8.90%. Excellent surrogate agreement
with one discretization does not remove discretization uncertainty.

RGB fields share one full-range symmetric-log scale for each variable across
both Re and both methods; the linear thresholds are 1 for vorticity and 0.02
for pressure. Signed differences use the explicitly shown linear scales.
The code verifies the retained figure hashes before displaying them.""")
    code("""
import hashlib
from flowmllab import cavity_comparison as comparison
GALLERY = OUT/'flow_comparison'
receipt = json.loads((GALLERY/'verification.json').read_text())
for relative, expected in receipt['source_hashes'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == expected
for name, expected in receipt['output_sha256'].items():
    assert hashlib.sha256((GALLERY/name).read_bytes()).hexdigest() == expected
for stem in ['streamlines_overlay', 'corner_streamlines_overlay',
             'vorticity_contours_overlay', 'vorticity_fields_and_signed_error',
             'pressure_contours_overlay', 'pressure_fields_and_signed_error']:
    display(Image(filename=str(GALLERY/(stem+'_Re1000_Re7500.png'))))
display(pd.read_csv(GALLERY/'vorticity_metrics.csv'))
display(pd.read_csv(GALLERY/'comparison_metrics.csv'))
display(pd.read_csv(GALLERY/'pressure_mesh_sensitivity.csv'))
# Independently re-evaluate the derivative error on the already-loaded held-out velocities.
fields = cases[7500]
omega_cfd = comparison.vorticity(fields['CFD'])['w']
omega_nn = comparison.vorticity(fields['Selected NN'])['w']
recomputed_error = np.linalg.norm(omega_nn-omega_cfd) / np.linalg.norm(omega_cfd)
recorded = pd.read_csv(GALLERY/'vorticity_metrics.csv').set_index('Re').loc[7500,'relative_l2']
np.testing.assert_allclose(recomputed_error, recorded, rtol=1e-11)
print(f'Re7500 independently recomputed vorticity discrepancy: {100*recomputed_error:.4f}%')
""")
    md(r"""## 7. Historical computing: time, hardware price and run cost

Ghia, Ghia and Shin used an **Amdahl 470 V/6** and coupled strongly implicit
multigrid. They report 1.5 CPU minutes at Re1000 on 129-by-129 nodes, and
16-20 minutes for high-Re cases up to Re10000 on 257-by-257 nodes
([paper, pp. 387 and 410](https://doi.org/10.1016/0021-9991(82)90058-4)).
The [Computer History Museum](https://www.computerhistory.org/revolution/mainframe-computers/7/169/677)
lists this 1975 mainframe at 3.5 MIPS, up to 8 MB and \$3.75 million.
As a dated desktop example, the [October 2024 Mac mini M4 launch](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/)
started at \$599 with 16 GB and 10 CPU cores. These are nominal hardware prices,
not inflation-adjusted prices or per-solve bills; the Mac did not produce our
Unity timings. MIPS is not modern floating-point throughput.

The table below sums actual one-core Unity Newton-solve wall timers for the
continuation sequence from an available Re1000 seed through Re10000. It excludes
seed creation, queue, transfer, training and temporal checks. Ghia's CPU timing,
initial state and algorithm differ, so dividing those times does not establish
a controlled speedup. No monetary run charge was recorded.

For an explicitly assumed one-core rate $r$ (currency per core-hour), a scenario
cost is $C=r\,t/3600$, with $t$ in seconds. Keep scenario estimates separate
from invoices and from the hardware purchase price.""")
    code("""
history = json.loads((GALLERY/'historical_cost_context.json').read_text())
timing = pd.DataFrame(history['flowmllab_timing'])
timing['summed_solve_wall_minutes'] = timing.summed_solve_wall_seconds / 60
display(timing[['nodes', 'core_allocation', 'summed_solve_wall_seconds',
                'summed_solve_wall_minutes', 'last_continuation_step_wall_seconds']])
ASSUMED_RATE_PER_CORE_HOUR = 0.10  # a classroom scenario, NOT Unity's price
scenario = timing[['nodes', 'summed_solve_wall_seconds']].copy()
scenario['illustrative_cost'] = scenario.summed_solve_wall_seconds / 3600 * ASSUMED_RATE_PER_CORE_HOUR
display(scenario)
print('Scenario only: excludes seed creation and all unrecorded costs; not an invoice.')
""")
    md(r"""## 8. Engineering exercises and required deliverables

**A — Optimizer/depth/weighting.** Using Section 2, report mean and seed range
at each validation Re for all five configurations. Compare Adam versus hybrid
at fixed depth, two versus five layers, and global versus per-mode scaling.
Identify improvements and deterioration, with parameter counts and evaluation
budgets. Do not select the model using Section 4 tests.

**B — Training information.** Explain the 3.2510% projection floor in Section 3.
Compare first-round and current validation errors after adding Re=500 and 750.
The training information changes; do not attribute this effect to optimization.

**C — Physical error.** For each test Re, report vector and component relative
L2 errors, plus the maximum signed centreline discrepancy and its location.
Compare the selected three-seed mean with both baselines. Report wins and
losses. Supply signed u/v profile plots, not bar charts.

**D — Numerical limits.** From Section 5 compare surrogate error with mesh
sensitivity, and wall/continuity errors with nonlinear residuals. Explain
why none of exact continuity, a small supervised loss or a small mesh
difference alone proves an accurate physical solution.

**E — Deliberate CPU training.** Enable the cell below and change one setting:
depth, width, scaling or Adam budget. Train only on the training POD coefficients
and score validation cases. Record runtime and parameter count. Published test
results are already visible, so further test-based experimentation is exploratory.
Predeclare fresh withheld cases for a new blind research claim.

**F — Flow structure and derivatives.** From Section 6 identify the primary
and corner vortices at both Re. Explain why Re1000 is not a held-out validation.
Report the vorticity L2 and maximum signed discrepancy, including its $(x/L,y/L)$
location. Compare velocity, vorticity and pressure errors with the mesh-sensitivity
rows. Explain why derivatives amplify differences and why the pressure plot is
not an independently trained/validated pressure result. Keep all signs and scales.

**G — Cost accounting.** From Section 7 distinguish the final continuation step
from the complete timed sequence. Recalculate the scenario at 0.05 and 0.50 per
core-hour. List the missing costs and explain why neither a price ratio nor a
ratio of the historical and current timings is a controlled speedup. Explain
what matched grid, tolerance, initialization, precision and timing scope would
be needed for such a claim.

Submit one CSV for A-D and F-G, the two signed physical-profile comparisons, and a short
interpretation separating representation, optimization and discretization errors.
Record software version and model/data hashes.""")
    code("""
RUN_NEW_CPU_EXPERIMENT = False
if RUN_NEW_CPU_EXPERIMENT:
    trunk = dict(np.load(OUT/'trunk.npz',allow_pickle=False))
    trial = op.Config('student_trial',depth=3,width=24,optimizer='hybrid',
                      scaling='global',evaluations=2000,adam_evaluations=600)
    model = op.fit(trial,7,trunk['train_coefficients'])
    prediction = op.reconstruct(trunk,op.predict_coefficients(model,op.VALIDATION))
    rows = []
    for j,reynolds in enumerate(op.VALIDATION):
        data = np.load(ROOT/f'results/cavity_highre_ghia/n512/re{reynolds}.npz',allow_pickle=False)
        truth = np.r_[data['u'].ravel(),data['v'].ravel()]
        rows.append({'Re':reynolds,**op.velocity_error(prediction[j],truth)})
    display(pd.DataFrame(rows))
    print('Parameters:',len(model['theta']),'seconds:',model['training_seconds'])
else:
    print('No training requested; frozen results remain unchanged.')
""")
    doc=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'flowmllab':{'unit':'week04','lab':8,'scope':'full-field neural comparison'}})
    execute(doc,ROOT)
    path=ROOT/'notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb'
    nb.write(doc,path)
    print(path,len(cells),'cells',flush=True)

if __name__=='__main__':build()
