"""Maintain the accepted October student-review lessons as reproducible overlays.

This authoring command applies source changes only. It never executes notebooks
or turns submitted outputs into author-verified evidence. Optional input is a
private folder of submitted notebooks; public builds use the committed JSON.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'qa'))
from notebook_review_updates import review_notebook_file
from notebook_math import normalize_markdown

LESSONS = {
'week01_1': r'''## Reading the implementation contract
The derivative example is a worked reference; Part B asks you to implement a
boundary-flux integral. A boundary segment needs two distinct ordered points for
trapezoidal integration. The example's five-point derivative validator therefore
must not be reused as the minimum-size rule for the flux integral. Validate the
quantity you actually compute. Use `np.trapezoid` with NumPy 2.

An independent expected value comes from analysis, not from calling the same
routine twice. For a constant horizontal velocity in a closed rectangle, the
right and left fluxes cancel and the top and bottom fluxes vanish. Derive these
four numbers before writing the function. A test that computes its expected
answer with the function under test cannot expose that function's axis error.
Record the commit and reviewed line ranges together, since edits move lines.
If a tool does not expose its model identifier, record **not exposed**. Small
round-off differences in a manufactured residual are interpreted against the
stated tolerance, not against an exact printed decimal.''',
'week02_1': r'''## Published evidence and a closed assessment are different
`OPEN_BLIND=False` keeps the live final-case evaluation closed. Published blind
tables and figures also reveal those answers, so they are hidden by the same
switch below. Freeze preprocessing, model settings, calibration and the metric
definitions before opening it. Once opened, these public cases are regression
cases for future changes; a new blind claim needs new cases.

Coverage counts how often the truth falls inside an interval; sharpness describes
its width. NLL (negative log likelihood) scores the full predictive distribution,
and CRPS (continuous ranked probability score) scores its cumulative distribution.
A scale factor can match coverage at one chosen probability without correcting
the entire calibration curve. Thousands of neighboring grid nodes do not create
thousands of independent flow cases. Use the development-only leave-one-case-out
cell to assess dependence on the calibration case; do not choose that case by
looking at final-case coverage.''',
'week04_1': r'''## Read an error budget before a speedup
A POD mode is a spatial pattern; its coefficient changes with time. Capturing
99.9% of snapshot energy limits a projection error on those snapshots, not the
future trajectory error of a Galerkin model. Closure error, DEIM approximation,
time integration and parameter transfer can still dominate. A reduced solver
that reconstructs the full field every step can be slower than its full model.

Compare the ROM with the CFD solution on the same grid first, then separately
compare that CFD grid with a finer one and with Ghia centerlines. A small ROM
error cannot remove a coarse-label error. Treat retained timing ratios as
machine-specific observations, not a speedup students must reproduce. A failed
physical comparison remains a failed comparison even if an ML metric is small.''',
'week04_2': r'''## Separate surrogate error from changing the CFD labels
The low-fidelity Stokes field omits convection; the learned correction estimates
what must be added to approximate Navier--Stokes on the specified grid. The
surrogate-to-label error and the coarse-to-fine CFD difference answer different
questions. A 0.1% surrogate error against a coarse field can coexist with a 20%
change in that field under refinement.

Pressure is reconstructed from momentum gradients with an aligned gauge. The
reported gradient residual measures compatibility of that reconstruction with
the supplied gradients; it is not a direct pressure-error norm and need not rank
models in the same order. Read velocity, pressure and compatibility together.
Optional retraining writes a new student directory. The retained figures remain
historical evidence; they do not become figures of your new fit.''',
'week07': r'''## Why a grid study can fail when the last pair looks close
Richardson extrapolation assumes a consistent asymptotic error law. With fine,
medium and coarse observations $q_1,q_2,q_3$ and a uniform refinement ratio $r>1$,
the monotone estimate is

$$p=\frac{\log |(q_3-q_2)/(q_2-q_1)|}{\log r}.$$

Here $q$ is one physical observable, such as drag or recirculation length, and
$p$ is an observed accuracy order. A change of sign between the two successive
differences invalidates this monotone estimate; an undefined GCI is a diagnostic,
not a number to replace by zero. For example, lengths 1.422, 1.418 and 1.437 do
not approach a limit monotonically. Keep the gate failed, add a declared finer
grid or longer sampling window, and repeat the same metric definitions.

Check diameter resolution separately for every Reynolds number using
$Re=UD/\nu$: the retained Re180 configuration uses 15 nodes per diameter,
where other listed configurations use 12. Domain blockage (here 12.5%) can alter
drag and shedding frequency even after a grid study. Grid independence, domain
independence and external physical validation are separate tests.

For modal sensing, an oracle gets true modal coefficients; an autonomous forecast
must generate its own. Eight sensors for eight modes can be badly conditioned.
Report the sensor count, placement, condition number and error together; a plot
without the underlying table is insufficient evidence.''',
'week07_1': r'''## Identify the model you actually run
The default trained classroom model is a coordinate MLP, not a branch/trunk
DeepONet. An MLP maps a finite vector to outputs. A DeepONet combines a branch
encoding an input function sampled at fixed sensors with a trunk encoding query
coordinates. The optional topology demonstration is separate and may be skipped.

A classical interpolation baseline may win; reporting that is a correct result.
Locate errors by physical region as well as globally: shock layer, near-wall
region and wake. Quote the region mask and normalization. Similar training and
validation errors suggest limited fit capacity or budget, but do not identify a
unique cause. Neither a favorable error nor a topology diagram proves operator
generalization to a new family of input functions.''',
'week07_2': r'''## A sensor-space check is not a field-space uncertainty check
The reduced state contains POD coefficients. The observation matrix maps those
coefficients to sensor readings. The innovation is the measured reading minus
its predicted reading; the Kalman gain determines how much that innovation
changes the state. A near-unit normalized innovation RMS tests uncertainty at
observed sensors. It does not certify coverage of the full field, where missing
POD modes and model discrepancy remain. Thus innovation RMS near 1.08 and field
coverage near 55% need not contradict one another.

If rank 8, 32 sensors and covariance scale 1000 are the largest searched values
and also the selected values, the search has reached its boundary. Extend the
search on development/validation cases before freezing a new protocol. Do not
extend it in response to test-case scores.''',
'week07_3': r'''## Compare information and compute budgets together
The main scratch fit uses 300 downstream steps; the extended scratch comparison
uses 5000. Those are different experiments. At each label count, report training
steps, optimizer, unlabelled source data, target labels and evaluation frames
beside the error. The extended control can reverse a conclusion drawn from the
short-budget curve. A pretrained initialization is not automatically better at
equal compute.

The pooled POD baseline must see the same allowed unlabelled frames if the claim
concerns the value of pretraining. Identical errors at one and two labels can
mean validation kept the same initial checkpoint, rather than that another label
contains no information. CPU training can take tens of minutes; the fair-budget
control is a substantial part of the runtime. Use the printed fresh output path
to find this run; saved historical text is not a new execution.''',
'week07_4': r'''## Count source labels as well as target labels
The target budget $k$ counts labels from the new trajectory. It does not include
the 2761 source lift labels used for the retained ridge readout. Report both
counts, along with unlabelled pretraining frames. The phrase "32 labels" alone
would hide this source supervision.

An aggregate improvement does not imply improvement on every Reynolds number.
Read the per-trajectory table, including Re115 and extrapolation at Re135
(development maximum 130). One encoder initialization cannot measure variation
between independently pretrained encoders. A baseline winning is an acceptable
scientific outcome, not a reason to change the evaluation cases.''',
'week08': r'''## Check the branch, the domain and the denominator
For Rayleigh flow with heat-capacity ratio $\gamma>1$, the supersonic branch has

$$1-\gamma^{-2}<T_0/T_0^*<1.$$

At $\gamma=1.4$ the lower limit is about 0.4898 and corresponds to infinite
Mach number. A requested ratio of 0.30 has no supersonic solution. The solver
must explain this physical limit before invoking a root finder. The sonic endpoint
has $M=1$; no finite maximum Mach should silently redefine the physical domain.

Residual and output error have different units and sensitivities. If a relation
is $F(y,a)=0$, a small perturbation obeys $\delta F\approx F_y\delta y$ with
input $a$ held fixed. A large $F_y$ can produce a large equation residual from a
small output error. A positivity parameterization enforces positivity alone,
not the equation, conservation or accuracy.

Compare baseline error on the common supported queries and report coverage on
the full query set. Error on 65% of queries is not directly comparable with
error on 100%. A pass close to a declared threshold is a narrow pass, not a
robust margin. Keep the original criterion when reproducing the experiment.''',
'week09': r'''## Local accuracy, geometry and the evaluation contract
In the zonal loss

$$\mathcal L=\alpha\,\frac{1}{N_v}\sum_{i\in V}e_i^2
+(1-\alpha)\,\frac{1}{N_m}\sum_{i\in M}e_i^2,$$

$V$ denotes reverse-flow samples, $M$ the remaining samples, and $N_v,N_m$
their counts. The scalar $\alpha$ weights regional mean errors, rather than
the number of pixels in each region. Select it on validation cases, then freeze
it before opening final cases. Always show unweighted global AND reverse-flow
validation errors beside the weighted fits. A validation guardrail, such as a
two-percentage-point global penalty, is a selection rule, not a guarantee on new
geometries. Map the largest local error; it may sit at the step lip rather than
inside the reverse-flow zone.

The classroom coordinate model is an MLP. A strict branch/trunk model that uses
only $(x/H,y/H)$ in its trunk has a different geometry representation and training
problem. A worse result is not by itself proof of an architectural limitation.
Compare seeds, budget, training error and explicit wall-relative features before
attributing the difference to geometry. Do not mix numbers from different runs.

In nozzle registration, a station returned only at grid nodes cannot distinguish
sub-cell shock motion. Equal detected stations do not prove identical physical
shocks. Report grid spacing, ties and every non-monotone pair. Target-assisted
alignment is a representation audit; deployment must use a training-only locator.
Historical holdouts are regression cases once inspected, not new blind cases.

Branch sensors and query coordinates have different roles. Fixed branch sensors
describe the input consistently across cases. Each query must still be paired
with its own target value; equal array shapes alone do not establish that pairing.''',
'week10': r'''## Trace every figure to the run that produced it
Retained author figures document a fixed experiment. Newly regenerated figures
must be displayed from the fresh output directory, next to that run's metrics.
The default cavity reproduction interpolates fields in log Knudsen number; it
does not retrain the article's per-Knudsen MLPs.

For two training Knudsen numbers $Kn_a,Kn_b$, the interpolation weight is

$$w=\frac{\log Kn-\log Kn_a}{\log Kn_b-\log Kn_a},\qquad
q=(1-w)q_a+wq_b.$$

$q$ is a field and $w$ is dimensionless. This is interpolation for $0\le w\le1$;
outside that interval it is extrapolation. Verify the weight by substitution.
For error $\|q-\hat q\|_2/S$, state the scale $S$. A velocity-component error
normalized by the lid speed can look small while its component-relative error
is large. Neither normalization may be silently substituted for the other.
Also inspect heat flux, stress and translational-temperature overshoot: matching
density alone does not validate nonequilibrium shock structure.''',
'week10_1': r'''## Collision-map errors and the low-temperature sampling domain
This executable lesson learns a Lennard--Jones deflection map; it does not run a
learned collision kernel inside a DSMC flow simulation. Impact parameter $b$ and
relative energy $E$ determine deflection angle $\chi$. The surrogate predicts
$\cos\chi$, so inspect its largest local errors near orbiting and glory features,
not only a thermal average.

An optimizer stopping at its iteration budget is not a convergence certificate.
Record the warning, iteration count, seed and library versions. Reproducing a
retained decimal is not a reason to tune on final energies.

The transport audit samples a Maxwellian relative-energy distribution and drops
states outside the tabulated interval. Report the number and fraction dropped
at every temperature, separately below and above the table. At low temperature,
a larger omitted low-energy mass can make a truncated integral look artificially
accurate when reference and surrogate omit the same region. Compare the two on
the same stated interval, then assess the omitted contribution separately.
Diffusion and viscosity weight scattering differently; a single small integrated
error cannot bound both cross sections at every energy.''',
'week11': r'''## What the detector knows, and what its score means
A pixel MLP cannot compute spatial derivatives from the velocity of one pixel.
When vorticity, compression, $Q_d$ and swirling strength are inputs, the learned
decision already has engineered physical information. Remove rotational inputs
on a new declared split, retrain, select thresholds on validation and report each
test case separately. A loss of skill shows dependence on those inputs for this
model and protocol; it is not a theorem about all neural networks. Report an
iteration-limit warning and an optimum at the edge of a threshold grid.

For noisy velocities, recompute every derivative channel from the noisy fields.
Do not add noise to $u,v$ while leaving their clean derivatives in the inputs.
Use pure shear and expansion as negative controls: large gradients are not
automatically vortices or compression layers.

Reconstruction error and segmentation Dice measure different tasks. Derivatives
amplify small-scale velocity errors. A threshold sweep changes the weak reference
itself, so derive thresholds on training data and use validation for sensitivity;
do not choose the reference that maximizes test Dice.

For vapor clouds, compare the network with an explicit connectivity rule and
report wall false positives separately. Dice calculated after masking solid
pixels does not penalize predictions inside the foil. Agreement with an
algorithmic reference is not accuracy against independently labeled cavitation.
A frame mask is not temporal cloud tracking or a shedding-frequency measurement.

Freezing a decoder does not freeze its input: if its shared encoder changes,
its predictions can change. Freeze normalization statistics too, or explicitly
measure and control old-task drift before claiming preservation.''',
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path)
    args=parser.parse_args()
    patchfile=ROOT/'qa/notebook_review_patches.json'
    patches=json.loads(patchfile.read_text(encoding='utf-8'))
    def add(path, ident, text, kind='markdown', before=None, append=False):
        key=path.relative_to(ROOT).as_posix()
        item={'kind':'insert','id':'oct20-'+ident,'cell_type':kind,
              'replacement':normalize_markdown(text) if kind=='markdown' else text}
        if before:item['before']=before
        if append:item['append']=True
        items=patches.setdefault(key,[])
        previous=next((p for p in items if p.get('id')==item['id']),None)
        if previous:
            if previous!=item:raise ValueError('Overlay changed; explicitly migrate existing cells: '+ident)
        else:items.append(item)
    # The source JSON is the build input, so a notebook rebuild retains these lessons.
    for folder,text in LESSONS.items():
        paths=list((ROOT/'notebooks'/folder).glob('*.ipynb'))
        if folder=='week04_1':paths=[ROOT/'notebooks/week04/W4_1_Classical_ROM_Cavity.ipynb']
        if folder=='week04_2':paths=list((ROOT/'notebooks/week04').glob('W4_Lab4*.ipynb'))
        for path in paths:
            add(path,'reading-guide',text)
    if args.submission:
        selected={
          'W10_DSMC_Data_Driven_Surrogates_Student.ipynb':[24,26,29,32],
          'W10_1_Collision_Map_Surrogate_Audit.ipynb':[14,17,20],
          'W11_Shock_Vortex_Identification.ipynb':[15,17,20,23],
          'W11_Lab2_Reconstruction_and_Identification.ipynb':[11,14,17,19],
          'W11_Cavitation_Cloud_Detection.ipynb':[26,29],
          'W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb':[26,29,32,41],
          'W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb':[30,33,35,37],
          'W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb':[5],
          'W8_Lab1_Exact_Gas_Dynamics_Student.ipynb':[10],
        }
        descriptions={
          'W10_DSMC_Data_Driven_Surrogates_Student.ipynb':[
             'Verify log-Knudsen weights against their analytical formula.',
             'Read the error normalization before comparing velocity components.',
             'Measure shock overshoot and the midpoint convention.',
             'Compare interpolation with extrapolation for the same fields.'],
          'W10_1_Collision_Map_Surrogate_Audit.ipynb':[
             'Locate local scattering errors before thermal averaging.',
             'Compare diffusion and viscosity cross sections on the same energy table.',
             'Count omitted low-energy pairs before interpreting the thermal integral.'],
          'W11_Shock_Vortex_Identification.ipynb':[
             'Define matched evaluation and recompute derivative features from primitive fields.',
             'Keep models frozen while adding velocity noise and recomputing all derivatives.',
             'Retrain on a disjoint split with and without rotational inputs.',
             'Check pure shear and expansion without positive target labels.'],
          'W11_Lab2_Reconstruction_and_Identification.ipynb':[
             'Load frozen weights and check that the replay matches the retained run.',
             'Compare native and reconstructed divergence on the same interior region.',
             'Derive diagnostic thresholds from training only and inspect validation sensitivity.',
             'Evaluate the frozen threshold sweep on the historical regression case.'],
          'W11_Cavitation_Cloud_Detection.ipynb':[
             'Locate an attached/cloud disagreement in Case24 and inspect its vapor fraction.',
             'Compare connectivity and neural masks; include predictions inside the solid wall.'],
          'W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb':[
             'Show both errors of the unweighted validation baseline next to the zonal sweep.',
             'Compare each final geometry, not just the mean.',
             'Locate the maximum local error in physical coordinates.',
             'Separate source identity from DSMC numerical verification.'],
          'W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb':[
             'Check the POD spectrum against the retained representation audit.',
             'Separate target-assisted profile errors from training-only station prediction.',
             'Summarize the newly generated six-field errors.',
             'Print the station resolution, tied stations and non-monotone pressure pairs.'],
          'W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb':[
             'Demonstrate why equal shapes do not prove query-target pairing.'],
          'W8_Lab1_Exact_Gas_Dynamics_Student.ipynb':[
             'Substitute changed targets back into the declared branch equations.'],
        }
        provenance=[]
        for name, indices in selected.items():
            submitted=next(args.submission.glob('week*/'+name))
            nb=json.loads(submitted.read_text(encoding='utf-8'))
            target=next((ROOT/'notebooks').rglob(name))
            add(target,'worked-audit-heading','## Worked audit and interpretation\n\nRun these cells after the main lesson. They compute diagnostics from the objects above; they do not replace the frozen scientific protocol. Predict the result before running, then explain any failure using the stated metric and input information.',append=True)
            for k,i in enumerate(indices):
                source=''.join(nb['cells'][i]['source'])
                assert nb['cells'][i]['cell_type']=='code'
                # Import copied cells with no submitted outputs or numerical answer prose.
                if name.startswith('W11_Cavitation') and i==26:
                    source=source.replace("largest = 1 + int(np.argmax(ndi.sum(swap, components, range(1, count + 1))))", "if count == 0:\n    raise ValueError('No attached/cloud swap exists in this frame; choose a frame using a predeclared rule.')\nlargest = 1 + int(np.argmax(ndi.sum(swap, components, range(1, count + 1))))")
                if name.startswith('W11_Shock') and i==20:
                    source=source.replace("warnings.simplefilter('ignore', ConvergenceWarning)   # reported explicitly below instead", "warnings.simplefilter('always', ConvergenceWarning)")
                if name.startswith('W11_Lab2') and i==19:
                    source=source.replace('assert (test_gap > 0).all()', "print('Reconstruction exceeds interpolation at each threshold:', (test_gap > 0).tolist())")
                if name.startswith('W11_Lab2') and i==11:
                    source='import numpy as np\nimport torch\n'+source
                    source=source.replace("weights_only=True)", "weights_only=True, map_location='cpu')")
                if name.startswith('W9_Lab1') and i==29:
                    # Retained comparisons remain visibly retained, not fresh measurements.
                    source="print('Fresh model errors; any merged contour metrics below are retained evidence.')\n"+source
                    source=source.replace('assert np.isclose(100 * row["full_relative_l2"], retained.loc[f"H{height}", "combined_relative_l2_percent"], atol=1e-6)', 'print(f"H{height}: fresh minus retained global error (points):", 100 * row["full_relative_l2"] - retained.loc[f"H{height}", "combined_relative_l2_percent"])')
                if name.startswith('W9_Lab1') and i==41:
                    source=source.replace('f"9 files, SHA-256 verified at commit {STEP_SOURCE_COMMIT[:7]}"', '"See the retained source manifest; identity is separate from convergence"')
                add(target,f'audit-{i}-text',f'### Audit {k+1}: '+descriptions[name][k]+'\n\nState which inputs and reference define this comparison. A failed check is a result to investigate, not permission to select new settings on the test cases.',append=True)
                add(target,f'audit-{i}',source,'code',append=True)
                provenance.append({'notebook':target.relative_to(ROOT).as_posix(),'submitted_cell_index':i,'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
        (ROOT/'qa/student_feedback_cell_sources.json').write_text(json.dumps({'source':'Aryan Mohta submission, 20 October 2026; accepted code adapted and reviewed','cells':provenance,'excluded':'Network re-downloads, inconsistent DeepONet result prose and unverified causal conclusions. Submitted outputs are never imported.'},indent=2)+'\n',encoding='utf-8')
    patchfile.write_text(json.dumps(patches,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for key in patches:
        if any(p.get('id','').startswith('oct20-') for p in patches[key]):
            review_notebook_file(ROOT/key)
    print('Student-feedback notebook overlays applied.')

if __name__=='__main__':main()
