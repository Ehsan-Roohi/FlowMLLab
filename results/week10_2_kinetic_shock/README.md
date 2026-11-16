# Week 10.2 - Kinetic-shock teaching evidence

[Week guide](../../weeks/week10_2/README.md) · [Executed notebook](../../notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb) · [Lecture](../../lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf)

This unit uses a pinned extract of Ehsan Roohi's
[PINNs-DVM study](https://arxiv.org/abs/2605.29211v2), with source hashes and case
bindings in the [data manifest](../../data/week10_2_kinetic_shock/manifest.json).
The teaching code includes NumPy moment/conservation diagnostics, a fresh CPU
characteristic BGK PINN and a separate finite-volume DVM. Retained paper profiles
remain clearly identified research outputs. They are not relabelled as fresh
teaching runs or independent held-out tests.

## Fresh CPU training

Section 10 trains a two-hidden-layer, 24-neuron-per-layer tanh model from random
weights using 100 Adam updates and 20 L-BFGS outer steps. The student can change
depth, width, optimizer, sampling, quadrature and density/momentum/energy loss
weights. The default has a 120-second ceiling and records its actual stop reason.

[pinn_training_protocol.json](pinn_training_protocol.json) declares all seven
configurations before fitting and DVM scoring. [pinn_cpu_report.json](pinn_cpu_report.json)
contains initial/final physics residuals, independent physics-audit residuals,
all optimizer/diagnostic work counts, timing and source hashes.
[pinn_cpu_states.npz](pinn_cpu_states.npz) preserves initial and final parameters;
it is a prediction record, not a complete optimizer restart checkpoint.

The independent NumPy DVM solves reduced g/h/k equations with minmod/upwind
finite volumes and SSPRK2. It uses matched physical initial/incoming states,
analytic transverse integration, 400/800 cells and axial quadrature orders 48/64.
Both-stage positivity and boundary/collision balance diagnostics are retained.
Two grids at fixed CFL check joint space/time sensitivity; no convergence order
or GCI is inferred. This transverse discretization differs from the paper's
original finite transverse quadrature, whose comparison is labelled secondary.

[pinn_cpu_profiles.npz](pinn_cpu_profiles.npz) stores all fresh physical fields,
initial predictions and references; [pinn_cpu_metrics.csv](pinn_cpu_metrics.csv)
lists all nine field errors. DVM values do not enter training or model selection.
These declared cases are demonstrations, not a search for the best DVM score.
Optimizer update budgets differ, so compare measured gradient/function counts
and setup/update times rather than outer steps alone. Host load affects timings.

Since tf/tau=0.02, the collision fraction is below 2%. Hard initial/inflow
transport makes the untrained reconstruction informative. Interpret changes
relative to that initial state, including worsening fields. This short CPU lab
does not establish stationary, long-time or Shakhov training performance.

In the retained default run, relative L2 heat-flux error changes from 0.3011% to
0.2364% and stress from 0.9829% to 0.5510%. Velocity changes from 1.0029% to
1.3684%, and temperature from 0.01297% to 0.01342%, so improvement is not uniform.
The fixed-CFL 400-to-800 DVM changes are 0.1213% for heat flux and 0.1374% for
stress. These sensitivities are not uncertainty bounds and do not by themselves
establish that small training gains are numerically resolved. The three-layer
run has lower heat-flux/stress errors in this seed; the predeclared two-layer
default is retained, and no universal architecture or optimizer ranking is claimed.

## Physical figures

**Tube conditions (all transient figures below):** x ∈ [−0.5, 0.5], t = 0.02,
constant τ = Kn = 1 in the source scaling, monatomic R_g = 1, γ = 5/3.
Initial left/right (ρ, u, T) = (1, 0, 1)/(0.125, 0, 0.8), with
H = [1 + tanh(20x)]/2 smoothing and incoming Maxwellian reservoirs.
Both sides initially rest: **no stationary upstream Mach number is imposed**.
BGK has Pr = 1; the Shakhov tube uses Pr = 2/3. See the
[physical definition](../../data/week10_2_kinetic_shock/README.md#physical-conditions-and-mach-number-interpretation).

| Figure | What it shows |
| --- | --- |
| [Kinetic model schematic](kinetic_model_schematic.png) | One physical coordinate, three molecular velocity components and two collision targets |
| [Moment hierarchy](moment_hierarchy.png) | Why the transverse fourth-moment marginal is needed for separate R and Delta |
| [Homogeneous relaxation](collision_rates.png) | Analytic stress and heat-flux time histories; fresh collision-rate quadrature is checked in Table D |
| [BGK tube](tube_bgk_profiles.png) | Pr = 1 tube above; parent/final equation-trained profiles against the matched 3200-cell BGK DVM |
| [Shakhov tube](tube_shakhov_profiles.png) | Pr = 2/3 tube above; parent/final profiles against the matched 3200-cell Shakhov DVM |
| [Mach-2 optimizer path](m2_optimizer_profiles.png) | Stationary M₁ = 2, Shakhov Pr = 2/3, τ = 1/ρ, upstream ρ₁ = T₁ = 1; seed 141001, 81 common core nodes in [−10, 10]; Table F retains all three seeds |
| [Moment-supervision comparison](supervised_bgk_profiles.png) | Pr = 1 tube above; equation-trained, five-field S0 and final eight-moment states against the same 6400-cell BGK reference, 399 common nodes |
| [Fresh PINN and DVM](pinn_from_scratch_profiles.png) | Pr = 1 tube above; random-start and trained CPU BGK predictions against the independent 800-cell DVM; 400/800-cell sensitivity is separate |
| [Fresh optimizer comparison](pinn_optimizer_profiles.png) | Signed heat flux and stress for the declared Adam, L-BFGS and hybrid runs |
| [Training components](pinn_training_history.png) | Unweighted scaled density/momentum/energy residuals versus actual gradient work and elapsed time |

Every Mach-2 plot concerns the same stationary Shakhov case. The transient tubes
are separately labelled and do not use a stationary constant-flux acceptance
test. Scientific legends are outside the plotted curves; signs and original
comparison coverage are retained.

## Metrics and interpretation

[profile_metrics.csv](profile_metrics.csv) records relative L2 percentages using
trapezoidal spatial weights on each case's declared nodes. Reference norms are
global over the comparison interval; no pointwise division at a zero crossing
is used. The notebook also plots unnormalized signed differences.

The supervised comparison uses the strict common overlap with the 6400-cell
reference, without endpoint extrapolation, alignment fitting or smoothing.
The source paper's corresponding 401-node table uses an unweighted norm. These
are explicitly different scoring conventions; the teaching CSV is not presented
as an exact reproduction of every final decimal in that table.

The five-field S0 was trained against a 400-cell reference. The final eight-moment
state uses 6400-cell targets and reference-derived guards. Both can be assessed
against the same 6400-cell reference here, but they are not an experiment in
which only the number of supervised moments changed. Intermediate eight-moment
states remain available in the extract and are labelled separately.

For Mach 2, split-half SSB improves the mean heat-flux error while slightly
worsening mean stress error relative to the preceding SSB stage. The stages
change objective/quadrature and cumulative work. The final runs retain failed
tail/projection checks; conservation alone does not certify the whole kinetic
solution. The final supervised tube likewise retains failed physical checks and
small-amplitude wrong-sign tails.

## Reproduction

Run the notebook on CPU to perform fresh training/reference calculations and
rescore saved profiles. [execution.json](execution.json) records its measured build-environment
runtime and code-cell count. Installation and hosted-Colab execution times are
not inferred from that local measurement.

[validation.json](validation.json) records the final notebook/PDF hashes,
embedded-figure check, focused tests, page-by-page review and preservation
of all 62 previously numbered notebooks. The repository release gate passes;
its unrelated Week 9 upstream-source check still requires an external cache.

To rebuild these figures and the new lesson only, from the repository root:

```sh
python qa/build_week10_2_materials.py --figures --notebook --execute --pdf
```

To regenerate the declared seven-run training evidence before rebuilding:

```sh
python qa/run_week10_2_pinn.py --run --figures
```

The PDF build needs licensed Times New Roman locally; font files are not bundled.
Source provenance and the distinction between the paper's CC BY 4.0 license and
the upstream repository's unspecified software license are retained in the
[extract notice](../../data/week10_2_kinetic_shock/README.md).
