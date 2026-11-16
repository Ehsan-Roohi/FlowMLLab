# Constant-lid Ghia PINN campaign protocol

Status: **implementation and bounded pilot protocol**, not seven validated PINN
solutions. A successful program exit, small training loss or visually plausible
vortex is not a validation result. This additive experiment leaves the existing
notebooks, CFD data and regularized-lid deep-cavity source unchanged.

## Question and information budget

Solve the steady, two-dimensional incompressible Navier–Stokes equations afresh
for each of Re = 100, 400, 1000, 3200, 5000, 7500 and 10000. The network receives
coordinates only; each run has its own fixed Re. This is not a parametric operator
predicting unseen Re. There are **no interior CFD observations, Ghia samples,
reference-derived initialization or reference-based model selection** in the
training engine. The current POD-DeepONet is a different, supervised experiment.

Use nondimensional x,y in [0,1], lid speed U=1, side length L=1, density rho=1,
kinematic viscosity nu=1/Re and pressure scaled by rho U². On the open lid,
u=1 and v=0. On the other three open walls, u=v=0. The two moving-lid corners
have incompatible boundary traces and are excluded as exact evaluation points;
we never replace the lid by a Gaussian or polynomial speed profile. Exported
corner entries are NaN with an explicit mask, not interpolated values.

## Exact boundary representation and the corner limitation

The network outputs two corrections N_psi(x,y), N_p(x,y). Velocity is obtained
from a streamfunction, u=psi_y and v=-psi_x, so continuity is an analytic identity.
Vorticity is omega=v_x-u_y=-Laplacian(psi); signed values are retained.

For the upper-left corner define X=x, Z=1-y, theta=atan2(Z,X), r²=X²+Z² and

- A = -pi²/(pi²-4), B = 2pi/(pi²-4), D = 4/(pi²-4);
- psi_L = (A+B theta) Z + D theta X;
- p_L = -2 nu (B Z+D X)/r².

For the upper-right corner set X=1-x with the same streamfunction expression
and reverse the pressure sign. The isolated corner pair satisfies the Stokes
momentum equations exactly; this identity is tested independently at radii
0.1, 0.01 and 0.001. The general right-angle singular-corner construction follows
[Burda, Novotny and Sistek, ICCFD7 (2012), Section 4](https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-3305_paper.pdf).
The coefficients and signs above were independently derived for our coordinate
and streamfunction convention, not copied from an incompatible convention.

Let w_R=x²/[x²+(1-x)²], w_L=1-w_R and b=16x(1-x)y(1-y). The global representation is

```text
psi = y² (w_L psi_L+w_R psi_R) + b² N_psi
p   = y² (w_L p_L+w_R p_R) + N_p(x,y)-N_p(0.5,0.5).
```

The pressure lifting is antisymmetric and zero at the centre, so p(0.5,0.5)=0.
The quadratic blend and y² factor impose the open-wall velocities exactly.
The network does not need a boundary penalty or a tunable continuity weight.

**This leading singular subtraction does not make the complete Navier–Stokes
residual square-integrable automatically.** Blending, the bottom-wall factor
and nonlinear advection leave subleading corner terms. A previously considered
generic rational lifting was rejected before training because its leading
angular mode was not biharmonic. That rejected implementation is not retained
as a scientific baseline.

Training uses uniform area samples in the square with the two top-corner
quarter-disks of radius epsilon removed. The declared baseline is epsilon=0.02;
repeat with epsilon=0.01 using the same initialization seed. This is a
**punctured-collocation approximation**. No artificial boundary conditions are
applied on the quarter-circles, and no residual mask is applied elsewhere.
Agreement between two collars is sensitivity evidence, not proof of full corner
consistency. Fixed outer audit points are common across both collars; raw shell
audits extend inside the excluded region.

## Objective, optimizer and finite budget

At each fixed collocation point compute, by float64 automatic differentiation,

```text
r_u = u u_x+v u_y+p_x-(u_xx+u_yy)/Re
r_v = u v_x+v v_y+p_y-(v_xx+v_yy)/Re
J = mean(r_u²+r_v²).
```

There is no clipping, residual-dependent weight, pointwise normalization, Ghia
loss or reference residual. Chunked evaluation accumulates the same full-batch
objective and gradient. Losses from different Re or collars are not directly
accuracy rankings.

The initial finite baseline is four tanh hidden layers of 64 units, two output
heads in one MLP, Xavier-normal weights, zero biases, float64, fixed 2048 IID
collocation points, Adam at 1e-3 for 2000 updates, then L-BFGS for 500 outer
updates. PyTorch L-BFGS uses strong-Wolfe line search, max_iter=1 per call,
history_size=50, tolerance_grad=1e-12 and tolerance_change=1e-14. All collocation
points are frozen from initialization; line-search closures see exactly the
same objective. Actual closures and residual-point evaluations are recorded,
because one L-BFGS outer update can evaluate many more points than one Adam
update. The last logged closure value is identified as such and is not falsely
called a newly recomputed accepted-endpoint loss.

Pilot order: Re100 and Re7500, seed7, both epsilon values. Run a first segment
of at most 50 Adam updates under the predeclared 2000/500 total budget, measure
GPU time and memory, then resume the same trajectory if the finite resource
budget is suitable. A separate ten-step L-BFGS cost probe may be disposable and
clearly labelled; its state must never be resumed back into Adam or pooled with
the honest trajectory. The full seven-case and additional-seed matrix is a
subsequent scheduling decision based on cost and implementation admission,
not favourable CFD test errors. No 42-job campaign is implicitly launched.

Residual-adaptive sampling, deeper networks and alternative optimizers remain
controlled future variants. They are not implemented or claimed as improvements
in this fixed baseline. If implemented, a finite candidate-pool proposal q_i
sampled with replacement needs unnormalized weights 1/(N q_i), detached
probabilities, accounted candidate-evaluation cost and an objective frozen
before L-BFGS. Superiority requires comparable budgets and independent seeds.

## Admission before GPU training

`--admit-only` must pass on the target device and precision. CPU tests include:

1. Exact open-wall values, including points 1e-6 from lid corners; maximum
   velocity error below 1e-10. The singular endpoints themselves are rejected.
2. Solid rotation psi=-(x²+y²)/2, p=(x²+y²)/2: zero steady momentum residual,
   zero divergence and omega=2 for arbitrary Re.
3. A separate polynomial psi=x³y², p=x²y with nonzero viscous terms, checked
   against hand-derived forcing; absolute residual discrepancy below 1e-10.
4. Both isolated corner solutions: relative Stokes cancellation below 1e-8.
5. Full-network directional loss gradient versus centred finite differences:
   relative discrepancy below 1e-5; float64 enforced.
6. Full-batch versus chunked objective/gradient equivalence.
7. Exact same-runtime restart equivalence during Adam, at the Adam/L-BFGS
   boundary and during L-BFGS, including optimizer history and sample state.

These are implementation gates. They do not establish solution accuracy.

## Fixed audits and honest acceptance

Audit seed983451 generates independent fixed uniform points outside radius0.02,
shared across every Re, training seed and collar. These points are never used
by the optimizer or to select an endpoint. Raw unweighted momentum and
continuity RMS, maximum residual and minimum sampled corner radius are reported.
Separate area-sampled quarter-annuli use radial edges 1e-4, 1e-3, 0.002, 0.005,
0.01, 0.02 and 0.05. Report all shell results, including inside the training
collar. Pressure and vorticity can remain singular; finite sampled values do
not imply finite full-domain norms.

The complete finite training schedule is finished and model/prediction hashes
are frozen **before** reading the retained CFD fields or Ghia tables. Post-freeze
scoring must include signed centreline profiles, whole-field vector errors,
near-wall profiles, pressure with a common gauge, vorticity, vortex structure,
and paired epsilon sensitivity on a common physical grid. A postprocessing
bridge records the exact retained reference path, grid, SHA256, prediction hash
and score time. Ghia's known questionable entries remain visible; use the
pre-existing screened/unfiltered policy in `flowmllab/ghia.py` without inventing
replacement data. Current CFD discretization uncertainty must accompany any
PINN-versus-CFD number, especially at high Re; CFD is not declared exact truth.

Preregistered targets for calling a case an **accepted pilot approximation**:

- all implementation gates pass; no nonfinite values away from exact corners;
- independent outer momentum RMS <=1e-2 and divergence RMS <=1e-10;
- screened Ghia u/v centreline relative L2 errors each <=2%, with unfiltered
  scores also shown;
- interior velocity-vector relative L2 versus the retained finest CFD <=5%,
  with the known CFD mesh sensitivity shown alongside;
- epsilon0.02 versus0.01 interior velocity-vector difference <=2%;
- no concealed near-wall, small-vortex or shell failure. If region-specific
  criteria are absent, show those distributions and label their assessment
  qualitative rather than declaring them passed.

These targets are fixed before seeing the PINN predictions. They are useful
acceptance thresholds, not universal mathematical convergence criteria. Any
failure stays in the report. No threshold is loosened to promote a pilot.
No claim of full-domain corner consistency or dynamic stability follows even
if the above finite comparisons pass. Additional seeds and refined collocation
are needed before a robust method-comparison or publication claim.

## Restart, provenance and storage

`checkpoint.pt` is written to a temporary sibling, flushed/fsynced and replaced
atomically only after a complete optimizer step. It includes all model tensors,
Adam/L-BFGS state/history, phase and step/closure counters, exact collocation and
audit points, Python/NumPy/PyTorch CPU/CUDA RNG states, dedicated sampler RNG,
configuration and canonical-LF source hashes (raw source hash also retained), objective hash, complete training history,
runtime history, and the frozen-endpoint record. Adaptive state is explicitly
`enabled:false`. There is no hidden best-state selection. A cross-device/runtime
resume is recorded; bitwise equivalence is claimed only within the admitted
same runtime. A changed source/config is rejected instead of silently migrating.

The prediction grid is part of the frozen configuration. The initial freeze timestamp is stored in the checkpoint. A completed export is byte-verified and reused without changing its timestamp or files; a missing manifest permits recovery of an interrupted export from the same frozen identity. A corrupt completed bundle is preserved and rejected.

The runner holds an OS file lock for the output directory. It will not start a
new trajectory over an existing checkpoint. SIGTERM/SIGINT/SIGUSR1 request a
stop at the next completed update; they never save midway through a line search.
It writes an initial recovery before training. A hard-killed process retains
the previous complete atomic checkpoint. The caller is responsible for checking
Slurm state before any resubmission and for durable verified archival copies.

Unity live I/O must use the actual allocated Workspace returned by `ws_list`.
The parent coordinates job submission, both eligible GPU partitions, expiry,
capacity and complete recovery. This module submits no jobs and spends no funds.
Keep one current recovery and one current verified result, not a checkpoint per
update. Do not delete unique state or reference scientific data.

## Commands and outputs

```bash
python qa/run_cavity_pinn_campaign.py --admit-only --device cpu
python qa/run_cavity_pinn_campaign.py --admit-only --device cuda
python qa/run_cavity_pinn_campaign.py --device cuda --output /allocated/workspace/case \
  --reynolds 100 --seed 7 --collar 0.02 --width 64 --depth 4 --collocation 2048 \
  --adam-steps 2000 --lbfgs-steps 500 --max-steps 50 --max-wall-seconds 1800
# Resume the same configuration: add --resume; remove --max-steps or set a segment bound.
```

`config.json`, `admission.json`, `checkpoint.pt`, `history.json` and
`current_receipt.json` describe the active trajectory. Only at its final fixed
endpoint are `frozen_predictions.npz`, `frozen_manifest.json` and
`final_physics_audit.json` produced. Prediction arrays x,y are one-dimensional;
u,v,p,psi,omega are `(y,x)`. The default grid is129x129; the two exact upper
corners are NaN and explicitly masked. Dense signed centreline arrays are also
included. No reference file is read during this export. The manifest records
prediction SHA256, frozen model/config/objective hashes and the freeze timestamp.

Run the focused tests with `python -m pytest tests/test_cavity_pinn_campaign.py`.
The generated test directory must be on the authorized external D: drive or
allocated HPC Workspace rather than accumulating on C:.

## Executed finite baseline (2026-11-16 UTC)

The initial cost pilots stopped after 50 Adam updates for Re100 and Re7500,
seed7, epsilon=0.02, chunk256. They are not completed accuracy experiments and
were not resumed into the main matrix. A separate identical-state chunk-cost
probe selected chunk2048 using measured runtime and objective/gradient
agreement only, before any reference comparison. The measured environment was
PyTorch2.5.1+cu118 and float64; the cost probe used one Quadro RTX8000 GPU.

Array1808042 contains 42 fresh trajectories: seven Reynolds numbers, seeds7/23/61,
and epsilon0.02/0.01. Every case uses the original 2000-Adam/500-L-BFGS schedule,
2048 collocation points, and chunk2048. All42 finished the finite schedule;
completion does not establish accuracy. Case runtime is descriptive because
the scheduler assigned different GPU models. No reference labels or scored
accuracy were used to select the chunk size or endpoint. The immutable source,
plan, complete optimizer/model/RNG checkpoints and frozen endpoints are retained
in the verified campaign recovery archive. The published engine canonical SHA256
is 507a6e0a6d632a173c8bdc8748564beffee77ce3664aeffcee25edc5ca305b9e.

Post-freeze evaluation uses the retained 512-cell CFD interior nodes directly
(511 by511), with boundary conditions audited separately. Ghia values are
queried at their original table coordinates. There is no retained CFD pressure
field in this square-cavity dataset; pressure accuracy is therefore not claimed.
Both raw and screened Ghia errors, signed fields, independent physics audits,
paired-collar differences and failures must remain visible in the final report.
