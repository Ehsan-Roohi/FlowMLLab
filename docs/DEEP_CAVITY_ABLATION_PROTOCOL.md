# Controlled D/W = 5 optimizer experiment

This is a **fresh optimizer-pipeline experiment**, not a rerun of the retained
paper workflows. Both arms start from the same Xavier-normal network for each
paired seed and use the same frozen collocation points. No CFD labels, historical
weights, or test fields enter training, stopping, or model selection.

## Preserved physical problem and source

The problem is the Gaussian-lid rectangular cavity in
[Roohi and McDevitt, arXiv:2610.38814v2](https://arxiv.org/html/2610.38814v2),
Sections 2–3. W = 1, D = 5, Re = 1000; physical coordinates are
x in [0,1], y in [-4,1]. The output coordinate h = y + 4 is height above the
floor. Reynolds number uses the width and maximum lid-speed scales.

The adapter imports the authenticated author `cavity.py` unchanged:

- U_lid(x) = [1-exp(-x²/0.001)][1-exp(-(x-1)²/0.001)].
- eta = (y+4)/5; B = 16 x(1-x) eta(1-eta).
- psi_lid = (y-1) eta² U_lid(x) exp[-(1-y)²/0.01].
- d = 1-y; E = exp{-4.212 [logaddexp(0,10(d-0.25))/10
  - logaddexp(0,-2.5)/10]}.
- psi = psi_lid + B² E N_psi; p = E N_p;
  u = partial_y psi, v = -partial_x psi.
- The two physical momentum residuals use viscosity 1/Re. The training loss is
  mean[(M R_x/E)² + (M R_y/E)²], with the original two upper-corner Gaussian
  mask radius 0.01. Continuity and wall velocities are hard constraints.

The network receives physical x,y and has six tanh layers of width 64,
21,122 parameters, Xavier-normal weights, zero biases and float64 arithmetic.
The entire transformed field is differentiated, including the depth envelope.
This is different from the earlier simplified rectangular teaching driver.

### Source acquisition and attribution

The exact source files were recovered read-only from the author's retained
Unity workspace `cavity-re1000-d5-20260930`, directory
`pinn-balanced-20261001/code`. Local and remote SHA256 values were checked:

| Source | SHA256 |
|---|---|
| cavity.py | 29ff1f7ba495de25c088b4eefcd200202f1a2da020248d4b3e2a43a7e07ba1dd |
| optimizer.py | 1a69e52dd45f463ad9187434f22f12d98c24e969c9b908c9c094d29d87548d4d |
| pinn_d5_balanced_experiment.py | 7b7c0b6836f1834644071a54518046a1b437514fd1bc2cd5ba0ee5a570f31010 |
| train.py (provenance only) | 62f21872b6358518ded273c37ebb255d72f35881cf1f3689b3a8142cdb8ebc7b |
| pinn_d5_balanced_run.py (provenance only) | f4ef1e305651877951e2c8f71247da7af603e5bdaf2c976dea02a4c2501fc74e |

The paper attributes the inherited lifting, envelope and optimizer implementation
to Christopher J. McDevitt and identifies the regional-distribution extension
separately. The inspected source headers explain the implementation but do not
contain an explicit redistribution license. The user authorized publication of the paper code and data; the existing
Week 13 lesson records permission for the McDevitt source. The byte-identical
author-supplied bundle is available at `data/week13_d5/author_source`, with
McDevitt/project attribution and an explicit scope notice. The repository MIT
license does not silently relicense inherited source. A hash mismatch is a hard
error, not permission to load a nearby implementation.

The original SOAP package version was not recovered. The new experiment pins
`pytorch-optimizer==3.3.0` and records its actual runtime version. It is not
claimed to be the unidentified historical SOAP binary. Dependencies are PyTorch,
NumPy and that pinned optimizer package; pytest is needed for admission only.
The production environment records its own exact versions.

## Controlled comparison

The two arms are `adam_lbfgs` and `soap_ssb`:

| Phase | Adam → L-BFGS | SOAP → self-scaled Broyden |
|---|---|---|
| First order | Adam, lr 0.002, betas (0.99,0.999), no weight decay | SOAP 3.3.0, same lr/betas/weight decay, precondition every step |
| Quasi-Newton | PyTorch L-BFGS, strong Wolfe, max_iter=1, history 100 | Authenticated SSBroydenLM, strong Wolfe c1=1e-4/c2=0.9, lr=1, amax=10 |
| Samples | Identical frozen balanced mixture in both phases | Identical frozen balanced mixture in both phases |

This compares two **staged pipelines**, so it cannot isolate which of the two
optimizer stages caused a difference. A factorial four-pipeline experiment
would be needed to attribute first-order and quasi-Newton contributions.
No hyperparameter tuning uses CFD discrepancies or the final audit.

The authored balanced sampler allocates 50% to the full box, 30% to the upper
one-width strip, 15% to the floor one-width strip and 5% to logarithmic lower
corner samples (r in [1e-4,0.25]). Only upper-corner disks of radius 0.02 are
excluded. The baseline `cavity.sample_points` excludes all four corners, but is
**not** the training sampler here. Equal point weights intentionally define a
regional mixture objective, not an unbiased uniform-area integral.

The paper redraws SOAP batches every 50 steps and transfers earlier trained
models; this controlled experiment freezes the balanced sample set for both
phases and starts fresh. Its smaller sample/budget settings must be reported.
Using unchanged equations does not make this an exact historical workflow replay.

### Current admitted cost gate

The first cost gate has one seed (12345), 8192 points, 50 first-order full
loss/gradient evaluations and **10 quasi-Newton closure evaluations**, not ten
quasi-Newton iterations. Run both arms sequentially in the same GPU allocation
with identical settings. This short run tests cost, finite behavior and recovery;
it does not establish flow accuracy or an optimizer winner.

```
python qa/run_deep_cavity_ablation.py --source-dir data/week13_d5/author_source \
  --output OUTPUT/adam_lbfgs_s12345 --method adam_lbfgs --seed 12345 \
  --points 8192 --chunk-size 8192 --warm-evaluations 50 --qn-evaluations 10 \
  --device cuda --purpose cost_smoke
python qa/run_deep_cavity_ablation.py --source-dir data/week13_d5/author_source \
  --output OUTPUT/soap_ssb_s12345 --method soap_ssb --seed 12345 \
  --points 8192 --chunk-size 8192 --warm-evaluations 50 --qn-evaluations 10 \
  --device cuda --purpose cost_smoke
```

The parent task submits jobs only after CPU/GPU admission and checks the actual
allocated scratch workspace, expiry, capacity and absence of duplicate writers.
No submission is performed by this driver.

After the independent admission and cost gate, the parent task declared the
scientific pilot: three paired seeds (12345,23456,34567), 8192 points, 2000
first-order and 8000 quasi-Newton evaluations per arm. Each seed runs both arms
sequentially in the same GPU job, with counterbalanced arm order across seeds.
This schedule is fixed before fresh initialization and was not selected from CFD
accuracy scores. Do not extend the cost-gate checkpoint into it.
To pause a scientific run while retaining its original schedule use
`--max-steps N` or `--max-seconds S`; restart with the exact contract and `--resume`.

### Observed admission and cost gate (15 November 2026)

Unity job 1810186 completed successfully. All 15 admission tests passed on
PyTorch 2.5.1+cu118 and SOAP package 3.3.0. Both cost arms used the same Quadro
RTX 8000, initial model and point hashes, with 60 full loss/gradient evaluations
per arm. Recorded optimizer-step time was 11.689 s for Adam → L-BFGS and
15.024 s for SOAP → self-scaled Broyden. The ten-evaluation quasi-Newton allowance
for SOAP → SSB ended before its first accepted SSB step; rollback preserved the
SOAP endpoint and counted the spent trials. This is a budget-admission outcome,
not evidence against a properly budgeted SSB refinement. Cost does not establish
physical accuracy. In particular, the small initial SSB store does not predict
later generator-history memory or matrix-product cost exactly.

## Accounting and recovery

Every full residual-loss/parameter-gradient calculation counts once, including
line-search trials. Chunk losses are sums divided by the same global N, so chunk
size does not change regional weighting. An evaluation-budget interruption inside
a line search restores the last accepted model and optimizer state; spent trial
evaluations still count. Therefore equal-budget endpoints can have different
numbers of accepted quasi-Newton steps.

The source SSB representation stores two vector generators and coefficient blocks
per update, plus global scales. Capacity is at least the entire quasi-Newton
budget; dropping any generator is fatal. This is the author's full accumulated
representation, not a short-history replacement. Ordinary PyTorch optimizer
state alone would omit H: the adapter explicitly persists H, scalar scales,
update/drop/renormalization counters and line-search count.

The atomic checkpoint contains model, optimizer (including SOAP eigenbases,
preconditioner and moments), all Python/NumPy/Torch/CUDA RNG state, initial model,
frozen training and audit samples, source/configuration hashes, phase counters,
history, runtime entries and the absence of adaptive-sampler state. SIGUSR1,
SIGTERM and SIGINT request saving at an accepted step boundary. A lock prevents
two Linux writers from advancing the same directory. A hard process kill can
replay work after the most recent checkpoint; retained trajectory evaluation
counts are not a reconstruction of unrecorded scheduler billing.

Final parameters and freeze UTC are checkpointed before assessment exports.
Resuming a complete bundle verifies its bytes and returns without overwriting it.
Interrupted exports are recovered from the immutable completed checkpoint and
retain the original freeze UTC. Source, objective or total-budget changes reject
resume. One current complete checkpoint is retained; original author data is
never altered.

`training_seconds` is synchronized optimizer-step time, including closures,
line searches and transactional state handling. `history.csv` elapsed_seconds
is active process wall time, also including audits/checkpoint I/O, excluding
preemption downtime. Warm history losses are evaluated **before** each update;
quasi-Newton rows are trial evaluations. `accepted` only marks an evaluated state
that equals a returned accepted endpoint. Never label all rows accepted losses.
Runtime entries record GPU model/capability, hostname, Slurm job/restart counts,
Torch/CUDA/package versions and phase. Wall times are descriptive if hardware
changes; equal function/gradient evaluation counts remain the primary budget.

## Independent physical assessment

The fixed audit includes original unweighted/unscaled momentum residuals,
continuity, wall discrepancies, full-depth one-width shells, upper-left region,
both floor cuts and the excluded near-lid corner support. Envelope ranges and
physical velocity amplitudes accompany the residuals; tiny deep velocities are
not interpreted using training loss alone.

`prediction.npz` stores x and h axes and u,v,p,psi,omega arrays in (h,x) order.
Pressure is the raw neural output. Omega is native automatic differentiation,
explicitly labelled; CFD comparisons must compute vorticity from both sampled
velocity fields with the same differentiation operator. Apply one whole-interior
pressure-gauge alignment, preserve it in all regional RMS metrics, and report
regional pressure shape/offset only as additional diagnostics. Include physical
velocity, pressure and vorticity profiles and streamlines, not bar charts.

Reference fields are loaded only in post-freeze assessment. Use exactly matched
geometry/lid/Re, regions and physical units; compare vector-velocity absolute RMS
and RMS/reference-RMS together, especially near the floor. Report all paired
seeds and both improvements and deterioration. The current code does not claim
that the short cost gate reaches the retained paper's accuracy.

## Admission

Set `FLOWMLLAB_D5_SOURCE=data/week13_d5/author_source` and run:

```
python -m pytest -q tests/test_deep_cavity_training.py
```

The tests cover physical-y product-rule derivatives, third-order smoothness,
Gaussian wall values, continuity, raw-versus-weighted equations, frozen pairing,
chunk-gradient equivalence, four optimizer state roundtrips, budget-aborted line
searches, RNG recovery, full-versus-resumed PDE trajectories, immutable completion
and interrupted-export recovery. Repeat in the actual runtime before production.
Independent manufactured-solution and directional-gradient checks additionally
review the implementation; passing admission is not scientific convergence.
