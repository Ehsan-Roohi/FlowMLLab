# Controlled area-stratified cavity PINN comparison

This is one bounded sampling intervention after the original finite-budget cavity
PINNs. It is not a new architecture or optimizer search. The earlier notebook,
original training engine, original runner, and all historical results remain
unchanged.

## Scientific question and fixed scope

Can better spatial coverage improve a data-free PINN at the **same number of
residual evaluations**, without changing the mathematical objective's area
measure? Test only Re = 100 and 7500, seeds 7, 23, and 61, and upper-corner collars
0.02 and 0.01: 12 fresh trajectories. Each is paired with its existing completed
baseline at the same Re, seed, and collar. Stop after this intervention whether
it improves or worsens the result. No CFD/Ghia values are read by training, and
no residual, validation score, or plotted endpoint selects the training duration.

The original constant-speed lid, analytical upper-corner Stokes lifting,
streamfunction boundary construction, pressure gauge p(0.5,0.5)=0, physical
coordinates, and Navier–Stokes derivatives are imported directly from the
unchanged `flowmllab/cavity_pinn_campaign.py`. Its canonical SHA256 is
`507a6e0a6d632a173c8bdc8748564beffee77ce3664aeffcee25edc5ca305b9e`.
The import refuses a changed source. This is not the Gaussian-lid D/W=5 study.

All production arms retain four 64-neuron tanh hidden layers, float64, identical
seeded initial model tensors, 2048 collocation points, Adam(lr=0.001) for 2000
updates, followed by the original strong-Wolfe L-BFGS parameters. The baseline
`lbfgs_steps=500` remains recorded as a paired control setting; **the new arm
stops by total closure work, not by completing 500 new L-BFGS updates**. The
number of accepted quasi-Newton updates can consequently differ.

## Fixed, conditional-uniform spatial strata

Let R=0.05, epsilon be the declared collar, and r be the smaller distance to the
two upper corners. The lower corners are not excluded. Partition the original
punctured training domain into these disjoint strata (boundaries have zero area):

| Stratum | Support | Area | Points |
| --- | --- | --- | ---: |
| Upper-corner annuli | epsilon < r < R, within the square | pi (R²−epsilon²)/2 | 512 |
| Remaining wall band | Outside (R,1−R)² and outside both upper radius-R disks | 1−0.9²−pi R²/2 | 512 |
| Interior | R < x,y < 1−R | 0.9² | 1024 |

The total retained area is A=1−pi epsilon²/2. For a point in stratum s, use
weight **w_i=A_s/(A n_s)**. Minimize

\[
\widehat L = \sum_s \frac{A_s}{A n_s}\sum_{i\in s}
\left[R_x(x_i,y_i)^2+R_y(x_i,y_i)^2\right].
\]

The weights are positive and sum to one. This estimates the **same normalized
area objective** as the original uniform-on-the-punctured-square sampler;
it does not give each stratum equal objective weight. The 512 annular points
are split equally between the two equal-area corner annuli. Their radii are
sampled as sqrt(epsilon²+(R²−epsilon²)U), and their angles uniformly in
(0,pi/2). Sampling radius uniformly would be incorrect. The wall band uses
conditional rejection sampling; the interior is sampled directly. All points,
weights and stratum IDs are frozen before fitting. No points are chosen from
residual magnitudes, and no points are redrawn during optimization.

## Matched work and predeclared endpoints

Baseline evidence comes from the verified recovery archive
`ghia_full_recovery.tar.gz`, SHA256
`d041091ee84b5e1150fc186b2f15087852f0e1d2929be78ce96d81558c953235`.
Only its training metadata are extracted. Each corresponding checkpoint was
streamed from the archive to verify its SHA256 against the original receipt;
no checkpoint or scientific field was copied into the metadata directory.

| Re | Seed | Total closures, epsilon=0.01 | Total closures, epsilon=0.02 |
| ---: | ---: | ---: | ---: |
| 100 | 7 | 3017 | 3017 |
| 100 | 23 | 3029 | 3011 |
| 100 | 61 | 3005 | 3005 |
| 7500 | 7 | 3023 | 3023 |
| 7500 | 23 | 3049 | 3015 |
| 7500 | 61 | 3008 | 3008 |

These values are read from the completed baseline receipts and bound before
variant initialization; they are not chosen from variant results. Every
successful closure evaluates all 2048 points. Count initial closures, repeated
closures, rejected line-search trials, and the work consumed by an unfinished
final line search. The total training residual-point count must equal
2048 times the paired baseline closure count. Counting only accepted updates
would hide substantial work and is not the comparison used here.

If the cap interrupts an L-BFGS update, restore the complete model and optimizer
state immediately preceding that update. Keep the spent closure/residual counts
and trial history. Freeze that last accepted state. A rejected trial's loss is
not its loss. Recompute the accepted endpoint's weighted objective **once as a
separate audit**, with its own audit counter, excluded from the training cap.
This audit cannot select the endpoint or trigger more training.

The seed-23 collar caps differ. Do not call those two runs an equal-work collar
convergence experiment. The primary contrast is always baseline versus
stratified at the same Re, seed, collar, and total closure work. Wall time is
secondary and descriptive: hardware and restart provenance are retained, and
the additional diagnostic work is reported separately.

## Recovery and immutable evidence

`cavity_pinn_stratified.Trainer` saves the model, complete Adam/L-BFGS state,
Python/NumPy/Torch/CUDA RNG states, sampler RNG, fixed points, weights, stratum
IDs, independent audit locations, stage/counters, full trial and step histories,
initial-state identity, paired-baseline evidence, and frozen endpoint metadata.
The optimizer, configuration, objective tensors, model, and complete nested
checkpoint each have SHA256 bindings. The original physical source and new
engine identity are bound by both canonical and exact raw-byte hashes; the CLI
also binds its runner identity. Preserve the original staged file bytes,
including line endings. The original training loader rejects changed
configurations, samples, weights, optimizer/RNG contents, source, or initial
tensors. A process lock prevents duplicate writers.

Frequent saves occur after complete steps and use an atomic replace. SIGTERM,
SIGUSR1 and SIGINT request a stop after the current step. A complete recovery
retains the declared schedule and accepted optimizer state. A new entry records
GPU name/capability, CUDA/Torch, hostname, Slurm job ID and restart count. No work
is launched by this module. Campaign orchestration owns workspace discovery,
capacity, preemption signals, submission and archive verification.

The final bundle binds checkpoint, budget contract, points/coverage metadata,
step history, closure history, predictions and physics audit by hash. On an
admitted compatible runtime, reopening a complete, valid bundle is a
verification-only operation; interrupted exports preserve the saved freeze
time. A failed hash check is an error, not permission to overwrite evidence.
The restart tests establish recovery on the tested runtime. **They do not
establish bitwise optimizer continuation across arbitrary hosts or platforms.**

### Observed platform-dependent sampler reconstruction

All 12 training endpoints completed and were frozen. The first generic scoring
attempt scored four Re=100 cases (seeds 7 and 23, both collars), then rejected
the other eight during `Trainer.load` with `Fresh initial model or deterministic
weighted sampler changed`. The original loader regenerates the initial model
and sampler as an additional check. This check is stricter than loading the
saved objective tensors.

Read-only checkpoint forensics found that all 12 saved checkpoint, complete
state, model and weighted-objective hashes remained valid. Fresh initial-model
tensors, weights, stratum IDs and independent audit locations matched. On the
CPU forensic host, four regenerated point sets differed from their saved values
in only 4--6 scalar coordinates in the annular stratum, by at most
1.1102230246251565e-16; the other eight matched exactly. The pattern was reversed
on the earlier GPU evaluation host. This establishes platform-dependent
last-bit sampler reconstruction, not checkpoint corruption. Forcing AVX2 versus
AVX512 on the same forensic host did not reproduce the host difference; the
specific math-library or dispatch cause remains unresolved.

The compact receipt is
`results/cavity_pinn_stratified/sampler_restore_forensics.json`, SHA256
`834814f4aec9c6542de4e0096a46795bf40de103b5a5ca5d3c75d76a9c29af68`.
The training engine, runner and original checkpoints remain unchanged. The
repair does not loosen a numerical tolerance, resample points, refit a model,
change an endpoint or alter a physical equation.

### Exact saved-state inference admission

`qa/evaluate_cavity_pinn_stratified.py` is a separate **evaluation-only** adapter.
It verifies checkpoint bytes, the complete nested payload, configuration and
source identities, the frozen model, optimizer state, saved points/weights/IDs,
audit points, RNG contents, work counters and paired-baseline evidence. It then
constructs only the original network's parameter containers and loads the exact
saved final tensors. It neither invokes `Trainer` nor regenerates the sampler.
Original saved state is authoritative for this inference operation; no new
objective is constructed from approximately equal points.

The adapter exposes prediction, not optimizer stepping, checkpoint saving or
training continuation. Optimizer and RNG identities are verified, but the
adapter does not certify their continuation on a different platform. An
independent load-only admission checked all 12 actual checkpoints with the
pinned Torch 2.5.1 runtime, preserving checkpoint bytes and process RNG states;
that admission read no reference values and made no field predictions. Its
receipt is `frozen_inference_cpu_admission_current.json`, SHA256
`77a6d31164f18a2884745d8f47f82ed9311aa4f4846997da2731d748933d2dd8`.
Passing this admission is evidence of exact restoration, not of flow accuracy.

## Admission and reproduction

Install the same admitted Torch/NumPy runtime as the paired campaign. The
focused training gates are:

```bash
python -m pytest tests/test_cavity_pinn_stratified.py -q
```

They check exact partition/counts and top-only exclusions; area-uniform annular
radii and analytic integrals; positive area weights; all three original
initializations; the original manufactured-equation/boundary gate; full versus
chunked weighted gradients and directional finite differences; Adam and L-BFGS
restart; budget abort with populated L-BFGS curvature; point/weight/ID/optimizer/
RNG/config tamper rejection; and immutable frozen export/reconstruction.
`study_kind='cpu_admission'` permits tiny synthetic unit fixtures. The production
CLI never accepts that flag or a small-network substitution.

The following documents the original training invocation; it is not a request
to rerun the completed study. A single counterpart uses its hash-verified four
small baseline JSON files and the actual allocated live workspace:

```bash
python qa/run_cavity_pinn_stratified.py \
  --baseline-metadata BASELINE_CONTROLS/ghia_re100_seed7_eps0.02_chunk2048 \
  --output LIVE_WORKSPACE/stratified/ghia_re100_seed7_eps0.02_chunk2048 \
  --device cuda --checkpoint-seconds 120
```

`--resume` is restricted to a compatible, admitted recovery environment and
must pass every existing check. A platform-dependent reconstruction failure is
not permission to alter the saved samples or bypass a training-loader check.
`--max-steps` and `--max-wall-seconds` limit one process entry only and never
change the declared closure budget. Do not create concurrent writers or restart
a completed arm as a fresh trajectory. All live output belongs in the actual
allocated HPC Workspace, not a durable `/project` directory.

## Evaluation and reporting

The required order is **exact-state inference wrapper, provenance bridge, then
paired comparator**. Do not call the generic evaluator's `Trainer.load` path
for this repaired study. The wrapper reuses the unchanged generic scoring
formulas after preparing all 12 native predictions successfully.

### Target-access chronology and completed-pass barrier

The earlier four-case partial scoring attempt already accessed CFD/Ghia
reference values. All 12 original trained endpoints had been frozen before that
access, and none was refit or selected afterward. The failed attempt and its
four derived records are preserved in a compact hash-bound provenance receipt.
The restoration change was made to enable inference from the existing saved
state, not in response to a reference error or a favorable case.

For the completed scoring pass, the wrapper must prepare and verify **all 12
native predictions before its first reference-value read**. If any projection
fails, that pass does not proceed to scoring. This is a completed-pass barrier;
it is not a retrospective claim that every native projection preceded the
first-ever target access. Published provenance must retain these distinctions:

- All training endpoints frozen before the earlier target access: true.
- All native projections frozen before the completed pass's reference read:
  required to be true before a complete comparison is admitted.
- All native projections frozen before the first-ever reference read: false.
- Earlier partial attempt: four cases scored, with no subsequent fitting or
  endpoint changes.

Use the retained evaluation output containing the earlier partial manifest and
its four artifacts, or its already verified prior-attempt receipt. Do not start
in an empty output directory that erases this chronology. The wrapper creates
its `completed_pass/` artifacts separately and preserves the previous attempt's
identity. Frozen prediction identities and original timestamps are checked on
replay; changed inputs or an incompatible manifest are rejected. Superseded
bulky derived arrays may be cleaned only after replacement verification and
preservation of the compact prior-attempt evidence.

### Reproduction command order

Run the following inside an admitted compute allocation, with the exact staged
source bytes, the verified forensic receipt and original frozen runs available.
Replace the illustrative run-root value with the actual allocated path returned
by workspace discovery. These commands perform evaluation only; they do not
submit jobs or train new models. Bash array expansion supplies all 12 runs to
both the wrapper and bridge:

```bash
STRATIFIED_RUN_ROOT=/actual/allocated/workspace/pinn-stratified-v1/runs
stratified_run_args=()
for re in 100 7500; do
  for seed in 7 23 61; do
    for eps in 0.01 0.02; do
      stratified_run_args+=(--run "$STRATIFIED_RUN_ROOT/ghia_re${re}_seed${seed}_eps${eps}_chunk2048")
    done
  done
done

python qa/evaluate_cavity_pinn_stratified.py "${stratified_run_args[@]}" \
  --output results/cavity_pinn_stratified --device cuda \
  --forensics results/cavity_pinn_stratified/sampler_restore_forensics.json

python qa/bridge_cavity_pinn_stratified.py "${stratified_run_args[@]}" \
  --raw-manifest results/cavity_pinn_stratified/evaluation_manifest.json \
  --output results/cavity_pinn_stratified \
  --baseline-controls results/cavity_pinn_stratified/baseline_controls/controls_manifest.json \
  --runtime-events results/cavity_pinn_stratified/runtime_events.json

python qa/compare_cavity_pinn_stratified.py \
  --baseline-manifest results/cavity_pinn_campaign/evaluation_manifest.json \
  --variant-manifest results/cavity_pinn_stratified/evaluation_manifest.json \
  --experiment-manifest results/cavity_pinn_stratified/experiment_manifest.json \
  --output results/cavity_pinn_stratified/comparison
```

The runtime-events receipt records the actual failed allocation and checkpoint
recovery. The bridge binds small original training and baseline-control metadata,
the unchanged raw score records, completed-pass barrier, prior failed attempt,
checkpoint forensics and inference-wrapper source. The comparator carries that
chronology and those hashes without changing either raw score manifest. Run the
focused evaluation, bridge and comparison tests when changing this postprocessing
code:

```bash
python -m pytest tests/test_cavity_pinn_stratified_evaluation.py \
  tests/test_cavity_pinn_stratified_bridge.py \
  tests/test_cavity_pinn_stratified_comparison.py -q
```

### Interpretation and limitations

Keep the method label explicit: area-weighted stratified sampling versus
original uniform fixed sampling. The generic scorer's method label alone does
not establish this distinction. Retain unchanged independent uniform audit
locations and all original corner shells, including shells inside the fitting
collar. Show signed velocity centerlines against CFD/Ghia, physical streamlines
and error fields, and report both improvement and deterioration across the three
paired seeds. Pressure has no CFD reference in this baseline and must not be
presented as validated pressure accuracy. Training completion, hard boundary
satisfaction and machine-small continuity do not establish high-Reynolds
accuracy. No full-domain corner-integrability claim follows from a finite
punctured-domain collocation experiment.

The comparison outputs are `paired_metrics.csv`, `paired_signed_profiles.csv`,
and `comparison_manifest.json`. Each pair retains its exact work cap and
recovery identities. Signed errors are retained; for a signed bias, improvement
means a smaller absolute bias, considered alongside RMS error. The report lists
wins and losses per metric, descriptive three-seed summaries, and all six collar
groups with both caps. It does not select an overall winner or assert collar
convergence. In particular, the seed-23 collar caps are unequal.

Optimizer seconds describe measured optimizer-step work. They exclude failed
allocations, queue/requeue delays, checkpoint I/O and separate audits; they are
not total campaign wall time. Hardware and runtime-event records must accompany
any timing comparison. Function/gradient and residual-point counts provide the
primary matched-work comparison.

The comparator reads existing scores and profile samples. It does not train,
load CFD field arrays, interpolate fields, or invent a pressure reference. Its
metadata-only unit fixtures are explicitly synthetic and are never presented
as scientific results. Use the final verified manifests to establish whether
the completed comparison is available; successful restoration alone is not an
accuracy result.
