# Step architecture comparison V5

This is a development experiment, not release accuracy evidence. It tests
whether the small geometry-feature MLP, ordinary DeepONet, or the existing
FlowMLLab Geom-DeepONet performs better under a shared training protocol.
It does not replace the accepted teaching outputs or continue V4 weights.

## Frozen scope

- Development heights: 16, 21, 25, 50, 75 percent; validation: 33, 58 percent.
- Kn=0.01; the same smoothed DSMC learning archive as V3/V4, hash
  `410907d46a040d53cbbd19fd8d44eeb7b41c05150f953fd4d9c6bb479da3479d`.
- H44/H67 were already inspected in earlier development. They are neither
  downloaded/copied nor opened here. An audit hook blocks the named test file.
- These validation geometries have also been used repeatedly. No independent
  generalization, low-data superiority, or publication claim follows from V5.
- Source helpers are pinned to commit
  `2e9199bfa6efafd60c506374029c30aa3b4e009e` with individual hash verification.
  This retains the reviewed step functions even if unrelated main-branch code changes.

## Models and controls

| Arm | Inputs | Architecture |
|---|---|---|
| `mlp` | h, normalized x/y, step-relative x, wall-relative coordinate, hx, hy, xy | two 48-wide tanh layers; practical small baseline |
| `deeponet` | branch: h; trunk: normalized x/y | two 128-wide tanh layers per branch/trunk; rank 48; two vector components |
| `geom` | branch: h; trunk: normalized x/y and analytic SDF/H | unchanged V3/V4 helper, width 48, SIREN omega=10, intermediate fusion and query pooling |

DeepONet has 52,242 parameters; Geom has 52,298; MLP has 2,882. The MLP is
deliberately the small existing practical architecture. Different inputs and
activations mean this is a whole-method comparison, not an isolated SDF ablation.

All three run from scratch with seeds 690, 691, 692 under both uniform sampling
and zonal sampling (alpha=0.6), for **18 controlled fits**. There is no alpha sweep.
For a given seed and sampler, every model receives exactly the same 60,000 drawn
development rows, including repeated zonal rows, and exactly the same batch order.
Sampling matches the original sklearn helper's global-pool rule. Zonal sampling
draws 36,000 rows from training U<0 and 24,000 from training U>=0. Repeats do not
count as additional CFD data; draw and unique-row counts are stored separately.

Inputs never contain reference velocities or a target-derived mask. A training
label may determine sampling priority; it is not a prediction input. Scalers
are fitted to all development rows before sampling, never validation targets.

Common training: component StandardScaler targets (including mean subtraction),
plain mean squared error, float32 Adam at 8e-4 with epsilon 1e-7, 90 epochs, no
early stopping, no learning-rate sweep, and no regularization. Batches contain
one geometry and at most 1024 drawn rows; no rows are dropped or padded. The same
geometry grouping is applied to the MLP and ordinary DeepONet. Actual optimizer
updates, target exposures, row hashes, and batch-schedule hashes are checked.
Costs are equal within a sampler; different sampler counts by geometry can
produce slightly different numbers of short batches between samplers.

The original Geom helper pools its query cloud. Thus zonally sampled training
clouds and full-field inference clouds need not produce the same geometry
embedding. V5 retains that behavior rather than silently altering the model.
A separate 1024-point context probe reports prediction change relative to the
reference norm, globally and in U<0. It is not model error or a mesh-independence
claim. All model-selection metrics use the full validation cloud consistently.

## Historical sklearn anchor

Two additional fits call the original sklearn helper unchanged, seed 690,
60,000 samples, up to 90 iterations, uniform and alpha=0.6. They use only the
five development geometries and predict only H33/H58. The expected validation
global/vortex errors are 5.445122169/87.963854380 percent (uniform) and
7.320898032/33.811286542 percent (zonal). Differences and a 0.01 percentage-point
reproduction tolerance are reported without concealing framework/platform drift.

These fits retain sklearn's mixed-geometry batches, initialization, regularization,
and stopping behavior. They are reproduction anchors, not additional members of
the shared TensorFlow optimizer comparison. Their weights, scalers, loss curves,
and validation predictions are saved without pickle. No development+validation
refit and no comparison to H44/H67 take place.

## Checkpoint selection and evidence

Validation is evaluated every five epochs and at the terminal epoch. For each
seed, define a shared global ceiling as the lowest logged MLP-uniform validation
global relative L2 plus **2 percentage points**. This is fixed as an algorithm
before running, uses no test data, and is shared by all six arms for that seed.
Each arm selects the eligible checkpoint with minimum mean validation vortex
relative L2; ties favor lower global error, then earlier epoch. There is no
epoch-zero fallback. If an arm has no eligible checkpoint, it is reported as
such and the all-seed selected aggregate remains NA; failed seeds are not removed
to manufacture an average. Terminal results are always reported separately.

The ceiling is a validation selection policy, not a guarantee on individual
geometries or unseen tests. Both validation geometries are equally weighted.
Metrics use physical joint U/V relative L2, separate U/V RMSE in the full field,
reference U<0 and main flow, negative-U IoU and predicted/reference negative-cell
count ratio. A negative-U footprint is not wall-shear-based reattachment length.

Selected checkpoint restoration is checked against logged validation metrics.
The report contains paired deltas against the MLP and Geom against ordinary
DeepONet, separately for selected and terminal models. No automatic winner or
claim of improvement is issued. Check convergence, all seeds, context shifts,
and per-geometry physical fields before proposing the next experiment.

## Unity execution

The launcher uses the existing `conda-tf220-clean` environment and A40 partition
settings already used successfully in V4. It installs no packages and does not
change the working checkout or existing run directories. It snapshots exact
reviewed local dependencies, fetching the pinned public versions only if needed.
An advisory lock and active-job check avoid accidental duplicate V5 submissions.
Run the downloaded runner with the environment's Python and `-I`:

```bash
"$BASE/conda-tf220-clean/bin/python" -I step_architecture_v5.py submit
```

The submitted job is `step-arch-v5`, 1 A40, 4 CPUs, 24 GB, a two-hour limit.
Full GPU execution is performed on Unity, not by the code-review environment.
The successful submission records its job ID and updates
`LATEST_STEP_ARCHITECTURE_V5`. The terminal command itself does not set `-e`,
change the parent shell's environment, activate conda, or exit the login shell.

```bash
(FLOW_BASE=/scratch4/workspace/roohie_umass_edu-mfc-a40-cv/flowmllab-geomdeeponet; FLOW_OUT=$(cat "$FLOW_BASE/LATEST_STEP_ARCHITECTURE_V5") && "$FLOW_BASE/conda-tf220-clean/bin/python" -I "$FLOW_OUT/step_architecture_v5.py" status) || echo V5_STATUS_FAILED
```

Replace `status` with `bundle` to build a review ZIP with histories, source,
learning data, row/schedule provenance, selected and terminal weights/predictions,
logs, and SHA256SUMS. Intermediate checkpoints stay in the original run.

## Local execution verification

```bash
python -I qa/test_step_architecture_v5.py
python -I qa/step_architecture_v5.py smoke --source-root /path/to/FlowMLLab --out /new/smoke/directory
```

The smoke run uses all six model/sampler combinations for one seed, 1024 draws,
two epochs, and CPU-compatible execution. It validates training, selection,
restoration, metrics, matched budgets and packaging, not scientific accuracy.
Its completion status is distinct from the full GPU experiment.

Pre-submission verification on 2026-09-05: 13 protocol tests passed, including
test-file access blocking, duplicate submission prevention, shell quoting, exact
draw preservation, metric definitions and no-eligible checkpoint behavior. A
six-arm CPU smoke passed with TensorFlow 2.20.0, Keras 3.15.1, NumPy 2.2.6 and
scikit-learn 1.6.1. It checked equal update/exposure/schedule records, H5 restore,
and review packaging. Independently calling the original sklearn helper reproduced
both historical validation rows within 0.01 percentage points. These are execution
and reproduction checks; the 18-fit A40 comparison has not been run here.

Reference: He et al. (2024), Geom-DeepONet,
<https://doi.org/10.1016/j.cma.2024.117130>. This is a 2D FlowMLLab adaptation,
not a reproduction of the original 3D mechanics benchmark.
