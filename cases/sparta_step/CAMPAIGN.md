# Pressure-driven SPARTA step campaign

> Archived refinement proposal: this matrix exceeds the current pilot-scale particle
> budget and is not the current execution plan. Start with the [fixed-budget GPU
> benchmark](GPU.md); do not use the older 39-run submission command for this stage.

This campaign follows the successful Unity pilot **64013621**. It keeps N2,
VHS via VSS alpha=1, 300 K diffuse walls, pressure ratio 2, outlet-reference
Kn=0.01, the 85.47 micrometre channel length, and step location x/L=0.3.
The molecular conventions remain explicit; this is not verified parity with
the original Bird input used in the manuscript.

## What the single submission contains

| Stage | Cases | Purpose |
|---|---:|---|
| Mesh study at h/H=0.50 | 9 | 1000x200, 2000x400, 3500x700; three independent seeds each |
| Fine grid, half timestep | 1 | Time-discretization sensitivity at seed 20260905 |
| Fine grid, 40 particles/cell | 1 | Particle-number sensitivity at seed 20260905 |
| Other geometries | 28 | Fourteen heights, two seeds each; 3500x700 grid |
| Total | **39** | **15 distinct step heights** |

Heights h/H: **0.16, 0.21, 0.25, 0.30, 0.33, 0.36, 0.40, 0.44, 0.50, 0.58,
0.60, 0.64, 0.67, 0.70, 0.75**. Geometry 0.50 already has three fine-grid seeds.
These are solver diagnostics; no neural model or ML split is selected here.
Do not describe reused geometries from previous ML experiments as newly blind
test evidence.

The timestep is **1.0254861656e-11 s on all three meshes**. Thus the mesh study
does not confound refinement with a timestep change. The half-dt control uses
5.1274308281e-12 s. Fresh cases warm up for 2.8713613 microseconds, then sample
for **4.3070419 microseconds**, six times the pilot sampling duration. Twelve
nonoverlapping blocks are retained, with equal physical block duration and
cell-moment sampling cadence across every resolution and timestep variant.
Nonoverlap does not imply statistical independence.

When available and provenance-compatible, the first coarse-grid case continues
the known pilot restart and adds 0.7178403 microseconds of warmup before the new
sampling window. Other seed replicates start independently. Missing pilot
restart falls back to a fresh run. An incompatible existing restart is an error.

## Resources and cost

These are **CPU MPI jobs**, with no GPU or TensorFlow dependency.

| Group | MPI ranks/job | RAM/job | Concurrent jobs | Time limit/job |
|---|---:|---:|---:|---:|
| Coarse controls | 16 | 48 GiB | 1 | 24 h |
| Medium controls | 32 | 96 GiB | 1 | 72 h |
| Fine controls/sensitivities | 64 | 192 GiB | 3 | 7 days |
| Geometry array | 64 | 128 GiB | 4 | 7 days |

Validation can use at most 240 simultaneous cores; geometry jobs at most 256.
Each job is confined to one x86_64 node. No change to the user's other jobs is
made. Unity's documented CPU partition maximum is 14 days:
https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/

The run table prints a sizing estimate extrapolated from the 62-minute pilot,
using particle count, number of timesteps and ideal rank scaling. Fine-grid
standard cases are roughly **37 hours each under that assumption**; half-dt or
double-particle controls roughly double it. Actual scaling, node speed, I/O and
queue delay are not known. This is a multi-day to multi-week campaign, not a
collection of one-hour pilots. `plan` prints the complete per-case estimates.

The default submission requires **500 GiB free filesystem headroom** for raw
blocks, cumulative fields and particle restarts. It checks filesystem free
space, not a user quota. Scratch data retention remains the user's responsibility.
The final review package excludes large restarts and full spatial dumps;
request `pack --case-id CASE` for one case's full fields.

## Submission and dependencies

Run `bash submit_campaign.sh FULL_COMMIT_SHA` as a child process. The bootstrap
downloads a small immutable case snapshot instead of cloning the repository.
It submits build/preflight, three control arrays, a validation gate, a geometry
array and a collector. Every dependent stage uses **afterok** and invalid
dependencies are cancelled rather than being left pending indefinitely.

The bootstrap writes the exact directory to `LATEST_SPARTA_STEP_CAMPAIGN` and
records each successful sbatch response immediately in `manifest.json`. A
partial submission is retained explicitly. A second normal invocation refuses
to duplicate the campaign; `--new-campaign` intentionally creates a separate one.

Modules are loaded only inside the Slurm jobs. The known-good OpenMPI 5.0.3
build path, matching launcher/library hashes, preserved HOME, short writable
node-local TMPDIR, stdin input deck and pre-read_surf clumped grid partition are
kept. The build also executes actual fresh and changed-rank restart smoke tests.

## Validation screen before geometry jobs

Predeclared screens on the eleven control cases:

- Positive net flow, mass mismatch below 2%, bulk half-window drift below 2%.
- Reverse axial momentum half-window drift below 10% in 0<x-x_s<3h, 0<y<h.
- Boundary-adjacent mean pressure within 10% of the reservoir targets.
- dt/tau < 1/4; fine-grid cases also require maximum cell/lambda <= 1/3.
- Medium versus fine three-seed means: vector velocity difference below 5%
  globally, below 10% in the fixed near-step window; pressure difference below 3%.
- The same field-difference limits for half-dt and double-particle fine controls
  against their corresponding seed-0 fine control.

Fields are compared through conservative spatial bins common to the three
meshes. Coarse/fine differences are reported but do not veto a successful
medium/fine refinement screen. Every control still must pass the flow and
stationarity screens. Seed averaging is not presented as a confidence interval,
and the single-seed sensitivity differences contain Monte Carlo variability.

Passing is an engineering screening result, not a proof of statistical
equivalence or publication validation. A failed screen records
`validation_report.json`, exits nonzero and prevents the geometry array from
starting. It does not silently loosen thresholds. Every completed geometry has
its own diagnostics. `training_data_approved` remains false throughout.

## Outputs and interrupted runs

Each solver attempt has its own immutable directory. It retains the input,
case/execution metadata, twelve raw block fields, cumulative field, raw surface
and boundary statistics, flux blocks, `restart.warm`, `restart.final`, physical
diagnostics, comparison probes and checksums. Existing attempts are not overwritten.

A retry from `restart.warm` must create a **new attempt** and collect a full new
sampling window after an additional settling block. SPARTA does not persist
running-average tallies or collision models in a restart. Both are explicitly
recreated, and the old/new sampled fields are never silently stitched together.
https://sparta.github.io/doc/read_restart.html
https://sparta.github.io/doc/fix_ave_grid.html

`resume --out "$OUT"` checks that all recorded jobs have left the queue, then
submits only incomplete cases and rebuilds their downstream dependencies. It
uses an intact warm restart when present, otherwise starts that case fresh.
Old attempts remain available. A failed scientific gate with otherwise complete
controls is not retried automatically: that requires a new scientific decision.

The completion marker requires the expected solver step, zero stuck particles,
correct raw field identity/coverage, a successful report and a final restart.
The collector additionally requires every case marker; it does not infer
success merely from its own exit code. Always inspect `sacct` and the per-case
markers together.

Commands use the immutable copied code:

```bash
python3 -I "$OUT/code/campaign.py" status --out "$OUT"
python3 -I "$OUT/code/campaign.py" resume --out "$OUT"
python3 -I "$OUT/code/campaign.py" pack --out "$OUT"
python3 -I "$OUT/code/campaign.py" pack --out "$OUT" --case-id grid_fine_s20260905
```

The figures from pilot 64013621 are evidence for that pilot only. Smoke tests
verify execution, restart handling and output checks, not campaign convergence.
