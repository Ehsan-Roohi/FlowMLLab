# GPU route: benchmark before moving the step campaign

SPARTA revision `95b9abaa8bd548991cc3c3f1c58b34722f7ade74` includes Kokkos
implementations of our pressure-and-temperature `emit/face subsonic` boundaries,
VSS/VHS collisions with smooth rotational relaxation, diffuse walls, grid moments,
thermal/grid, boundary and surface tallies, and grid/check. Computation styles are
explicitly suffixed `/kk` in GPU decks. For walls, retain `diffuse` and let `-sf kk`
select the accelerated implementation: this source pin rejects an explicit
`diffuse/kk` at runtime because its dispatch expects the canonical name. This was
caught in the actual Kokkos host-backend regression. The `ave/time` and `ave/surf` fixes still
use host implementations; transfer/aggregation overhead is included in timings.

This is **not a GPU production campaign submission**. It establishes a usable
build, actual solver compatibility, measured speed, and memory use at the existing pilot scale.
It neither cancels nor duplicates the separate CPU campaign.

## One allocation, a measured comparison

`submit_gpu_benchmark.sh FULL_FLOWMLLAB_SHA` downloads immutable code and requests
one A40, 16 CPU cores, 48 GiB host RAM, and a four-hour wall limit in Unity's `gpu-preempt`
partition. Only batch jobs load modules. Use `--gpu a100` or `--gpu h100` after the
SHA to request a different documented architecture; the allocation must match.
No GPU speedup or four-hour completion time is promised by this wall limit.

1. Load `cuda/12.6` and `openmpi/5.0.3-cuda12.6`, check a real CUDA kernel, and record
   the allocated device UUID, compute capability, and memory. Build CPU and CUDA
   executables separately from the same pinned source. Use the same MPI library.
2. Solve tiny fresh cases at h/H=0.16, 0.50, 0.75 with both backends. Test CPU→GPU
   and GPU→CPU particle restarts, including two CPU ranks to one GPU rank. Require
   complete fields, positive moments, correct solid masking and area, thermal
   pressure consistency, valid block/wall outputs, and zero stuck particles.
3. Restart the completed CPU pilot `step-pilot-20260905T142106Z-8xo_trzs` on each
   backend, alternating arm order across three repeats. CPU uses 16 MPI ranks;
   GPU uses one rank and one device, on the **same allocated node**. Never reuse
   the old CPU pilot wall time as the speedup denominator.

The particle policy is fixed at the successful pilot: **1000 x 200 grid, PPC20,
unchanged particle weight**, and about **5.1 million initial simulated particles**.
No 70/125-million-particle memory probes, refined grids, or PPC40 cases are submitted.
A budget guard rejects accidental mesh/PPC enlargement, and this benchmark no
longer derives its dimensions from the archived 39-run refinement proposal.
The actual particle inventory may fluctuate through physical inlet/outlet transport;
particles are never deleted or rescaled to force an artificial exact count.

The benchmark requires the actual pilot's `pilot/restart.final` and `case.json` on
Unity. It does not try to manufacture particles from uploaded averaged fields.
The input restart is hashed, and mesh/physics/particle-weight provenance must
match. Each repeat starts at that same checkpoint: these are performance repeats,
**not three independent production datasets**. Random trajectories can differ
between backends even with identical seed labels.

## Preemption, checkpoints and migration

For the existing **pending** benchmark, use:

```bash
bash submit_gpu_benchmark.sh FULL_FLOWMLLAB_SHA --replace-job 64022083
```

The migration verifies the saved job ID, Slurm owner, job name and working
directory. It holds that pending job, submits the replacement on hold, installs
its recovery dependency, cancels only the old pending job, and releases the new
one. Its transaction record makes repeating the command idempotent. If the old
job has started or finished in the meantime, it reports that state and leaves it
intact. An interrupted migration can be completed by repeating the same command.
Unrelated jobs and the original CPU pilot files are never altered.

The GPU job has `--requeue` and append-mode Slurm logs. A small **CPU recovery job**
(one core, 512 MiB, five-minute limit) waits on `afterany` for it. If Unity cancels
a preempted allocation instead of requeuing it, that recovery job resubmits into
`gpu-preempt`, using the **same run directory and code**. Only `PREEMPTED`,
`NODE_FAIL` and `TIMEOUT` qualify. A user cancellation, solver failure or OOM does
not trigger an automatic retry. There is a maximum of eight allocation starts;
all files remain available at the limit. The recovery job being pending with
`Dependency` is normal. This does not rely on advance signal delivery: Unity
documents that preempt-partition jobs can be killed after two hours, and Slurm
does not guarantee a preemption warning from a wall-time `--signal` request.

Recovery boundaries for this short benchmark:

| Interruption point | What the next allocation does |
| --- | --- |
| Source checkout / compilation | Reuses pinned source and completed CMake objects |
| Tiny preflight | Reuses checksum-verified completed cases; retries only an incomplete case |
| Before the 700-step warmup checkpoint | Starts the incomplete arm from the original pilot restart |
| After the warmup checkpoint / during sampling | Reads sealed warm particles, re-equilibrates for 350 steps, then collects a **new full 1,400-step** window |
| After a verified arm | Reuses its fields, report and timings without rerunning it |

The warm particle checkpoint is written to a temporary name, closed by SPARTA,
then renamed and given an atomic SHA-256 receipt by the rank-zero `shell` command.
Unsealed files and checksum mismatches are skipped. Final-arm receipts are written
only after full output validation. Every attempt has its own directory, so partial
averages and logs are retained rather than overwritten or concatenated. This
checkpoint interval is a **stage boundary**, not a claim of saving every timestep.

SPARTA restarts do not store running averages, RNG state or collision maxima.
Consequently a resumed trajectory is not bitwise identical; its sampling window
is restarted after the extra equilibration. Timing records identify the allocation,
host and resumed status. Only uninterrupted CPU/GPU pairs in the same allocation
enter reported speedups. Cross-allocation pairs and resumed arms remain in the
report as diagnostics. If no pair qualifies, `speedups` is empty; no misleading
mixed-node median is printed. Lost attempts are excluded from successful-attempt
wall time and retained in their own logs.

`gpu_benchmark.py status --out RUN_DIRECTORY` shows the current GPU ID, allocation
history, recovery job IDs, completed-arm count and committed checkpoint count.
`verify_gpu_resume.py` deliberately SIGKILLs real CPU and Kokkos host-backend
solvers during sampling, rejects a truncated newer checkpoint, resumes and checks
the complete new block window, and ensures completed arms are not rerun. Its
mock-Slurm checks cover pending-job migration and bounded retry policy. Actual
Unity preemption and CUDA execution remain allocation-side checks.

## Interpreting the report

`gpu_benchmark_report.json` records median CPU/GPU timing ratios separately for:

| Quantity | What is timed |
| --- | --- |
| Warmup loop | 700 steps of transport, collisions, pressure boundaries and checks |
| Sampling loop | 1,400 steps with grid, mass-flux, surface and boundary tallies, compressed block output |
| End-to-end | MPI startup, checkpoint reads/writes, setup, all loops and output; Python postprocessing excluded |

Both timed arms use the campaign timestep `1.0254861656216755e-11 s` and cell
moment cadence of 35 steps. To produce four blocks in a short run, output blocks
are every **350** steps instead of the production **35,000**: sampling timing is
an intentionally frequent-output workload, not a production wall-time forecast.
The sampling interval is only `1.4356806318703457e-8 s`. Short-window field L2,
mass-flow and pressure differences are diagnostics, **not a validated CPU/GPU
statistical equivalence test**. A longer matched sampling comparison and the
numerical and sampling sensitivity checks are still needed for scientific data acceptance;
this performance test does not approve the archived expensive refinement matrix.

`nvidia-smi` samples allocated-device memory every 500 ms. The maximum sampled
value is not an exact peak. It measures the existing pilot case; it does not
establish capacity for a larger simulation. Increasing particle count is outside
this run's scope. Any future dataset plan must respect the same particle budget.

Use `gpu_benchmark.py status --out RUN_DIRECTORY`; it queries the exact saved job
ID through `sacct`, then shows timings and log tails. A successful job ends with
`SPARTA_GPU_BENCHMARK_COMPLETE TRAINING_DATA_APPROVED=False` and automatically
writes `gpu_benchmark_review.tar.gz`. `pack --out ...` also works after a failure.
The pointer is `LATEST_SPARTA_STEP_GPU_BENCHMARK`. Duplicate submission is refused;
repeating a migration command prints its saved status instead of creating another run.

## Build and validation boundaries

The 2026-09-05 A40 allocation 64023942 built both binaries but stopped at the
shared-library gate: Unity's CUDA module put `nvcc` on PATH without adding
`libcudart.so.12` to the runtime search path. No solver timing resulted. The
launcher now resolves that library only inside the selected compiler's toolkit,
records its path/hash per allocation, prepends its directory for batch children,
and verifies the linked GPU library matches it. It does not install packages,
change the login environment, or accept a CUDA stub/another toolkit. The Linux
regression is `python qa/test_sparta_cuda_runtime.py`; a real GPU benchmark is
still required after these path-resolution tests.

The bundled Kokkos is 5.0.2, requiring C++20, CMake >=3.22 and NVCC >=12.2.
The upstream CUDA preset defaults to Hopper90, so A40 explicitly selects
AMPERE86 and A100 AMPERE80. A tiny CUDA kernel fails early on a toolkit/driver or
device mismatch. We disable asynchronous CUDA allocation and GPU-aware MPI
communication for this one-GPU test. The latter avoids assuming a working direct
GPU MPI transport; it does not turn off GPU computations.

`verify_gpu.py` runs the actual pinned Kokkos **Serial backend** in ordinary CI,
with a separate plain CPU executable. It exercises the same styles, fresh solves,
restart conversions, tally deck and timing parser. This check does **not** compile
CUDA or execute an NVIDIA device; those checks occur in the Unity allocation.

Sources: [SPARTA acceleration manual](https://sparta.github.io/doc/Section_accelerate.html),
[pinned pressure-boundary implementation](https://github.com/sparta/sparta/blob/95b9abaa8bd548991cc3c3f1c58b34722f7ade74/src/KOKKOS/fix_emit_face_kokkos.cpp),
[Unity modules](https://docs.unity.rc.umass.edu/documentation/software/modules/module-usage/),
[Unity GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/),
[Unity partitions](https://docs.unity.rc.umass.edu/documentation/cluster_specs/partitions/),
[Slurm preemption](https://slurm.schedmd.com/preempt.html),
[SPARTA restart limitations](https://sparta.github.io/doc/read_restart.html).
