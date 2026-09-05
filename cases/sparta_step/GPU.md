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
build, actual solver compatibility, measured speed, and short-run memory capacity.
It neither cancels nor duplicates the separate CPU campaign.

## One allocation, a measured comparison

`submit_gpu_benchmark.sh FULL_FLOWMLLAB_SHA` downloads immutable code and requests
one A40, 16 CPU cores, 128 GiB host RAM, and a four-hour wall limit in Unity's `gpu`
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
4. Run fresh fine-grid GPU capacity probes for h/H=0.16 with PPC20 (~70 million
   particles) and h/H=0.50 with PPC40 (~125 million). They initialize all production
   tallies and advance 70 steps; huge field/restart dumps are omitted. A failed
   probe records its failure while preserving completed paired benchmark results.

The benchmark requires the actual pilot's `pilot/restart.final` and `case.json` on
Unity. It does not try to manufacture particles from uploaded averaged fields.
The input restart is hashed, and mesh/physics/particle-weight provenance must
match. Each repeat starts at that same checkpoint: these are performance repeats,
**not three independent production datasets**. Random trajectories can differ
between backends even with identical seed labels.

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
existing mesh/dt/PPC/seed checks are still needed for scientific data acceptance.

`nvidia-smi` samples allocated-device memory every 500 ms. The maximum sampled
value is not an exact peak. A 70-step fresh-state capacity pass does not establish
memory headroom after long flow evolution or during checkpoint/output operations.
CUDA out-of-memory may mean using a larger-memory GPU or splitting one case
across GPUs; it must not be fixed by silently lowering particle count or resolution.

Use `gpu_benchmark.py status --out RUN_DIRECTORY`; it queries the exact saved job
ID through `sacct`, then shows timings and log tails. A successful job ends with
`SPARTA_GPU_BENCHMARK_COMPLETE TRAINING_DATA_APPROVED=False` and automatically
writes `gpu_benchmark_review.tar.gz`. `pack --out ...` also works after a failure.
The pointer is `LATEST_SPARTA_STEP_GPU_BENCHMARK`. Duplicate submission is refused;
`--new-run` is an intentional new benchmark, never a silent retry.

## Build and validation boundaries

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
[Unity GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/).
