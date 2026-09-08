# Deep-cavity CFD reference campaign (research work, not retained evidence)

Reference: M. Cheng and K. C. Hung, *Computers & Fluids* 35 (2006),
1046–1062, https://doi.org/10.1016/j.compfluid.2005.08.006.

## Scope and solver choice

Use OpenFOAM v2406, double precision, laminar incompressible simpleFoam,
as an independent finite-volume **steady solution candidate** generator.
The paper used LBM; these are not reruns of the authors' solver.
MFC's compressible formulation is unnecessary for this incompressible reference.
SU2 can provide a later independent cross-check but is not run here.
No turbulence model, compressibility correction, or smoothed lid is used.
Width and lid speed are one; depth D=H/W, nu=1/Re.
Top velocity is (1,0,0), other walls are stationary; front/back are empty (2-D).
The corner discontinuity is classical. Existing smoothed-lid PINNs are NOT
matched-reference comparisons until boundary conditions are reconciled.

## Matrix

Each row expands to three meshes, Nx=80,120,180, Ny=D*Nx, Nz=1.
The refinement factor is 1.5, with the same symmetric wall-grading family.

| Re | D | Case count |
|---|---|---:|
|100,400,1000|1|3|
|1000|1.1,1.15,1.2,1.25|4|
|1000|2.2,2.3,2.4,3.2|4|
|100,500,1000|5|3|
|500,1000|7|2|

16 physical cases / 48 mesh solves. The Re=5000 cases remain outside this
initial steady campaign and require a transient/stability assessment first.
This is the complete **agreed core matrix**, not every case in the reference.

## Execution and restart

First push these sources, retrieve the exact commit in a separate Unity checkout,
then submit task 0 with FLOWML_CFD_SMOKE=1. It tests mesh, dictionaries, MPI,
field writing and reconstruction, not physical convergence. Only after success
submit the 48-member production array (maximum four simultaneous jobs).
Each task uses eight CPU ranks, 16 GB, cpu-preempt, avx512, two-hour slots.
No simulation runs on the login node. Never overwrite existing reference files.

SIMPLE has a 20,000-iteration initial budget, not a convergence declaration.
Checkpoint writes occur every 500 iterations; the latest three are retained.
Before walltime the batch trap requests writeNow, waits for a clean solver exit,
and requeues, bounded to 20 restarts. Restart uses latestTime and existing
processor decomposition. Abrupt node loss during a write can require selecting
the preceding complete checkpoint manually; do not promise crash-proof writes.
Keep complete solver logs. A solver-finished-unvalidated marker means only exit
success, never that vortices or convergence have been validated.

## Required acceptance audits before PINN reference use

1. Inspect old CFD archives for raw fields, mesh, BCs and convergence metadata;
   reuse only compatible, auditable data. Do not label old plots as new evidence.
2. Verify steady iterative convergence using field changes, continuity and
   momentum residuals, not a prescribed iteration count. Track depth bands.
3. Compare vortex number/sign, centre and streamfunction strength across all
   three meshes. Report uncertainty and lack of asymptotic refinement honestly.
   Extend/refine if tiny deep vortices remain below numerical uncertainty.
4. Compute streamfunction with independently checked sign/scaling; global
   velocity error does not establish accuracy of very weak bottom vortices.
5. Cross-check square cavities against an established benchmark, then the paper's
   deep-cavity table/figures. Do not silently correct apparent paper typos.
6. Test transient evolution and time-step independence on representative deep
   cases before interpreting steady SIMPLE solutions as dynamically attained.
7. Preserve raw fields and provenance; publish validated results separately.
   No new DOI/release and no claim of journal-ready evidence at submission time.
