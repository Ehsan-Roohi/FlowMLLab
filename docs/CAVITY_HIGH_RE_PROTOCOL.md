# New high-Reynolds square-cavity CFD protocol

This campaign computes new FlowMLLab CFD fields at Re = 3200, 5000, 7500,
10000. Ghia's samples are read **only after** the Navier--Stokes calculation.
It supplements, and does not replace, the retained Re = 100, 400, 1000 runs.

- Unit square, lid speed 1, no slip on the other walls, discontinuous lid at
  the two top corners. This is the standard two-dimensional Ghia geometry.
- Uniform meshes: 128, 256, 512 cells per side (129, 257, 513 nodes).
- Central second-order streamfunction/vorticity differences and Thom wall
  vorticity: the same spatial equations as `common/w4utils.py`. No upwind
  artificial diffusion, smoothing, reference fitting or neural inference.
- New damped Newton solution of the **steady** discrete equations; analytical
  Jacobian checked against finite differences and an independent grid stencil.
  Continuation starts from the retained Re=1000 CFD field. The step in Re is
  at most 500 and is halved if a nonlinear solve fails. Intermediate values
  initialize the solver; they are not additional Ghia validation cases.
- Nonlinear target: max absolute Poisson and vorticity residual < 1e-8.
  Independent acceptance: both equation defects < 1e-6, finite fields, wall
  conditions and discrete continuity checked from the saved arrays.
- Mesh sensitivity: all three native centreline profiles, plus differences
  of medium/coarse profiles from the finest one on a shared evaluation grid.
  A small equation residual is not a proof of mesh convergence.
- There is no time step in the Newton solve. A separate SSPRK3 physical-time
  audit compares matched trajectories on 256 cells at dt = 0.002 and 0.001, from tU/L = 0 to 10, sampled every 0.5. Both start from the identical perturbation delta psi = 1e-4 sin(pi x)^2 sin(pi y)^2. It is
  a time-discretization check, not proof of asymptotic physical stability.
- A converged steady 2-D branch must not be advertised as a stable 3-D physical
  flow. High-Re cavity steady solutions and physical stability are distinct
  questions ([Erturk et al. 2005](https://doi.org/10.1002/fld.953),
  [Erturk 2009](https://doi.org/10.1002/fld.1887)).
- Report both complete published-point scores and the previously declared
  flagged-point sensitivity from `flowmllab/ghia.py`. Keep the anomalous printed
  values visible; do not replace them or claim an author-issued erratum.
- Pressure is not reconstructed in this extension. Ghia centreline velocity
  comparisons do not validate pressure, nor do they qualify PINN predictions.
- Three existing numerical tests pass locally: analytical Jacobian, independent
  spatial stencil/wall terms, and stationary/physical-time RHS consistency.

The first direct jump from the retained Re=1000 field to Re=3200 on 128 cells
failed its Newton line search (vorticity residual 12.836). This preflight failure
motivated continuation; it is not presented as a valid CFD solution.

## Reproducibility and storage

Live CPU calculation: Unity array 1767993, one independent task per mesh; temporal array 1768145 depends on completion of its 256-cell task,
partitions cpu,cpu-preempt, no artificial array concurrency limit.
Workspace allocated with ws_allocate: `flowml-ghia-highre-20261115`.
Actual returned path: `/scratch3/workspace/roohie_umass_edu-flowml-ghia-highre-20261115`.
Expiry checked: 2026-12-15 14:04:59. Only one current, atomically verified complete
continuation checkpoint per mesh is kept. Successful target fields and source
hashes are preserved. No unrelated jobs or workspaces are modified.

The source and Re=1000 seed were SHA256 verified after upload. The pinned CPU
environment is Python 3.12, NumPy 2.2.6 and SciPy 1.15.3. At submission the new
workspace occupied 152 MB. Shared filesystem free space was 62 TB; this is not
an assertion about personal quota (no quota command was available).

Verified final fields and provenance are to be retained on Lexar D: in the
existing FlowMLLab repository, under `results/cavity_highre_ghia`. Do not publish
unverified partial snapshots as final results. Existing frozen neural fits and
their disclosed train/validation/test roles are unchanged.

The finite-horizon paired-time check requires a velocity RMS difference below
1e-6 U and below 1 percent of the final perturbation RMS. Report the measured
values even if a check fails. This criterion does not assert long-time stability.

## Completed campaign and retained evidence

All tasks completed: high-Re array 1767993, temporal array 1768145,
matched low-Re array 1768918, training-support array 1769311, and low-Re
validation-driven support array 1771973. The last array adds Re=500 and 750
after validation identified a POD representation floor; held-out neural test
cases remain Re=3200 and 7500. These extra PDE solves seed from Re=1000,
without fitting benchmark or held-out neural targets.

The final matrix contains 15 Reynolds cases on three grids, 45 native fields.
All final hashes match on Lexar and all independent discrete-equation checks
pass. This is not a mesh-independence certificate: native 256-to-512 vector
differences increase from 2.03% at Re3200 to 6.32% at Re10000. The complete
tables retain these differences and the corner maxima. Neural predictions
therefore emulate the documented 512-cell discretization.

Three fine-grid intermediate roots (6000,6500,9000) were captured only after
their continuation steps completed, by reading atomic checkpoint snapshots.
Conversion verifies the captured SHA256 and rechecks the native equations.
No live checkpoint was edited or moved. The remaining final fields were
downloaded after their producing jobs had completed. Checkpoint/recovery
data remain on the allocated scratch workspace; valuable final fields and
complete provenance are retained on the external project drive and in the
authorized software release.
