# Controlled cavity sensitivity campaign

All candidates are unvalidated. Do not replace retained references automatically.
Use a separate absolute FLOWML_CFD_RUNS and export FLOWML_CFD_SENSITIVITY=1.
Submit unity_week13_cfd.sbatch with --array=0-5%2; its CPU-preempt, checkpoint,
MPI and restart settings are unchanged. Publish the code before Unity pulls it.

| Task | Re | H/W | Nx | Ny | Single changed factor |
|---|---|---|---|---|---|
| 0 | 100 | 5 | 270 | 1350 | Mesh refinement, original schemes |
| 1 | 500 | 5 | 270 | 1350 | Mesh refinement, original schemes |
| 2 | 100 | 5 | 180 | 900 | leastSquares gradient |
| 3 | 500 | 5 | 180 | 900 | leastSquares gradient |
| 4 | 100 | 5 | 180 | 900 | linearUpwind convection with original gradient |
| 5 | 500 | 5 | 180 | 900 | linearUpwind convection with original gradient |

Each starts from rest and targets 120000 total SIMPLE iterations. This is an
iteration budget, not an accuracy certificate. Preserve final-500 iteration
drift checks; a 270 mesh may need additional convergence time.

Compare each scheme variant against the corresponding original 180 mesh, and
refinement against original 120/180 meshes. Never pool variants as mesh levels
in audit_week13_cfd.py's automatic grid grouping; use isolated groups instead.
Report each vortex's signed streamfunction, centre and vorticity, local
reconstruction consistency and iteration drift. Global velocity L2 alone is
insufficient. No scheme is assumed more accurate a priori. Before attributing
paper discrepancies to the solver, independently audit streamfunction extraction
and exact benchmark conventions. Nektar++ is a possible independent method,
not yet an installed, submitted or verified comparison in this campaign.
