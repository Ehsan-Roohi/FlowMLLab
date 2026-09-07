# Week 4.2 Unity qualification protocol

This protocol executes the lid-driven-cavity PINN from Christopher J. McDevitt's
DeepPlasma repository, used with his permission. The external source is not
redistributed: Unity checks out commit `fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b`
and verifies the LF checkout SHA-256
`391a2174cb9f6e8863c14d7b350077e54209134de9f85a17719c398146f91458`.

The first submitted job is a three-step GPU smoke test at Re=100. Its purpose is
to verify imports, CUDA float64 automatic differentiation, the dense
SSBroyden2 optimizer, plotting, and evidence capture. It is explicitly not a
scientific accuracy result. A long Re=5,000 job may be submitted only after the
smoke evidence has been inspected and a matching CFD reference and acceptance
thresholds have been frozen.

Unity jobs for this case are submitted to `gpu-preempt` with one A100. Because
the partition can preempt jobs, production runs must add restartable checkpoints
before their step count or wall time is increased. The runner now atomically
checkpoints the network, the dense inverse-Hessian state, fixed collocation set,
completed step, configuration, and CPU/GPU random-number states. A stable run ID
is reused after requeue, and an incompatible configuration is rejected.

The predeclared Re=100 qualification uses 16,384 fixed collocation points and
300 SSBroyden2 steps. It compares both centerlines and 8,192 independently drawn
interior points with the retained conventional CFD field. The thresholds fixed
before seeing the PINN result are 10% for the vertical-u centerline, 15% for the
horizontal-v centerline, and 15% for the interior velocity vector. This is a
near-matched comparison: the PINN regularizes the moving lid near both corners,
whereas the retained CFD case uses the classical discontinuous lid.

The independent audit reports unmasked momentum residuals, continuity, hard-wall
errors, environment versions, hardware, source identity, and job identity. A
contour is watermarked as unvalidated until comparison with the matched CFD
reference is complete.

The base interpreter is Unity's existing PyTorch 2.5.1+cu124 environment. The two
otherwise-missing imports are installed into a project-local target directory
from `qa/requirements-week42-unity.txt`; the shared environment is not modified.
The Slurm preflight imports the complete dependency chain used here, verifies
the pinned package versions and upstream APIs, and performs a float64 CUDA
operation before the training driver is called. This targeted gate is necessary
because Unity's shared environment contains unrelated packages whose metadata
conflict with each other; those packages are neither imported nor modified.
