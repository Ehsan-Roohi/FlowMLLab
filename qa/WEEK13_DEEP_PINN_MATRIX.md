# Deep-cavity PINN matrix

This restartable A100 campaign matches the available CFD geometries rather than
extrapolating conclusions from the earlier square/depth-two pilot.

| Task | Reynolds number | Depth/width | Initial seed |
|---:|---:|---:|---:|
| 0 | 100 | 5 | 1234 |
| 1 | 500 | 5 | 1234 |
| 2 | 1000 | 5 | 1234 |
| 3 | 500 | 7 | 1234 |
| 4 | 1000 | 7 | 1234 |

Each task uses float64, 32,768 fixed collocation points, 5,000 Adam steps and
10,000 SSBroyden2 steps.  SIGUSR1 triggers an atomic checkpoint ten minutes
before the two-hour `gpu-preempt` limit, after which Slurm requeues the same
task and the runner resumes model, optimizer, points and random states.

This is a one-seed feasibility stage.  A journal claim requires independent
seeds and matched pointwise comparison against the retained Nektar++ fields.
The existing runner records unmasked held-out residuals, corner residuals,
exact hard-wall errors, optimizer histories and equal-aspect field figures.
