# Week 13 rectangular-cavity PINN qualification protocol

This protocol separates a reproducible research experiment from a classroom
demonstration.  It adapts the permitted DeepPlasma lid-driven-cavity network and
SSBroyden2 optimizer at pinned upstream commit
`fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b`.  The upstream file is verified by
SHA-256 before execution; it is not vendored or silently modified.

## Initial matrix

| Case | Reynolds number | Depth/width | What can be qualified now |
|---|---:|---:|---|
| S100 | 100 | 1 | independent residuals, walls, and retained CFD field |
| S400 | 400 | 1 | independent residuals, walls, and retained CFD field |
| D100 | 100 | 2 | independent residuals, walls, and vortex topology |
| D400 | 400 | 2 | independent residuals, walls, and vortex topology |

The depth-two cases are **not** labelled field-validated unless a citable raw CFD
field on the same geometry, coordinates and nondimensionalization is added.  The
Cheng--Hung rectangular-cavity benchmark is used to audit vortex topology, not
to manufacture a pointwise error metric from a published figure.

## Optimizer and restart design

Each case uses float64 on an A100 in `gpu-preempt`.  Adam provides a fixed
warm-up and SSBroyden2 continues from those weights.  This is a staged optimizer
trajectory, not a matched claim that one optimizer is universally superior.
The runner records training residuals, held-out unmasked residuals, continuity,
and a separate top-corner residual.  A falling collocation loss is therefore
insufficient by itself.

SLURM sends `SIGUSR1` ten minutes before the two-hour limit.  The process writes
an atomic checkpoint containing model, optimizer, collocation points and random
states; the array job then requeues itself and resumes from the same state.

## Frozen interpretation gates

For square cases the retained near-matched CFD comparison uses:

- relative L2 error of `u(0.5,y)` no greater than 10%;
- relative L2 error of `v(x,0.5)` no greater than 15%;
- relative L2 vector error at 8,192 seeded interior points no greater than 15%.

Hard-wall errors, full held-out momentum residuals and top-corner residuals are
reported without deleting inconvenient values.  A deep case is research-ready
only when residual convergence, exact wall enforcement, stable held-out trends,
and the expected vortex topology agree.  Every published result must also retain
the job id, GPU model, dependency lock, upstream commit and upstream digest.

## Unity submission

Run from a clean clone of the exact FlowMLLab commit and provide four absolute
environment paths:

```bash
export FLOWML_PINN_PYTHON=/path/to/python
export FLOWML_PINN_DEPS=/path/to/site-packages
export DEEPPLASMA_LDC=/path/to/LDC_module_square.py
export FLOWML_PINN_REFERENCE=/path/to/cavity_data.npz
sbatch qa/unity_week13_pinn_matrix.sbatch
```

The array writes only to `tmp/week13-pinn-*`.  Results enter the retained course
evidence only after the audit gates have been checked case by case.
