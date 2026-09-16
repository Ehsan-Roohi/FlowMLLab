# Week 4.2: 51 × 51 refinement

The 184 retained cavity cases were solved again with 51 nodes in each
direction. The same-case Stokes fields were also recomputed at 51 × 51. No
25-node field was upsampled for training or evaluation. The previously selected
POD rank-24, width-96 neural correction was retrained with three seeds.

The original test cases had already been inspected. These remain regression
tests, not a new blind benchmark. The steady vorticity-equation RHS is below
`2e-5` for every accepted 51-node reference case.

## Mean velocity relative L2 error

| Train → test | 25 × 25 | 51 × 51 |
| --- | ---: | ---: |
| Constant → constant | 0.107% | 0.097% |
| Constant → diverse | 19.920% | 16.555% |
| Diverse → constant | 2.044% | 1.960% |
| Diverse → diverse | 0.917% | 0.810% |
| Diverse → shape OOD | 8.428% | 7.878% |
| Diverse → Reynolds OOD | 10.204% | 6.425% |

The 25-node and 51-node errors each use their own numerical Navier–Stokes
reference. `grid_shift.csv` compares the two CFD references by sampling the
51-node velocity at the 25-node positions. The mean relative shift is 20.59%
for constant-lid tests and 17.81% for diverse-lid tests. A single refinement
step does not establish mesh convergence; smaller surrogate error does not
establish more accurate physical flow.

On all six same-family tests for each lid type, the predicted lower-right
positive streamfunction maximum is on the same 51-node grid point as the
reference. The primary-vortex node matches in 5/6 constant and 6/6 diverse
cases. These are grid-resolved recirculation measurements, not independent
validation of a secondary vortex.

## Files

- `dataset.npz`: newly solved 51-node Navier–Stokes fields and residuals.
- `stokes.npz`: matched 51-node Stokes fields and linear-solve diagnostics.
- `cases.json`: unchanged case manifest from the 25-node expanded study.
- `*_seed*.pt`: selected retrained models; `*_history.csv`: training histories.
- `prediction_*.npz`: retained predictions for each train/test family pair.
- `metrics.csv`, `summary.csv`, `grid_comparison.csv`, `grid_shift.csv`,
  `vortex_metrics.csv`: numerical evaluation.
- `solver_audit.json`: case count and maximum accepted steady RHS.

Regenerate with `python qa/run_week04_2_grid51.py generate`, then
`python qa/run_week04_2_grid51.py train`, then
`python qa/build_week04_2_grid51.py`. The companion figures are
[`Cavity_constant_grid51_streamlines_vorticity.png`](../../figures/Cavity_constant_grid51_streamlines_vorticity.png)
and [`Cavity_diverse_grid51_streamlines_vorticity.png`](../../figures/Cavity_diverse_grid51_streamlines_vorticity.png).
