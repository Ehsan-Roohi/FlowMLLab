# Four-frame multi-scale cylinder CNN

This directory contains the executed correction of the archived phase/POD wake
failure in `results/cylinder_ml/`.

## Frozen protocol

- development Reynolds cases: `60, 80, 90, 110, 120, 140`;
- complete-case validation: `Re=100`;
- retained held-out interpolation: `Re=105` (historically frozen and now open;
  it is not a fresh blind case for future tuning);
- four consecutive `u,v,p` fields plus Reynolds and the fluid mask predict the
  next `u,v,p` field;
- `snapshot_stride=25`, hence `dt*=25 U/D=0.1042`;
- three spatial convolution scales and a residual/persistence initialization;
- field, gradient-difference, vorticity-consistency, and divergence loss;
- exact no-slip velocity imposed with the analytical-circle fluid mask.

The CNN is trained on 64x64 patches but is fully convolutional and evaluated on
the complete 240x96 field. The LBM uses D2Q9 TRT and Bouzidi interpolated
bounce-back on the analytical circular wall.

## Executed one-step result

| Complete case | Model | Vorticity relative L2 | Mean station profile L2 | Space-time spectral incoherence |
|---|---|---:|---:|---:|
| Re=100 validation | persistence | 10.638% | 16.206% | 1.49096% |
| Re=100 validation | cubic extrapolation | 0.923% | 1.357% | 0.00957% |
| Re=100 validation | multi-scale CNN | **0.771%** | **0.728%** | **0.00257%** |
| Re=105 retained interpolation | persistence | 11.009% | 16.924% | 1.60373% |
| Re=105 retained interpolation | cubic extrapolation | 1.053% | 1.581% | 0.01273% |
| Re=105 retained interpolation | multi-scale CNN | **0.815%** | **0.804%** | **0.00313%** |

At `Re=105` the CNN/LBM enstrophy ratios at `2D,4D,6D,8D` are
`1.0035, 0.9993, 1.0037, 1.0035`. The comparison is therefore not inflated by
using persistence alone: the CNN also beats the matched cubic extrapolation
baseline on complete-field vorticity, downstream profiles, and the complex
space-time spectrum. Exact field, divergence, enstrophy, spectrum, seed, model,
and gate values are stored in `multiscale_cnn_metrics.json`.

The video is a teacher-forced one-step forecast: each target uses four previous
true LBM fields. It is not an autonomous rollout, and the teaching-grid LBM is
not claimed to be grid-converged DNS.

## Executed autonomous-rollout audit

The same frozen CNN was also fed back recursively from four true initial
frames, over six starting phases. It passes the predeclared 15% mean-vorticity
gate through 10 steps, then fails it: the retained `Re=105` mean errors are
`0.826%, 4.225%, 8.814%, 20.222%, 35.369%, 57.328%, 90.724%` at horizons
`1, 5, 10, 20, 30, 40, 50`. This failure is retained deliberately; no rollout
claim is made from the visually accurate one-step movie. See
`re105_retained_rollout.png` and the `blind_rollout` block in the JSON record.

`re105_lbm_vs_multiscale_cnn.webp` is the looping, README-safe preview of the
retained full-resolution MP4. It is generated at 14 frames per second and
1400-pixel width so the repository front page animates without a click.

## Reproduce

```bash
python -m pip install -e '.[ml]'
python qa/run_cylinder_multiscale_cnn.py --workers 4
python qa/run_cylinder_multiscale_cnn.py --reuse-weights --run-blind
```

Dense LBM caches are generated under `results/cylinder_cnn/cache/` and excluded
from version control. The second command does not evaluate the retained
`Re=105` case unless all predeclared `Re=100` validation gates pass.
