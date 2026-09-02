# Four-frame multi-scale cylinder CNN

This directory contains the executed correction of the archived phase/POD wake
failure in `results/cylinder_ml/`.

## Frozen protocol

- development Reynolds cases: `60, 80, 90, 110, 120, 140`;
- complete-case validation: `Re=100`;
- untouched blind test: `Re=105`;
- four consecutive `u,v,p` fields plus Reynolds and the fluid mask predict the
  next `u,v,p` field;
- `snapshot_stride=25`, hence `dt*=25 U/D=0.1042`;
- three spatial convolution scales and a residual/persistence initialization;
- field, gradient-difference, vorticity-consistency, and divergence loss;
- exact no-slip velocity imposed with the analytical-circle fluid mask.

The CNN is trained on 64x64 patches but is fully convolutional and evaluated on
the complete 240x96 field. The LBM uses D2Q9 TRT and Bouzidi interpolated
bounce-back on the analytical circular wall.

## Executed result

| Complete case | Model | Vorticity relative L2 | Mean station profile L2 | Mean enstrophy error |
|---|---|---:|---:|---:|
| Re=100 validation | persistence | 10.64% | 16.21% | 0.516% |
| Re=100 validation | multi-scale CNN | 0.771% | 0.728% | 0.263% |
| Re=105 blind | persistence | 11.01% | 16.92% | 0.484% |
| Re=105 blind | multi-scale CNN | 0.815% | 0.804% | 0.285% |

At `Re=105` the CNN/LBM enstrophy ratios at `2D,4D,6D,8D` are
`1.0035, 0.9993, 1.0037, 1.0035`. Exact values, normalized transverse PSD
errors, field errors, divergence, seed, model size, and every validation gate
are stored in `multiscale_cnn_metrics.json`.

The video is a teacher-forced one-step forecast: each target uses four previous
true LBM fields. It is not an autonomous rollout, and the teaching-grid LBM is
not claimed to be grid-converged DNS.

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
from version control. The second command refuses to open `Re=105` unless all
predeclared `Re=100` validation gates pass.
