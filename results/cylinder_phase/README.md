# Phase-stable cylinder-wake surrogate

This directory records the long-horizon correction to the archived recursive
CNN failure. The model is a phase-stable learned Fourier decoder; it is not
presented as an autoregressive CNN.

## Frozen protocol

- development cases: `Re = 90, 110, 120, 140`
- complete-case validation: `Re = 100`
- fresh test opened once after selection: `Re = 95`
- retained historical test: `Re = 105`
- initialization: four consecutive true fields
- rollout: 277 future fields with zero future CFD inputs
- selection: harmonic order chosen only by validation global vorticity error
- gates: global and worst-frame vorticity error below 15%; Strouhal error below 2%

| Split | Re | Global vorticity error | Worst frame | Last frame | Strouhal error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 100 | 6.037% | 7.959% | 7.959% | 0.343% |
| Fresh test | 95 | 4.281% | 5.194% | 5.194% | 0.264% |
| Retained test | 105 | 4.625% | 6.240% | 6.155% | 0.253% |

Exact unrounded values and the frozen split are in
`phase_stable_metrics.json`. The CFD inputs are checksummed and published in
the [`cylinder-cfd-v1` GitHub release](https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1).

Regenerate after downloading the release data into `data/cylinder_cfd/`:

```bash
python qa/validate_cylinder_cfd_dataset.py
python qa/run_cylinder_phase_stable.py
```

The summary figure reports the rollout gates, LBM-versus-decoder Strouhal
values, and the fresh-test wake spectrum. The WebP is the autoplay README demo;
the MP4 is the full-resolution version.
