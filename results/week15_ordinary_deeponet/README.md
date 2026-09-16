# Week 15 ordinary DeepONet OpenFOAM evidence

This directory contains a new, reproducible fixed-domain DeepONet baseline for
the Week 15 geometry-holdout experiment. It is not a recovered historical run.

- Dataset: the committed 130-case sampled OpenFOAM archive.
- Split: 107 training, 11 validation, and 12 test cases.
- Test geometries: `g009`, `g023`, `g036`, and `g048` at available Reynolds
  numbers 25, 50, and 100.
- Seeds: 17, 29, and 43; 400 epochs per seed.
- Branch input: Reynolds number only.
- Trunk input: normalized fixed-grid `x,y`.
- Deliberately excluded: mask, signed distance, and geometry identity.
- Checkpoint selection: lowest sampled validation MSE at ten-epoch audits.

Across the 36 seed/case evaluations, the three-seed, 12-case means are:

| Metric | Ordinary DeepONet |
|---|---:|
| velocity relative L2 | 28.96% |
| centered-pressure relative L2 | 299.57% |
| reverse-flow IoU | 0.406 |

For `g009_Re100_medium`, seed-wise velocity errors are 46.69%, 47.14%, and
46.61%. This repeatable failure is the intended fixed-domain baseline: cases
with the same Reynolds number but different masks receive identical operator
inputs before the output is masked for evaluation.

Each `ordinary-deeponet-seed*` directory stores its manifest, validation
history, case metrics, and all 12 raw test prediction bundles. The training
implementation is [`qa/train_week15_vanilla_deeponet.py`](../../qa/train_week15_vanilla_deeponet.py)
and the three-seed workflow is
[`week15-ordinary-deeponet.yml`](../../.github/workflows/week15-ordinary-deeponet.yml).
