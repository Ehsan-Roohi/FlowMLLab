# Week 15 capacity and GeoPT experiment — in progress

The existing accepted figures/results remain the baseline. No conclusion about
improvement is available until these runs finish and saved checkpoints are
independently reloaded. This directory is not a completed benchmark.

## Frozen first experiment

Compare the existing 3-block/64-width Transolver with:

1. 8 blocks, width 256, 8 heads, 32 slices, MLP ratio 2, random initialization.
2. Exactly the same model initialized using the official GeoPT checkpoint.

100 training / 8 validation / 19 retrospective test cases, with g005 quarantined.
19,200 updates per run, 1,024 uniformly selected fluid points per update,
train-only scaling, normalized u/v/pressure MSE, AdamW/cosine schedule.
Checkpoints are selected only on full-cloud validation; no intermediate test
scores are used to stop or tune the run. Seed 17 runs are currently launched;
seeds 29 and 43 and the other architectures are not yet completed.

GeoPT source: https://github.com/Physics-Scaling/GeoPT
Checkpoint: https://huggingface.co/GeoPT/GeoPT_Pretrained_Models/tree/45df8bd3d02f9ab36a25d86d0a048e9d72572fd3
Checkpoint SHA256: c0b1b9c4e5d533dbc249190d3d1fbe8e6b066b36d325cf4377cc0d613c02d1c2

The adapted network has 3,859,011 parameters; 3,855,680 parameters are loaded
through the upstream shape-filtering policy. The incompatible first projection
and the task-specific output normalization/head are initialized for this task.
The original 3D input/dynamics prompt is not reproduced by this 2D transfer.
Thus this is an explicit backbone-transfer experiment, not a reproduction of
the nozzle paper's implementation or its full pretrained-input protocol.

## Other architectures

GeoPT weights cannot be assigned to DeepONet, Geom-DeepONet, Geo-FNO, SMART,
DoMINO, or proprietary-inspired proxies. Their larger configurations require
architecture-specific depth/width changes, parameter counts and compute records.
The historical, LR-tuned and matched-budget rows are variants of some of the
same architectures; they are not ten independent architectures.

The archived model adapters are being recovered to preserve their original
architectures. No enlarged result for those architectures is claimed here.
A full comparison must retain the existing split, use validation-only selection,
report seed variability and evaluate velocity, centered pressure and filtered
reverse-flow IoU across all 19 test cases rather than only g049/Re=100.

## Commands

```bash
python qa/download_geopt_checkpoint.py --output /tmp/geopt_weights
python qa/train_family_transolver_large.py --seed 17 --threads 2 \
  --output results/week15_capacity/transolver_8x256_scratch
python qa/train_family_transolver_large.py --seed 17 --threads 2 \
  --pretrained /tmp/geopt_weights/GeoPT_8layers.pt \
  --output results/week15_capacity/transolver_8x256_geopt
```

CUDA is selected automatically when available. CPU smoke benchmarks verified
forward/backward operation and pretrained loading. Training is unfinished;
no test results or updated scientific comparison figures have been released.
