# Week 13 rectangular-cavity PINN evidence

These are retained outputs from four restartable Unity `gpu-preempt` jobs using
one NVIDIA A100, float64, 16,384 fixed collocation points, 1,000 Adam steps and
3,000 SSBroyden2 steps per case. The public runner verifies McDevitt's pinned
DeepPlasma source by commit and SHA-256; the upstream file is not vendored here.

| Re | D=H/L | independent momentum RMS | top-corner RMS | CFD u-centerline | CFD v-centerline | CFD vector field | interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 100 | 1 | 0.496 | 2.43 | 2.35% | 4.83% | 3.34% | all frozen near-matched CFD gates pass; corner residual remains large |
| 400 | 1 | 0.0475 | 0.237 | 8.28% | 10.13% | 10.57% | all frozen near-matched CFD gates pass |
| 100 | 2 | 0.370 | 1.92 | n/a | n/a | n/a | residual-audited hypothesis; no matched raw field |
| 400 | 2 | 0.131 | 0.671 | n/a | n/a | n/a | residual-audited hypothesis; no matched raw field |

All wall-velocity errors are zero to reported precision and continuity is at
floating-point scale because the velocity is differentiated from a hard-lifted
streamfunction. These structural facts do not certify momentum accuracy.

The square reference uses a classical lid whereas the PINN uses a smooth corner
regularization; it is a near-matched screening reference, not final publication
evidence. The deep cases have no raw CFD field at the identical geometry,
coordinate convention and smooth lid in the supplied archive. Cheng and Hung
are used only as a topology reference.

## Optimizer interpretation

The first pilot ended at 1,000 SSBroyden2 steps. After its residuals were
inspected, all four exact checkpoints—not selected seeds—were continued to
3,000 steps. The continuation generally reduced held-out residuals, but the
`Re=400, D=2` fixed held-out set deteriorated late even though a second 8,192-point
independent audit was lower. The `Re=400, D=1` CFD field error also increased
while its residual decreased. This non-monotonicity is retained rather than
replaced by a post-hoc attractive endpoint.

Consequently the matrix establishes feasibility and exposes a stopping-rule
problem. It does not establish that SSBroyden2 is universally superior, that the
deep fields are correct, or that streamfunction PINNs solve high-Re cavities.

## Provenance and files

- Initial array: `64062832`; continued array: `64064728`.
- Retained per-case job ids: `64064729` through `64064731`, with the fourth case
  under `64064728` as recorded in each audit.
- Upstream commit: `fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b`.
- Upstream SHA-256:
  `391a2174cb9f6e8863c14d7b350077e54209134de9f85a17719c398146f91458`.
- `audit.json`: independent residuals, walls, CFD gates and claim status.
- `optimizer-history.jsonl`: masked training, fixed held-out and corner history.
- `fields.png`, `loss.png`: geometry-faithful fields and optimizer trajectory.
- `environment-lock.txt`: exact Python environment.
- `manifest.json`: retained-file hashes. Checkpoints remain on Unity and are not
  published because they are large execution state, not needed by the audit notebook.

Run `python qa/build_week13_materials.py --publish-copy` to regenerate the
lecture PDF, audit notebook and overview figures from these immutable records.
