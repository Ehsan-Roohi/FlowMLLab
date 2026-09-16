# V5: existing Unity GPU run, report-level audit

Imported on 2026-09-05 from the author's existing Unity run; **not a new run**.
Slurm job **64004321** completed on an NVIDIA A40, exit 0, elapsed 7:05.
The report records 415.204 seconds for the experiment, TensorFlow 2.20.0,
Keras 3.15.1, NumPy 2.2.6 and scikit-learn 1.6.1. There were 18 controlled fits
and two separate historical sklearn reproduction anchors.

This is development evidence on repeatedly used H33/H58 validation geometries,
not an independent test, a paper replication, or an accepted research benchmark.
The data are the existing author-supplied DSMC step archive described in
[data provenance](../../DATA_PROVENANCE.md), not data generated for this audit.
See the [frozen protocol](../../qa/step_architecture_v5.md) for inputs and controls.

## All arms, equal-weight validation means across three seeds

Errors are physical joint U/V relative L2 percentages. The vortex region is
reference U<0, not a Q-criterion label or a wall-shear reattachment measurement.

| Model | Sampler | Eligible seeds | Selected global | Selected vortex | Terminal global | Terminal vortex |
|---|---|---:|---:|---:|---:|---:|
| MLP | uniform | 3/3 | 7.2533 | 103.7377 | 6.8611 | 108.3983 |
| MLP | zonal | 0/3 | NA | NA | 10.2863 | 44.3103 |
| DeepONet | uniform | 0/3 | NA | NA | 14.0063 | 190.4409 |
| DeepONet | zonal | 0/3 | NA | NA | 23.3082 | 93.4109 |
| Geom-DeepONet | uniform | 3/3 | 6.1862 | 60.8503 | 10.5215 | 104.3206 |
| Geom-DeepONet | zonal | 0/3 | NA | NA | 36.8317 | 106.3438 |

[All 18 seed records](seed_metrics.csv) include rejected arms and terminal scores.
NA means no checkpoint met the fixed global-error ceiling; it does not mean
the training crashed. Geom-uniform improves the selected validation scores
relative to MLP-uniform, but its vortex error remains approximately 61%.
Its seed-691 terminal vortex error is 169%, so a terminal-epoch claim would be
misleading. No arm is promoted to the front page on these results.

For every seed/sampler group, the original report has identical drawn-row hashes,
batch-schedule hashes, target exposures and optimizer updates across the three
models. Each fit sees 5.4 million target rows including repeated draws, not
5.4 million independent DSMC samples. The two sklearn anchors reproduce their
historical global/vortex scores within 0.01 percentage points: uniform
5.4451/87.9639%, zonal 7.3209/33.8113%. These use a different optimizer/batching
protocol and are not extra controlled arms.

Geom's query-cloud context sensitivity is non-negligible: the largest reported
global/vortex shifts are 3.5037/13.1104% of the corresponding reference norm.
These shifts are prediction changes, not model-error scores. Additional training
alone does not resolve context dependence or establish generalization. A new
architecture experiment must declare its context and validation rules first.

## Provenance and verification boundary

- Original `report.json` SHA256:
  `d317067866d809a2f80ee540ec9384aab103c2277ec678cb3d669aef9cadab61`.
- Executed runner SHA256:
  `62d179fe42cc1831f17408ed3980eb4fbe0c80177b7ecf46715d49e3110070ef`;
  this matches `qa/step_architecture_v5.py` in Unity checkout `6f463ac`.
- Pinned helper commit: `2e9199bfa6efafd60c506374029c30aa3b4e009e`.
- Learning archive SHA256:
  `410907d46a040d53cbbd19fd8d44eeb7b41c05150f953fd4d9c6bb479da3479d`.
- All 13 V5 protocol tests passed on Unity/Linux during this audit. They use
  mocked submission and do not launch extra GPU jobs.

The CSV is a compact extraction of the actual JSON report read through Jupyter.
The full report, per-geometry predictions, training histories and weights remain
in the original Unity run. This import checks report consistency and Slurm
completion; it does **not** independently re-run checkpoint inference. Use the
runner's `bundle` command for a complete reproducibility archive before promoting
the result beyond development. No release, DOI or scientific superiority claim.
