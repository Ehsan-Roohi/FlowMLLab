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

## Vortex-aware sampling: relation to Roohi--Mahdavi and the negative result

This experiment deliberately tested whether the vortex emphasis used in the
Roohi--Mahdavi micro-step study transfers to the present architecture
comparison. In that paper, the reference recirculation zone is defined by the
streamwise-velocity condition $U_{\mathrm{DSMC}}<0$, and the objective balances
separately normalized errors inside and outside that zone. The V5
implementation expresses the same scientific idea through stratified sampling:
for every zonal arm, 60% of the 60,000 optimizer draws come from training cells
with $U_{\mathrm{DSMC}}<0$ and 40% from the remaining fluid cells. Uniform and
zonal arms otherwise use the same development geometries, validation
geometries, optimizer family, epoch budget and three seeds.

The reference-derived mask is used only to prioritize **training labels**. It
is never supplied as an inference input, and no H33/H58 validation velocity is
used to construct a model input. This distinction is essential: using a
target-derived vortex mask at inference would leak the answer and would not be
a deployable vortex predictor.

The controlled V5 result is mixed and does not support a general improvement
claim:

| Architecture | Uniform global % | Zonal global % | Change | Uniform $U<0$ % | Zonal $U<0$ % | Local interpretation |
|---|---:|---:|---:|---:|---:|---|
| MLP | 6.8611 | 10.2863 | +3.4252 points | 108.3983 | 44.3103 | large local reduction, but global degradation |
| ordinary DeepONet | 14.0063 | 23.3082 | +9.3019 points | 190.4409 | 93.4109 | 50.95% relative local reduction, but still 93.41% error |
| Geom-DeepONet | 10.5215 | 36.8317 | +26.3102 points | 104.3206 | 106.3438 | no local benefit and severe global degradation |

These are terminal, three-seed, equally weighted H33/H58 validation means.
They are not independent test scores. Most importantly, **zero of nine zonal
fits** (three architectures times three seeds) produced a checkpoint satisfying
the predeclared global-error ceiling. The apparent DeepONet vortex gain is
therefore not an accepted model improvement: it trades away too much of the
full field, remains inaccurate within the reverse-flow region, and fails the
selection gate on every seed. Geom-DeepONet shows that the intervention is not
even directionally reliable across architectures.

This outcome does not contradict the article result. The retained
Roohi--Mahdavi comparison changed the reported recirculation-zone error from
14.6135% to 11.9413% (an 18.28% relative reduction), while the full-domain error
changed only from 2.1739% to 2.2254%. V5 instead changes the sampling
distribution of three newly controlled models, uses H33/H58 as repeatedly
examined validation geometries, fixes $\alpha=0.6$ without an alpha sweep, and
does not reproduce the article architecture, training history or model state.
It is a transfer test of the idea, not a replication of the paper.

Several mechanisms can explain the limited transfer without being claimed as
proven causes. First, reverse-flow cells are a small and highly localized
fraction of the domain, so repeated zonal draws reduce coverage of the main
flow. Second, $U<0$ identifies reverse flow, not a vortex center, circulation,
Q-criterion region or reattachment location; reducing error on those cells need
not recover vortex topology. Third, Geom-DeepONet pools its sampled query
cloud, so changing the sampling distribution also changes its global geometry
context. Finally, validation geometries and training heights can differ in the
size and position of their recirculation pockets, making a fixed sampling ratio
an imperfect proxy for the target physics.

The defensible conclusion is consequently narrow: vortex-aware sampling can
redirect capacity toward reverse-flow cells, as the MLP and ordinary DeepONet
local errors demonstrate, but it did not produce an acceptable overall model
under the frozen V5 protocol. Future work should tune the zonal weight only on
unopened validation data, retain a global-error guard, and report reverse-flow
IoU, magnitude, component topology and reattachment diagnostics alongside the
full-field error. The relevant source study is Roohi and Mahdavi, *Analysis of
the rarefied flow at micro-step using a DeepONet surrogate model with a
physics-guided zonal loss function*, Microfluidics and Nanofluidics 30, 44
(2026), [doi:10.1007/s10404-026-02899-8](https://doi.org/10.1007/s10404-026-02899-8).

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
