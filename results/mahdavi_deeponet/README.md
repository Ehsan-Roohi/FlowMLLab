# Roohi--Mahdavi DeepONet teaching evidence

This directory supports the two Week-9 research-to-classroom notebooks.

## What is directly reproducible here

- `step_source_manifest.json` pins the micro-step source repository at
  commit `c3f211376b42b8dc30daad380eaef5e0ab800b5c`, records SHA-256 and row-count
  contracts for all nine smoothed height cases, freezes the 5/2/2
  development/validation/test split, and records the still-missing DSMC setup
  metadata.
- `step_height_learning_7cases.npz` and `step_height_test_2cases.npz` are
  compact full-parent-grid derivatives published with the corresponding
  author's permission. File-level separation prevents the notebook from
  opening H44/H67 before the final held-out gate. The QA cross-check proves
  exact source-row coordinates and exact U,V equality after the documented
  float32 conversion (maximum absolute quantization: `1.1432e-5` for U and
  `3.4546e-6` for V in source units).
- `step_privileged_input_audit.csv` records the independent audit showing that
  the published-repository inference path constructs local input patches from
  the held-out DSMC `U,V` field. Lab 1 therefore uses a separate leakage-free
  coordinate surrogate whose inputs are only `h/H`, `(x,y)`, and known geometry.
- `step_teaching_selection.csv`, `step_teaching_test_metrics.csv`, and
  `step_teaching_protocol.json` record the new leakage-free classroom run. The
  validation-only rule selects `alpha=0.6`; neither test archive nor a
  target-derived feature participates in selection.
- `step_article_contour_metrics.csv`, `step_article_case_coverage.csv`, and
  `step_article_contours/` use the final published paper's numbering and
  reconstruct the exact stored DSMC/NN comparisons available for Figure 6
  (`Kn=0.004`, `Kn=0.02`) and Figure 15 (H44, H67). The exact Figure-6 `Kn=1`
  DSMC contour is also rebuilt; its neural counterpart is explicitly marked
  unavailable because the pinned repository has no stored prediction. DSMC
  and NN share color limits, the solid step remains masked, and axes preserve
  the data-domain `L/H=5` aspect ratio. Stored NN fields are retained
  privileged-input article results, not an autonomous surrogate claim.
- `step_leakage_free_contour_metrics.csv` and
  `step_leakage_free_contours/` provide the corresponding independently
  held-out H44/H67 visual validation for the classroom coordinate model.
  Their generator fits only the seven-file learning archive, freezes the
  validation-selected setting, and opens the separate two-case test archive
  afterward.
- `nozzle_centerline_15cases.npz` is a compact derivative of all 15 public
  DSMC Tecplot snapshots in
  [`Ehsan-Roohi/roohi-nozzle-pod-reproducibility`](https://github.com/Ehsan-Roohi/roohi-nozzle-pod-reproducibility),
  pinned at commit `e1b234ba499408d3b6224633972f939f3b2301d6`.
- The archive retains the max-y symmetry centerline, 101 streamwise stations,
  seven physical fields, and independently recomputed density-shock
  diagnostics. It is only 64 KiB, so the classroom path does not download the
  15 full Tecplot files.
- `nozzle_pod_reference.csv` records the published/repository 15-snapshot
  density-POD audit. The FlowMLLab release gate recomputes those values from
  the compact archive.

## What is retained article evidence

- `step_paper_evidence.csv` transcribes the reported full-domain versus
  recirculation-zone ablation. These values remain separate from the new
  notebook-generated leakage-free teaching results and from the upstream
  privileged-input stored predictions.
- `nozzle_paper_field_errors.csv` and `nozzle_hard_case_baselines.csv`
  transcribe held-out and hard-case results from the nozzle manuscript. They
  are displayed as paper evidence, not recomputed by the compact notebook.
- The nozzle notebook recomputes the real DSMC centerline POD and fits a small
  centerline POD/branch teaching model. It is not the article's trained full
  two-dimensional six-output shock-aligned surrogate.

Every source file hash, derivation rule, article link, and claim boundary is in
`provenance.json` and `step_source_manifest.json`. Validate a checkout of the
step repository with:

```bash
flowmllab mahdavi --root . \
  --step-source /path/to/roohi-step-dnn-mahdavi \
  --require-step-data
```

Rebuild the compact nozzle archive from a checkout of its source repository
with:

```bash
python qa/build_week9_mahdavi_deeponet_data.py \
  --source /path/to/roohi-nozzle-pod-reproducibility --root .
```

Rebuild the compact step archives from the pinned source checkout with:

```bash
python qa/build_step_height_archive.py \
  --source /path/to/roohi-step-dnn-mahdavi --root .
```

Reproduce the teaching selection and held-out metrics with:

```bash
python qa/run_step_height_teaching_validation.py --root .
```

Rebuild the article-evidence and leakage-free contour sets with:

```bash
python qa/build_step_article_contours.py \
  --source /path/to/roohi-step-dnn-mahdavi --root .
python qa/build_step_leakage_free_contours.py --root .
```

The final article text contains one case-label inconsistency: its problem
statement lists `Kn=0.2`, while Figure 6, the figure discussion, the pinned
source code, and the stored prediction use `Kn=0.02`. The contour
reconstruction follows the figure and repository artifact. Figure 6 also shows
`Kn=1`; the exact DSMC field is rebuilt, but no NN field is fabricated because
the pinned repository does not contain the corresponding stored prediction.

The recorded test result is deliberately reported as a tradeoff, not a single
accuracy claim: the selected zonal model lowers vortex relative L2 from
50.745% to 19.863% at H44 and from 83.038% to 33.257% at H67, while global
relative L2 rises from 7.228% to 9.594% and from 5.934% to 11.267%,
respectively.

The nozzle derivative remains CC BY 4.0. The step derivatives have a separate,
author-specific publication permission; see `DATA_LICENSE.md`.
