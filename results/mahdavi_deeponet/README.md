# Roohi--Mahdavi DeepONet teaching evidence

This directory supports the two Week-9 research-to-classroom notebooks.

## What is directly reproducible here

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
  recirculation-zone ablation. The source micro-step DSMC fields and trained
  checkpoint are not public; the accompanying notebook's flow fields are
  explicitly manufactured for teaching.
- `nozzle_paper_field_errors.csv` and `nozzle_hard_case_baselines.csv`
  transcribe held-out and hard-case results from the nozzle manuscript. They
  are displayed as paper evidence, not recomputed by the compact notebook.
- The nozzle notebook recomputes the real DSMC centerline POD and fits a small
  centerline POD/branch teaching model. It is not the article's trained full
  two-dimensional six-output shock-aligned surrogate.

Every source file hash, derivation rule, article link, and claim boundary is in
`provenance.json`. Rebuild the compact archive from a checkout of the source
repository with:

```bash
python qa/build_week9_mahdavi_deeponet_data.py \
  --source /path/to/roohi-nozzle-pod-reproducibility --root .
```

The derived DSMC data remain CC BY 4.0; see `DATA_LICENSE.md`.
