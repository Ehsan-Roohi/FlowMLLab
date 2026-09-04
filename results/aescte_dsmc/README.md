# Week 10 article-reproduction evidence

This directory contains deterministic, machine-readable evidence generated from
the committed DSMC tables for the Week 10 laboratory.

## Reproduce

```bash
python qa/build_week10_aescte_dsmc_data.py
python qa/run_week10_aescte_validation.py
```

The first command parses the raw tables, writes compact NumPy archives, and
records SHA-256 checksums in `data_manifest.json`. The second command rebuilds
all figures and metrics, then enforces the predeclared numerical gates.

## Retained gates

- primary cavity variables: every normalized RMSE below 2%;
- mono/diatomic shock profiles: every relative L2 error below 1.5%;
- Maxwell speed PDF: unit normalization to numerical precision.

`validation_summary.json` is the compact release-facing result.
`week10_validation_metrics.csv` retains every reported field and case. Higher
DSMC moments (`qx`, `qy`, `txy`) are included for diagnosis but are not hidden
inside the primary-variable average.

The supplied raw cavity data cover 10 and 30 m/s lid speeds, and the supplied
diatomic shock profiles cover Mach 1.4 through 1.9. Targets outside that
contract are reported as unavailable instead of being reconstructed from a
paper raster.
