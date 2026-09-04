# Article-backed DSMC data for Week 10

This directory contains the complete numerical tables supplied by the authors
for the teaching reproduction of Roohi and Shoja-Sani, *Data-driven surrogate
modeling of DSMC solutions using deep neural networks*, Aerospace Science and
Technology 168 (2026) 110785, DOI
[`10.1016/j.ast.2025.110785`](https://doi.org/10.1016/j.ast.2025.110785).

Included raw material:

- fourteen 50 by 50 DSMC cavity fields: seven Knudsen numbers at each of the
  supplied lid speeds 10 and 30 m/s, including heat flux, shear stress, wall,
  and run metadata tables;
- six 300-point diatomic nitrogen shock profiles covering Mach 1.4--1.9;
- seven 300-point monatomic shock profiles covering Mach 1.4--2.0;
- four SPARTA cavity input decks; and
- the supplied SBT relaxation reference implementation.

Run `python qa/build_week10_aescte_dsmc_data.py` to rebuild compact NPZ files
and `python qa/run_week10_aescte_validation.py` to regenerate every retained
metric and figure. `results/aescte_dsmc/data_manifest.json` records a SHA-256
for every raw and derived file.

The attached cavity archive contains 10 and 30 m/s cases; it does not contain
the 100 m/s field mentioned in the paper. Likewise, no trained article
checkpoint or diatomic Mach-2 target was supplied, so these are not presented
as independently regenerated results.
