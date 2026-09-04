# FlowMLLab v1.3.0

FlowMLLab v1.3.0 adds an article-backed DSMC teaching and reproduction module,
and improves the full-field micro-nozzle assessment.

## New Week 10 DSMC material

- commits 85 checksummed source files covering 14 rarefied-cavity fields,
  six diatomic-shock profiles, seven monatomic-shock profiles, SPARTA inputs,
  run/wall metadata, and a relaxation script;
- adds `flowmllab.aescte_dsmc` for parsing, compact archives, log-Knudsen cavity
  synthesis, POD-branch shock operators, metrics, and Maxwell distributions;
- adds one complete Colab notebook and a 14-page rendered lecture derived from
  Roohi and Shoja-Sani, Aerospace Science and Technology 168 (2026) 110785;
- regenerates all cavity, shock, and equilibrium figures from numerical tables;
- retains a maximum primary cavity NRMSE of 1.281% and maximum shock-profile
  relative L2 error of 1.018%; and
- adds a dedicated deterministic GitHub Actions rebuild gate.

## Improved micro-nozzle evidence

- evaluates all six article outputs at the three held-out back pressures;
- combines local complete-field interpolation with a POD-trunk neural branch,
  with the wide-gap route selected on internal development folds;
- lowers the maximum held-out full-field error below 15%;
- generates separate 16, 25, and 30 kPa contour comparisons; and
- stores a machine-readable FlowMLLab-versus-article error table.

## Reproduce

```bash
python qa/build_week10_aescte_dsmc_data.py
python qa/run_week10_aescte_validation.py
python qa/run_nozzle_field_validation.py
python -m unittest discover -s tests -v
python qa/validate_course_release.py
```

The release contains the executable data, notebook, lecture PDF, generated
figures, CSV/JSON evidence, tests, and rebuild workflows.
