# FlowMLLab v1.4.0

This release adds rarefied hypersonic-cylinder teaching material and restores a
complete, ordered visual course homepage. All weeks have separate rows, including
2.1, 4.1, 7 and 7.1; Weeks 5 and 6 retain their shared project pack with distinct goals.

## Rarefied cylinder: Week 7.1

- A notebook and lecture connect 20 author-supplied DSMC Mach cases to operator learning.
- The compact, checksummed dataset contains 44,500 finite teaching points.
- Whole-case train/validation/interpolation/extrapolation splits retain a strong
  field baseline and the lower accuracy of the CPU teaching operator.
- The homepage renders the original 400-by-400 Mach-8.5 fields, with a separate
  full-grid provenance record; compact teaching metrics are unchanged.

## Course gallery and research tools

- Restore the cavity animation, CFD and pressure validation, UQ, ROM, continuum
  and rarefied cylinders, gas dynamics, micro-step/nozzle, and DSMC shock figures.
- Place the three micro-step comparisons in one image row and keep the distinction
  between target-assisted article reconstruction and independent teaching inference.
- Add a clearly labelled synthetic Week-2 response figure from the notebook equation.
- Retain nozzle symmetry-boundary/registration diagnostics and controlled Unity
  micro-step MLP, DeepONet and Geom-DeepONet comparison runners added since v1.3.0.
- Detailed protocols and limitations remain in docs/RESULTS_GUIDE.md.

## Validation and scope

The publication workflow runs the package tests, scientific smoke test, and full
course release QA before creating the GitHub release. It builds and checks the
wheel and source distribution and attaches them to the release.

The source contains 25 notebooks and 12 lectures delivered as 11 PDFs. Numerical
validity limits and failed baselines are retained. The six private LJ audit ZIPs
are not included; this release does not claim ab initio Jäger-paper reproduction.

```bash
python -m pip install -e ".[test]"
python -m unittest discover -s tests -v
flowmllab smoke --root .
python qa/validate_course_release.py
```

The GitHub publication workflow passed, including scientific validation and
distribution checks. Zenodo archived version 1.4.0 with version-specific DOI
[10.5281/zenodo.22315623](https://doi.org/10.5281/zenodo.22315623), in the
FlowMLLab version family (concept DOI 10.5281/zenodo.22074169).
