# FlowMLLab v1.5.1

This author-requested release consolidates the course through Week 13:
33 notebooks, 19 PDF lectures, retained numerical evidence and reproducible
scientific-ML experiments. It preserves the historical v1.4.1 archive.

The v1.5.0 tag did not pass clean Linux qualification and was not published
as a GitHub/Zenodo release. Version 1.5.1 includes the missing lecture-test
dependency and the cross-platform numerical-regression correction below.

## Course and evidence additions

- Week 1.1: specification-first AI-assisted scientific software, analytic
  verification, physical acceptance gates and transparent development records.
- Weeks 5 and 7: sparse sensing, modal forecasting and frozen-case evaluation;
  Week 7.2: causal cylinder state estimation with explicit uncertainty limits.
- Week 4.2: PINN foundations and a retained Re=100 Unity cavity qualification.
- Week 10.1: classical scattering and collision-surrogate audit, with separately
  attributed historical cylinder contours and surface pressure/heating.
- Weeks 11 and 12: shock/vortex inference and DSMC moment reconstruction,
  synthetic controls, research provenance and fresh Noise2Noise-style fits on
  existing real DSMC observations.
- Week 13: continuous-text research lecture, CPU evidence-audit notebook,
  four retained float64 A100 runs at Re=100/400 and D=H/L=1/2,
  Adam-to-SSBroyden2 histories and restartable gpu-preempt submission.
- Recovered cylinder/nozzle provenance, dataset and preprocessing audits,
  expanded validation tools, and executed notebook previews.
- Week 13 homepage gallery: each complete case is displayed at full width,
  ordered by Reynolds number, with each aspect ratio on its own line.

## Scientific scope

Week 13 is a four-case, one-seed feasibility study. Square-cavity PINNs pass
the declared near-matched CFD screening gates; the CFD and PINN lid treatments
differ at the corners. Deep cavities have no matched raw reference fields.
Full-domain and corner residuals, late optimizer deterioration, and the
post-pilot extension from 1,000 to 3,000 SSBroyden2 steps remain explicit.

Research-data provenance, known nozzle exporter defects, uncertainty
under-coverage, interpolation baselines and unqualified SPARTA pilot tools are
retained with their limitations. Inclusion in a software release does not
establish a new paper benchmark or resolve a source-data defect. Dataset and
upstream-code permissions remain separate from the software MIT license.

## Verification and distribution

Week 1.1 diagnostic metrics allow floating-point roundoff across libm/BLAS
implementations (`rtol=1e-10`, `atol=1e-13`). Dataset hashes, thresholds,
metadata and acceptance decisions still match exactly, and physical gates
are independently recomputed with their original limits. Tests reject
material drift, nonfinite values and changes to the frozen contract.
No retained numerical data or acceptance record has been regenerated.

Publication is gated by the package tests, scientific smoke test, complete
course validator, notebook execution with retained-data hash checks, and
wheel/source-distribution validation. The release attaches the Python
distributions, Week 13 lecture, notebook and four full-width field figures.
The source archive includes the retained course tree and source code.

The version-specific Zenodo DOI is added to current citation metadata after
the GitHub-Zenodo integration confirms publication. The release family uses
concept DOI [10.5281/zenodo.22074169](https://doi.org/10.5281/zenodo.22074169).
The v1.4.1 DOI [10.5281/zenodo.22348207](https://doi.org/10.5281/zenodo.22348207)
continues to identify that historical archive.
