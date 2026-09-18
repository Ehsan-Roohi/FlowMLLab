# FlowMLLab v1.6.1

This patch release promotes the repository-wide teaching and reproducibility audit completed after v1.6.0.

## Highlights

- Improves definitions, notation, worked examples, and evidence interpretation across Weeks 1, 7.2, 9.3, 10.1, 11, 12, 14, and 15.
- Expands the Week 1 numerical-foundations lecture to a 27-page treatment of conservation laws, pressure elimination, streamfunction-vorticity coupling, finite differences, pressure recovery, stability, convergence, and validation.
- Rebuilds the Week 13 rectangular-cavity PINN notebook in teaching order with a real CPU training exercise and explicit separation of training loss, held-out residual, and CFD field error.
- Adds a staged Week 3 GPU cavity runner and repairs fresh-checkout paths and fallbacks in the beginner notebook sequence.
- Adds robust-scale Week 12 heat-flux figures while retaining the original full-range evidence.
- Removes private or upload-era context from public notebooks and documentation.
- Makes release QA index-driven, adds optional-dependency handling, and preserves failed scientific gates and limitations rather than relabelling them as successes.

## Release verification

- 40 indexed notebooks and 24 indexed lecture PDFs.
- Package tests, scientific smoke tests, release QA, notebook HTML reproduction, and GitHub Pages deployment pass on the release candidate.
- The audited update contains 72 changed paths relative to v1.6.0-era `main`, including rebuilt PDFs and notebooks whose bytes were verified before merge.

The version-specific Zenodo DOI for v1.6.0 remains unchanged. The all-versions DOI is [10.5281/zenodo.22074169](https://doi.org/10.5281/zenodo.22074169).
