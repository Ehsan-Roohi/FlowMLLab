# FlowMLLab v1.9.0 — Week 16 reference validation

Release candidate. Publication is conditional on the numerical and teaching-material gates; see `PUBLICATION_STATUS.md`.

Week 16 adds the original NASA SEEB-ALR STEP geometry, wind-tunnel pressure records and coordinate macros, plus the NASA-hosted LAVA comparison. The Gmsh/SU2 driver creates an axisymmetric Euler calculation at Mach 1.6 and retains exact configuration, mesh, solution, residual history and provenance hashes for three mesh levels. Comparison uses source-prescribed coordinates and a fixed window, with no fitted alignment or pressure rescaling.

The notebook independently integrates the Taylor-Maccoll cone equations, separates experimental CFD validation from neural testing, and loads a portable neural checkpoint without retraining. Its retrospective finer-CFD evaluation has 6.39% aggregate waveform error, 1.93% mean peak error and 1.57% mean pressure-drag error. The worst finer-case waveform error is 14.63%; extrapolation errors remain substantially larger. A distinct recovered historical prediction snapshot has 7.15% aggregate finer-CFD waveform error. These are different fitted models and are labelled separately.

Detailed teaching guides explain geometry units, pressure normalization, data splitting, POD/scaler fitting, checkpoint inference, mesh sensitivity and claim boundaries. Two independent agent reviews addressed the NASA numerical evidence and the portable-model evaluation.

## Scope and references

This is a reduced educational study, motivated by Zheng et al., *Aerospace Science and Technology* 178 (2026), 113218, DOI [10.1016/j.ast.2026.113218](https://doi.org/10.1016/j.ast.2026.113218), and the distinct earlier article DOI [10.7638/kqdlxxb-2025.0081](https://doi.org/10.7638/kqdlxxb-2025.0081).

It does not reproduce the authors' full-aircraft CAD, 3,480-case dataset, neural weights, atmospheric propagation or ground-level PLdB. Those claims require additional author data and implementations. NASA CFD agreement does not establish experimental validity of a neural network trained on the two-parameter teaching body. The recovered finer-CFD neural arrays support numerical comparisons but do not restore missing original solver logs.

NASA sources and hashes are retained in `cases/week16_lowboom/reference/source_manifest.json`. See the Week 16 lecture, NASA reference guide, model validation guide and executable audit reports for definitions and limitations.
