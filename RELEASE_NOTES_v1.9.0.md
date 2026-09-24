# FlowMLLab v1.9.0 — Week 16 reference validation

Release candidate. Publication is conditional on the numerical and teaching-material gates; see `PUBLICATION_STATUS.md`.

Week 16 adds the original NASA SEEB-ALR STEP geometry, wind-tunnel pressure records and coordinate macros, plus the NASA-hosted LAVA comparison. The Gmsh/SU2 driver creates an axisymmetric Euler calculation at Mach 1.6 and retains exact configuration, mesh, solution, residual history and provenance hashes for three mesh levels. Comparison uses source-prescribed coordinates and a fixed window, with no fitted alignment or pressure rescaling.

The notebook independently integrates the Taylor-Maccoll cone equations and distinguishes experimental CFD validation from neural testing. The new SU2 8.0.1 cone study has 1.1616% finest-grid pressure error and 1.0064% last-two pressure change. Ten fresh design checks give 20.6646% peak-pressure reduction and 3.5218% pressure-drag reduction on the finer mesh.

All 44 original training/validation/test/extrapolation geometries have separately regenerated SU2 8.0.1 data and full-field physical checks. The new fixed-architecture model has its own portable checkpoint and fitting/audit logs. Its finer CFD waveform/peak/drag errors are 7.2036%/2.2373%/2.2740%, with 15.3716% worst-case waveform error. Extrapolation waveform error is 34.2862%; this limitation is retained. Historical SU2 8.5.0 data, failed physical checks and model identities remain available as an instructional audit trail.

A separate unchanged historical checkpoint has 7.1499% aggregate waveform, 2.6343% peak and 2.0615% drag error against eight new physically checked finer CFD references. Its worst waveform error is 15.9701%. These numbers do not describe the new clean-label fit. The preserved original 8.5.0 test fields failed the added enthalpy allowance despite tiny residuals.
Detailed teaching guides explain geometry units, pressure normalization, data splitting, POD/scaler fitting, checkpoint inference, mesh sensitivity and claim boundaries. Two independent agent reviews addressed the NASA numerical evidence and the portable-model evaluation.

## Scope and references

This is a reduced educational study, motivated by Zheng et al., *Aerospace Science and Technology* 178 (2026), 113218, DOI [10.1016/j.ast.2026.113218](https://doi.org/10.1016/j.ast.2026.113218), and the distinct earlier article DOI [10.7638/kqdlxxb-2025.0081](https://doi.org/10.7638/kqdlxxb-2025.0081).

It does not reproduce the authors' full-aircraft CAD, 3,480-case dataset, neural weights, atmospheric propagation or ground-level PLdB. Those claims require additional author data and implementations. NASA CFD agreement does not establish experimental validity of a neural network trained on the two-parameter teaching body. The recovered finer-CFD neural arrays support numerical comparisons but do not restore missing original solver logs.

NASA sources and hashes are retained in `cases/week16_lowboom/reference/source_manifest.json`. See the Week 16 lecture, NASA reference guide, model validation guide and executable audit reports for definitions and limitations.
