# FlowMLLab v1.9.0 — Week 16 reference validation

This release covers the physically checked teaching-body CFD, neural surrogate, retained design comparisons and Taylor–Maccoll verification. NASA validation is explicitly deferred and is not part of the accepted results.

Original NASA SEEB-ALR geometry, experimental records and LAVA reference remain available for study. The 8,000-iteration SU2 8.0.1 replay did not converge (density residual log10 −2.3811) and failed physical checks (41.24% stagnation-pressure error, 25.48% maximum total-enthalpy deviation). No NASA CFD agreement is claimed. Further NASA work is a separate research task.

The notebook independently integrates the Taylor-Maccoll cone equations and distinguishes experimental CFD validation from neural testing. The new SU2 8.0.1 cone study has 1.1616% finest-grid pressure error and 1.0064% last-two pressure change. Ten fresh design checks give 20.6646% peak-pressure reduction and 3.5218% pressure-drag reduction on the finer mesh.

All 44 original training/validation/test/extrapolation geometries have separately regenerated SU2 8.0.1 data and full-field physical checks. The new fixed-architecture model has its own portable checkpoint and fitting/audit logs. Its finer CFD waveform/peak/drag errors are 7.2036%/2.2373%/2.2740%, with 15.3716% worst-case waveform error. Extrapolation waveform error is 34.2862%; this limitation is retained. Historical SU2 8.5.0 data, failed physical checks and model identities remain available as an instructional audit trail.

A separate unchanged historical checkpoint has 7.1499% aggregate waveform, 2.6343% peak and 2.0615% drag error against eight new physically checked finer CFD references. Its worst waveform error is 15.9701%. These numbers do not describe the new clean-label fit. The preserved original 8.5.0 test fields failed the added enthalpy allowance despite tiny residuals.
Detailed teaching guides explain geometry units, pressure normalization, data splitting, POD/scaler fitting, checkpoint inference, mesh sensitivity and claim boundaries. Two independent agent reviews addressed the NASA numerical evidence and the portable-model evaluation.

## Scope and references

This is a reduced educational study, motivated by Zheng et al., *Aerospace Science and Technology* 178 (2026), 113218, DOI [10.1016/j.ast.2026.113218](https://doi.org/10.1016/j.ast.2026.113218), and the distinct earlier article DOI [10.7638/kqdlxxb-2025.0081](https://doi.org/10.7638/kqdlxxb-2025.0081).

It does not reproduce the authors' full-aircraft CAD, 3,480-case dataset, neural weights, atmospheric propagation or ground-level PLdB. Those claims require additional author data and implementations. NASA CFD agreement does not establish experimental validity of a neural network trained on the two-parameter teaching body. The recovered finer-CFD neural arrays support numerical comparisons but do not restore missing original solver logs.

NASA sources and hashes are retained in `cases/week16_lowboom/reference/source_manifest.json`. See the Week 16 lecture, NASA reference guide, model validation guide and executable audit reports for definitions and limitations.
