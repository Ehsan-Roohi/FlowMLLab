# Week 1.1 retained evidence

This directory records the executable scientific contract for the
vendor-neutral AI-assisted scientific-software module.

## Result

The reference Cartesian velocity diagnostic is **accepted** for the bounded
laboratory claim. It recovers observed vorticity order `p = 1.997`, reaches
`4.015e-4` relative L2 error on the 129 by 129 manufactured field, preserves
manufactured incompressibility to roundoff, and passes the independent checks
on the accepted `Re=100` FlowMLLab cavity field.

`acceptance_record.json` contains the exact input digest, thresholds, raw
metrics, individual gate decisions, and final decision. The file deliberately
contains no mutable timestamp. `week01_1_acceptance_summary.png` visualizes the
same retained values without becoming the numerical source of truth.

## Reproduce

```bash
python qa/build_week01_1_materials.py --execute
python -m unittest tests.test_scientific_software -v
```

The builder regenerates the JSON, figure, executed notebook, and lecture PDF.
The release validator recomputes the numerical contract rather than trusting
the committed decision string.

## Claim boundary

The result qualifies one second-order Cartesian diagnostic on one analytic
field family and one accepted 65 by 65 cavity case. It does not qualify
unstructured meshes, turbulent-flow features, arbitrary boundary stencils,
other solvers, or the correctness of a coding agent.

## Provenance and originality

The analytic field, code, exercise, figure, and narrative are original
FlowMLLab materials. The CFD input is the fixed FlowMLLab cavity archive whose
SHA-256 is stored in the JSON record. A public course-topic survey motivated
the generic subject of specification-driven development; no external
assignment, solution, slide, figure, code, or distinctive wording was reused.
