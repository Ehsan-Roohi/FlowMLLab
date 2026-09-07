# Week 1.1 scientific specification

## Change request

Implement and audit a two-dimensional velocity-field diagnostic that returns

\[
\nabla\cdot\mathbf{u}=\partial_x u+\partial_y v,
\qquad
\omega_z=\partial_x v-\partial_y u.
\]

The array contract is `field[y, x]`; coordinate vectors are one-dimensional,
strictly increasing, and may be nonuniform. The implementation must reject
nonfinite values and incompatible shapes rather than silently broadcasting.

## Scientific question

Can code proposed by a human or coding agent be accepted as a research
diagnostic based on evidence fixed before the final result is inspected?

## Evidence layers

1. **Analytic verification:** use the independently derived streamfunction
   `psi = sin(pi*x)^2 sin(pi*y)^2`. Verify vorticity against its exact
   derivative on four grids and recover second-order convergence.
2. **Physical audit:** evaluate the diagnostic on the accepted `Re=100`
   lid-driven-cavity field in `data/cavity_data.npz`. Check incompressibility,
   wall velocities, and agreement with the archived vorticity convention.
3. **Provenance:** bind the decision to the SHA-256 digest of the complete input
   archive and write every threshold and gate outcome to JSON.

## Frozen acceptance gates

| Gate | Acceptance threshold | Why it exists |
| --- | ---: | --- |
| Dataset identity | exact reference SHA-256 | fails closed if the input archive drifts |
| Observed vorticity order | `p >= 1.90` | distinguishes convergence from execution |
| Finest-grid vorticity relative L2 | `<= 5e-4` | bounds absolute discretization error |
| Manufactured divergence RMS | `<= 1e-12` | checks the analytic solenoidal identity |
| Cavity divergence RMS | `<= 1e-12` | checks the qualified field and axis convention |
| Cavity/archive vorticity relative L2 | `<= 4e-2` | allows the archive and diagnostic stencils to differ near walls |
| Wall-velocity maximum error | `<= 1e-12` | protects the lid/no-slip boundary contract |

The two outer grid layers are excluded from derivative comparisons. This is a
declared stencil-support decision, not post-hoc removal of unfavorable points.

## Human authority boundary

A coding agent may propose an implementation, tests, documentation, or a
refactor. The investigator remains responsible for the mathematical sign
convention, array axes, boundary interpretation, thresholds, reference data,
claim wording, and the final accept/reject decision. Passing software tests is
necessary but not sufficient for a scientific claim.

## Required disclosure

Every submission must state:

- tool/model and date used, or `No coding agent used`;
- the exact task specification supplied to the tool;
- files or functions proposed by the tool;
- material human corrections and why they were necessary;
- commands used for independent verification;
- known limitations and any failed gate retained in the report.

## Claim boundary

This laboratory qualifies one differential diagnostic on one analytic field
family and one accepted cavity case. It does not validate arbitrary grids,
unstructured-mesh operators, turbulent-flow diagnostics, the CFD solver as a
whole, or the correctness of an AI system.

## Independent-authoring and source note

The module is an original FlowMLLab exercise built from FlowMLLab-owned code and
the fixed cavity archive. A public course-topic survey motivated the generic
idea of specification-driven, human-agent development; no external assignment,
solution, slide, figure, code, or distinctive wording was incorporated.

Methodological background:

- Wilson et al., *Good Enough Practices in Scientific Computing*, PLOS
  Computational Biology 13(6), 2017. https://doi.org/10.1371/journal.pcbi.1005510
- Sandve et al., *Ten Simple Rules for Reproducible Computational Research*,
  PLOS Computational Biology 9(10), 2013. https://doi.org/10.1371/journal.pcbi.1003285
- Roache, *Verification and Validation in Computational Science and
  Engineering*, Hermosa Publishers, 1998.
- CS146S topic overview, Stanford University, consulted only at syllabus level:
  https://themodernsoftware.dev/
