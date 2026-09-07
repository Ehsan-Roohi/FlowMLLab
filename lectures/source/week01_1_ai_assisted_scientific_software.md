# Week 1.1 - AI-assisted scientific software

## Lecture purpose

This original FlowMLLab lecture introduces specification, mathematical
verification, physical acceptance gates, content-addressed provenance, and
accountable human review for code proposed by either a person or a coding
agent. The authoritative rendered lecture is
`../week01_1_ai_assisted_scientific_software.pdf`; its deterministic builder is
`../../qa/build_week01_1_materials.py`.

## Slide sequence

1. Scientific question and retained cavity velocity field.
2. Execution, mathematical verification, physical validation, and reproducibility.
3. Explicit scientific specification and array contract.
4. Differential operators, sign convention, and second-order interior stencil.
5. Streamfunction construction and analytic vorticity.
6. Four-grid convergence plot and observed order.
7. Real cavity vorticity and interpretation of the archive audit.
8. All seven measured values alongside their frozen acceptance thresholds.
9. A plausible axis-swap failure that executes but is rejected.
10. Concrete implementation brief and human review workflow.
11. Input identity, execution context, and reproducible decision evidence.
12. Evidence-based assessment rubric and discussion question.
13. Bounded conclusion and references.

## Visual design and build

The 13-page landscape lecture uses a consistent typographic hierarchy,
typeset mathematical expressions, full-size scientific figures, and open
tables. The lecture renderer is `../../qa/week01_1_lecture.py`, called by the
material builder. Every text box has an overflow check. The cavity keeps
equal axis scaling. Its vorticity uses an explicitly labeled symmetric-log
color scale (linear within +/-1) to show the core and wall extremes without
clipping data. No measurements or acceptance thresholds were changed for
the redesign.

## Retained numerical anchors

- Four manufactured grids: 17, 33, 65, and 129 points per direction.
- Observed vorticity order: approximately 1.997.
- Finest-grid analytic relative L2 error: approximately 4.02e-4.
- Accepted real field: the complete Week-1 `Re=100`, 65 by 65 cavity case.
- CFD interior divergence RMS: approximately 2.42e-16.
- Post-processed versus archived vorticity relative L2: approximately 2.902%.
- Wall-condition maximum error: zero at stored precision.
- Axis-swapped adversarial implementation: relative L2 approximately 1.232,
  retained as an expected rejection.

Exact values and thresholds are in
`../../results/week01_1_scientific_software/acceptance_record.json`.

## Sources and originality

The complete narrative, mathematical example, code, graphics, and assessment
are independently authored for FlowMLLab. The public Stanford CS146S page was
consulted only for the syllabus-level topic of agent-driven development.

- Wilson et al. (2017), *Good Enough Practices in Scientific Computing*.
  https://doi.org/10.1371/journal.pcbi.1005510
- Sandve et al. (2013), *Ten Simple Rules for Reproducible Computational
  Research*. https://doi.org/10.1371/journal.pcbi.1003285
- Roache (1998), *Verification and Validation in Computational Science and
  Engineering*.
- Stanford CS146S public topic overview: https://themodernsoftware.dev/
