# Week 9 — Rarefied micro-step and micro-nozzle

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 8](../week08/README.md) · [Week 10 →](../week10/README.md)

Geometry-dependent and shock-aligned operators.

![Held-out H44 micro-step: DSMC and independent teaching-model contours](../../results/mahdavi_deeponet/step_independent_contours/held_out_H44_independent.png)

## Lecture and notebooks

**Lecture:** [Lecture 9](../../lectures/week09_rarefied_deeponet_case_studies.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lab 1: micro-step zonal-loss DeepONet | [Open notebook](../../notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb) |
| Lab 2: shock-aligned micro-nozzle DeepONet | [Open notebook](../../notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb) |
| Lab 3: moving-throat nozzle data-alignment audit | [Open notebook](../../notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb) |

**Module guides:** [Week 9 labs](../../notebooks/week09/README.md)

## What you will work on

- Geometry-dependent operator learning
- Physics-guided zonal objectives
- Shock-aligned rarefied-flow operators
- Branch/trunk data contracts for operator learning

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict geometry-dependent step fields and pressure-dependent nozzle fields.<br>
**CFD / data:** Author-supplied DSMC archives; the nozzle uses modified Bird-family exports with a documented boundary defect.<br>
**Learning method:** Step: geometry/coordinate MLP. Nozzle: shock-aligned POD with polynomial or tanh-MLP coefficient maps. Article and experimental DeepONet results are distinguished in the linked reports.



The independent H44 teaching model uses geometry and coordinates only.
[Step evidence and provenance](../../results/mahdavi_deeponet/README.md)

![Micro-nozzle 25-kPa full-height parity-completed view: DSMC half-domain, selected registered-POD model and absolute differences](../../results/nozzle_transport/nozzle_P25_fields_full_domain.png)

![Micro-nozzle 25-kPa profiles comparing DSMC, interpolation and learned branches](../../results/nozzle_transport/nozzle_P25_profiles.png)

The selected registered-POD polynomial model and trained neural branches are
compared with the original interpolation baseline. The displayed transverse
velocity on the symmetry plane is the prescribed **V = 0** boundary condition,
not a learned accuracy result. The full-height field view is a parity-completed
visualisation about that plane, not new CFD. Raw exports have a documented symmetry defect;
these are historical-holdout regression results, not fresh blind validation.
[Nozzle report](../../results/nozzle_transport/README.md)
· [Raw boundary audit](../../results/nozzle_transport/symmetry_boundary_audit.png)

#### Week 9 Lab 3 — Moving-throat data-alignment audit

**Problem:** Detect a silent branch/trunk correspondence error before fitting a nozzle operator.<br>
**Reference:** Independent quasi-1D isentropic solutions for three moving-throat geometries.<br>
**Learning method:** Matched 2×48 tanh MLP surrogates isolate the effect of correct versus corrupted trunk-coordinate pairing.

![Blind nozzle-surrogate predictions and errors before and after coordinate repair](../../results/nozzle_alignment_audit/nozzle_alignment_impact.svg)

In a controlled quasi-1D stress test, reusing the first nozzle's coordinates
for every target preserves array shape but raises mean relative L2 error on
three unseen geometries from **1.63% to 6.72%**. Correct pairing reduces blind
error by **76%** under the same architecture, training cases and random seed.
This isolates the value of the pre-fit audit; it is not a DSMC accuracy claim.
[Run the audit](../../notebooks/week09/W9_Lab3_Nozzle_Data_Alignment_Audit.ipynb) ·
[Metrics and protocol](../../results/nozzle_alignment_audit/README.md) ·
[Companion notes](../../lectures/week09_3_nozzle_data_alignment.pdf)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 8](../week08/README.md) · [Week 10 →](../week10/README.md)
