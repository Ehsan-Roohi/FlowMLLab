# Week 13 — Rectangular-cavity PINN research audit

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 12](../week12/README.md) · [Week 14 →](../week14/README.md)

Streamfunction PINNs: build one on CPU, then audit deep-cavity and four-case research runs.

![Nektar++ CFD and PINN fields for the Re=1000, D/W=2.2 cavity](../../results/week13_deep_cavity/cfd_pinn_fields.png)

## Lecture and notebooks

**Lecture:** [Lecture 13](../../lectures/week13_rectangular_cavity_pinn.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Rectangular-cavity PINNs | [Open notebook](../../notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb) |

**Module guides:** [Week 13 lab](../../notebooks/week13/README.md)

## What you will work on

- Physics-informed neural networks: streamfunction formulation, hard wall constraints, autodiff residuals, Adam then quasi-Newton; residual versus field error

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

The front page retains only two representative comparisons. The notebook has three parts:
build and train a small streamfunction PINN on CPU and judge it against the Week 1 CFD
reference; inspect the deep-cavity field beside CFD; audit the four-case research matrix.
Complete fields, loss histories and the audit tables are in the [Week 13 notebook](../../notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb).

**Problem:** Solve square and deep lid-driven cavities.<br>
**CFD / data:** Nektar++ CFD and PINN fields are shown in separate, labelled rows on a common grid. The square case retains a near-matched CFD/PINN validation panel.<br>
**Learning method:** Streamfunction PINN with hard wall constraints, trained using Adam then SSBroyden2 in float64.

The Week 13 module develops streamfunction PINNs for square and deep lid-driven
cavities. Compare **Re = 100 and 400**, **H/L = 1 and 2**, and **Adam followed
by SSBroyden2** using retained float64 A100 runs, exact wall constraints and
independent residual checks. The selected D/W=2.2 comparison uses Nektar++ at
t=120 and PINN checkpoint 55118. Its 3.59% velocity relative L2 is not a final
mesh/steady-convergence claim: the retained lid profiles differ.

[Lecture](../../lectures/week13_rectangular_cavity_pinn.pdf) ·
[Notebook and results](../../notebooks/week13/README.md) ·
[Reproduction protocol](../../qa/WEEK13_PINN_MATRIX_PROTOCOL.md)

Preparatory reading covers [PINN residuals and hard boundary constraints](../../lectures/week04_2_pinn_cavity.pdf)
(this reading and its `results/week04_2_pinn_cavity/` evidence keep their original
Week 4.2 file numbering; the Week 4.2 row of the course table is the
Stokes-to-Navier-Stokes lab),
including [McDevitt's DeepPlasma cavity code](https://github.com/cmcdevitt2/DeepPlasma/tree/fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b/LDC),
used with his permission. The earlier [Re=100 qualification and CFD comparison](../../results/week04_2_pinn_cavity/README.md)
provides supporting evidence for this module.

**Selected case A — Re=1000, D/W=2.2:** CFD and PINN speed/streamlines and
mean-zero pressure use common scales. The pressure scale is symmetric-log and
retains the full range.



**Selected case B — Re=100, D/W=1:** the PINN fields, pointwise velocity
difference and CFD/PINN centreline profiles are kept as the compact square-case
qualification. This selected near-matched run has 3.10% interior velocity
relative L2; the separate frozen four-case matrix reports 3.343% for its
`Re=100`, `D/W=1` checkpoint.

![Near-matched CFD validation of the Re=100, D/W=1 PINN](../../results/week04_2_pinn_cavity/qualified_validation.png)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 12](../week12/README.md) · [Week 14 →](../week14/README.md)
