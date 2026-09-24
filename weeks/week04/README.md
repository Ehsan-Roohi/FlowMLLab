# Week 4 — Cavity surrogates and DeepONet

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 3](../week03/README.md) · [Week 4.1 →](../week04_1/README.md)

CFD datasets, field surrogates and operator learning.

![POD–DeepONet cavity fields, Ghia checks, blind errors and cost](../../results/pod_deeponet/pod_deeponet_ghia_validation.png)

## Lecture and notebooks

**Lecture:** [Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lab 1: CFD data production | [Open notebook](../../notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) |
| Lab 2: scalar and field surrogates | [Open notebook](../../notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) |
| Lab 3: POD-DeepONet cavity | [Open notebook](../../notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) |

**Module guides:** [Week 4 labs](../../notebooks/week04/)

## What you will work on

- Data qualification
- Scalar and coordinate surrogates
- Operator learning with an interpretable trunk

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Map Reynolds number to cavity fields.<br>
**CFD / data:** FlowMLLab finite-difference streamfunction–vorticity Navier–Stokes solver.<br>
**Learning method:** POD–DeepONet: learned parameter-to-coefficient map with a fixed POD spatial basis.



Complete-case testing combines field error, wall/divergence checks, reference
centerlines and measured inference cost.
[Model and validation evidence](../../results/pod_deeponet/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 3](../week03/README.md) · [Week 4.1 →](../week04_1/README.md)
