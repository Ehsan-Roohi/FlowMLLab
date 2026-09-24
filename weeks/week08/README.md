# Week 8 — Gas dynamics and SciML

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.4](../week07_4/README.md) · [Week 9 →](../week09/README.md)

Exact compressible-flow branches and learned inverse maps.

![Gas-dynamics model evidence, blind errors and matched-budget comparisons](../../results/gas_dynamics_week8/week8_model_evidence.png)

## Lecture and notebooks

**Lecture:** [Lecture 8](../../lectures/week08_gas_dynamics_sciml.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lab 1: exact gas dynamics before ML | [Open notebook](../../notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb) |
| Lab 2: gas-dynamics SciML evidence | [Open notebook](../../notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb) |

**Module guides:** [Week 8 labs](../../notebooks/week08/README.md)

## What you will work on

- Exact compressible-flow references
- Branch-aware gas-dynamics SciML
- Dimensional scaling and CFD bridge

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict and invert compressible-flow relations.<br>
**CFD / data:** Exact gas-dynamics relations and numerical root finding; no spatial CFD run.<br>
**Learning method:** MLP inverse maps versus interpolation and radial-basis-function baselines.



Preserve physical branches while comparing exact solvers, interpolation and
learned inverse maps under matched budgets.
[Benchmarks and validity limits](../../results/gas_dynamics_week8/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.4](../week07_4/README.md) · [Week 9 →](../week09/README.md)
