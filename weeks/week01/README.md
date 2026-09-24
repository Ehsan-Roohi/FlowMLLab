# Week 1 — Numerical foundations

[Course home](../../README.md) · [All weeks](../README.md) · [Week 1.1 →](../week01_1/README.md)

Python, numerical methods and CFD validation.

![Cavity CFD benchmark and Ghia velocity validation](../../results/article_figures/fig02_cavity_benchmark.png)

## Lecture and notebooks

**Lecture:** [Lecture 1](../../lectures/week01_numerical_foundations.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Python for CFD and AI | [Open notebook](../../notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/01_python_for_cfd_ai_fluids.ipynb) |
| TensorFlow for AI in fluids | [Open notebook](../../notebooks/week01/02_tensorflow_for_ai_fluids.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/02_tensorflow_for_ai_fluids.ipynb) |
| Cavity CFD and Ghia validation | [Open notebook](../../notebooks/week01/03_cavity_ghia.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/03_cavity_ghia.ipynb) |

**Module guides:** [Week 1 labs](../../notebooks/week01/)

## What you will work on

- Eulerian fields, nondimensionalization, boundary conditions
- Python/NumPy/TensorFlow for scientific work
- Numerical convergence versus validation

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Lid-driven cavity benchmark.<br>
**CFD / data:** FlowMLLab finite-difference streamfunction–vorticity Navier–Stokes solver.<br>
**Learning method:** No neural network; CFD is checked against Ghia centreline data.



Start with a numerical solution and an independent benchmark: cavity fields,
centerlines and Ghia comparisons establish what a useful training label means.
[Figure contract](../../ARTICLE_FIGURE_MAP.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [Week 1.1 →](../week01_1/README.md)
