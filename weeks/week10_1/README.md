# Week 10.1 — Ab initio collision DeepONet

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 10](../week10/README.md) · [Week 11 →](../week11/README.md)

Molecular scattering and DSMC cylinder contours.

![Jäger Ar–Ar cylinder temperature: Exact and DeepONet at different output times](../../results/abinitio_deeponet_cylinder/temperature_exact_deeponet.png)

## Lecture and notebooks

**Lecture:** [Lecture companion](../../lectures/week10_1_abinitio_collision_deeponet.md)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Classical collision map and surrogate audit | [Open notebook](../../notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb) |

**Module guides:** [Research fields](../../results/abinitio_deeponet_cylinder/README.md)

## What you will work on

- Classical scattering, surrogate audit and a separate research companion

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Compute cylinder flow using a collision-angle surrogate.<br>
**CFD / data:** Author-supplied DS2V-based DSMC research runs.<br>
**Learning method:** DeepONet supplies scattering-angle tables, not whole-field predictions; the CPU lab is a separate Lennard-Jones analog.



Author-supplied research fields from the later DeepONet collision-angle package,
related to [Roohi, Shoja-sani and Stefanov, PoF 38, 057123 (2026)](https://doi.org/10.1063/5.0328463).
DSMC generates these fields using exact-derived or DeepONet-derived angle tables;
this is not a whole-field neural prediction or reproduction of the article's MLP.
Shared colors, **different times and sampling windows**: qualitative comparison only.
[All four colored fields and provenance](../../results/abinitio_deeponet_cylinder/README.md)
· [Surface pressure and heat flux](../../results/abinitio_deeponet_cylinder/README.md#surface-pressure-and-heat-flux)
· [Week 10.1 lecture companion](../../lectures/week10_1_abinitio_collision_deeponet.md)

[Run the CPU collision-map lab](../../notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb):
solve a reduced Lennard-Jones scattering problem, check analytic limits and
audit a fitted surrogate using transport integrals. This executable analog is
separate from the Jäger research fields above; it does not reproduce the
article's potential or network.

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 10](../week10/README.md) · [Week 11 →](../week11/README.md)
