# Week 4.2 — Stokes-to-Navier-Stokes correction

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4.1](../week04_1/README.md) · [Week 5 →](../week05/README.md)

Matched Stokes input and learned Navier–Stokes correction.

![Week 4.2 diverse-lid example: Navier–Stokes reference, Stokes-corrected prediction and absolute errors in velocity and recovered pressure](../../figures/Cavity_diverse_velocity_pressure.png)

## Lecture and notebooks

**Lecture:** [Week 4.2 companion](../../lectures/week04_2_stokes_to_navier_stokes.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lab 4: Stokes-to-Navier-Stokes correction | [Open notebook](../../notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) |
| Lab 4 addendum: 51 x 51 grid validation | [Open notebook](../../notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Grid51_Validation.ipynb) |

## What you will work on

- Multi-fidelity correction

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Predict the nonlinear cavity-flow correction from a matched Stokes field.<br>
**CFD / data:** Constant and spatially diverse lid conditions, solved independently on 25 × 25 and 51 × 51 grids.<br>
**Learning method:** A POD-based neural correction conditioned on the Stokes field and Reynolds number; pressure is recovered afterwards.



![Week 4.2 diverse-lid speed contours with streamlines and primary and corner recirculation markers, plus interior vorticity and error](../../figures/Cavity_diverse_streamlines_vorticity.png)

This held-out diverse-lid example shows reference fields, predictions and absolute
errors, including speed streamlines and interior vorticity. Across six retained
same-family tests per lid type, the primary and lower-right recirculation centers
match the reference grid nodes in all cases; mean interior vorticity errors are
0.12% for constant lids and 1.49% for diverse lids. The contours are linearly
interpolated for display; errors use the original 25 × 25 samples. These same-grid
regression tests do not establish grid-independent CFD accuracy or independently
validated corner vortices. [See the constant-lid vortex figure](../../figures/Cavity_constant_streamlines_vorticity.png)
· [Run the Week 4.2 lab](../../notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb)
· [Read the ten-page companion](../../lectures/week04_2_stokes_to_navier_stokes.pdf)
· [Inspect the retained results](../../results/stokes_refined/README.md)

The 51 × 51 refinement solves all 184 cases again and retrains the selected
three-seed POD correction. Mean same-family velocity errors are 0.097% for
constant lids and 0.810% for diverse lids, compared with 0.107% and 0.917%
at 25 × 25. The 25-to-51 CFD velocity change is still about 20.6% for constant
tests and 17.8% for diverse tests when the fine solution is sampled on the
coarse grid. This is a better resolved surrogate experiment, not a claim of
mesh-independent CFD accuracy.

![Week 4.2 diverse-lid 51 by 51 speed, streamlines, vorticity and errors](../../figures/Cavity_diverse_grid51_streamlines_vorticity.png)

[Open the 51 × 51 validation notebook](../../notebooks/week04/W4_Lab4_Grid51_Validation.ipynb)
· [Read the 51 × 51 PDF addendum](../../lectures/week04_2_grid51_validation.pdf)
· [Inspect the refined data and metrics](../../results/stokes_grid51/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4.1](../week04_1/README.md) · [Week 5 →](../week05/README.md)
