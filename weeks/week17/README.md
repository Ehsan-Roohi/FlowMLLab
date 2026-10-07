# Week 17 — Attention as a learned kernel

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 16](../week16/README.md) · [Week 18 →](../week18/README.md)

Q/K/V algebra, permutation tests and attention interpretation.

![Re105 velocity contours: CFD reference with sixteen sensors, SensorSet reconstruction and absolute error](../../docs/assets/transformer-contours/week17.png)

## Lecture and notebooks

**Lecture:** [Lecture 17](../../lectures/week17_cfd_transformer.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Attention as a learned kernel | [Open notebook](../../notebooks/week17/W17_CFD_Transformer.ipynb) | [Setup and reproduction guide](../../notebooks/week17/README.md) |

## What you will work on

- Attention as a learned kernel on a wake

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

<a id="transformer-labs"></a>

**Problem:** Which attention properties follow from the algebra, and which need physical evidence?<br>
**CFD / data:** Retained simulated wake observations; the contour preview uses Re105 frame 140 and the saved seed-17 SensorSet prediction.<br>
**Learning method:** Build Q, K and V, test permutation equivariance and vary temperature/key masks.

The velocity contours compare the CFD reference, reconstruction from 16 sensors and absolute spatial error. The notebook separately inspects trained attention heads: their weights describe model computations and do not establish causal physical influence.



[Student notebook](../../notebooks/week17/W17_CFD_Transformer.ipynb) · [Lecture](../../lectures/week17_cfd_transformer.pdf) · [Setup and assignment](../../notebooks/week17/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 16](../week16/README.md) · [Week 18 →](../week18/README.md)
