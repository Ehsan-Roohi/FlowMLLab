# Week 7.4 — Diverse-wake pretraining

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.3](../week07_3/README.md) · [Week 8 →](../week08/README.md)

Representation transfer and target-label efficiency for lift.

![Lift decoding error versus target-label budget on four Reynolds trajectories](../../results/week07_4_diverse_pretraining/homepage_week07_4.png)

## Lecture and notebooks

**Lecture:** [Lecture 7.4](../../lectures/week07_4_diverse_wake_pretraining.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Diverse-wake pretraining and lift decoding | [Open notebook](../../notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_4/W7_4_Diverse_Wake_Pretraining.ipynb) |

**Module guides:** [Week 7.4 lab](../../notebooks/week07_4/README.md)

## What you will work on

- Representation transfer to external force labels

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Decode instantaneous lift on Reynolds trajectories excluded from pretraining.<br>
**CFD / data:** Sixteen compact D2Q9-TRT LBM trajectories, Re60-135; eleven development cases, Re105 validation, four target cases.<br>
**Learning method:** Frozen pretrained and random encoders plus ridge, compared with POD-32 plus ridge.



The retained pretrained encoder achieves **12.31% mean lift NRMSE with 32 target labels**, versus **16.02% for the random encoder with 128**: better mean accuracy with one-quarter as many target labels. POD-32 leads through k=32; the pretrained encoder has lower mean error at k=64 and 128. This is a MAPA-inspired teaching result, not a reproduction of MAPA or a universal neural advantage. Source labels used in ridge selection are additional to k. One encoder initialization and coarse, fixed-geometry CFD limit the claim.
[Protocol, per-trajectory results and limitations](../../results/week07_4_diverse_pretraining/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.3](../week07_3/README.md) · [Week 8 →](../week08/README.md)
