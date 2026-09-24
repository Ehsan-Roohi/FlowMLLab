# Week 7.3 — Self-supervised pretraining and label efficiency

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.2](../week07_2/README.md) · [Week 7.4 →](../week07_4/README.md)

Masked-autoencoder pretraining on unlabelled wakes; error versus number of labelled frames against gappy POD.

![Hidden-pixel error against the number of labelled Re110 frames](../../results/week07_3_pretraining/label_efficiency.png)

## Lecture and notebooks

**Lecture:** [Lecture 7.3](../../lectures/week07_3_masked_pretraining.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Self-supervised pretraining and label efficiency (cylinder wake) | [Open notebook](../../notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_3/W7_3_Masked_Pretraining_Label_Efficiency.ipynb) |

**Module guides:** [Week 7.3 lab](../../notebooks/week07_3/README.md)

## What you will work on

- Self-supervised pretraining and label efficiency

## Suggested route

Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.

## Experiment and results

**Problem:** Complete a wake field from 25% of its patches for a new trajectory with few labelled frames.<br>
**CFD / data:** The four retained FlowMLLab D2Q9–TRT LBM wakes (Re = 90, 100, 105, 110).<br>
**Learning method:** A masked autoencoder pretrained on the unlabelled Re90/Re100 wakes (He et al., 2022; the MAPA protocol of Tang, Spalding and Cogan, 2026), used zero-shot, with a frozen linear probe, fine-tuned and from scratch; gappy POD with transferred, target-only and pooled bases as the matched classical baselines.



The pretrained pipeline beats the same architecture from scratch at every label
count under the matched 300-step downstream budget. At `k = 1, 2`, validation
early stopping keeps the zero-shot pretrained weights; label-driven improvement
starts at `k = 4`. With the full pretraining budget the from-scratch model still
fails at `k = 2`. Gappy POD with a basis transferred from the unlabelled wakes is
nevertheless an order of magnitude more accurate (**about 1.4%** zero-shot
against **about 18%** for the network), and its pooled basis improves with every
label. The classical win is retained and explained: this periodic wake is low-rank.
[Protocol, all methods and limits](../../results/week07_3_pretraining/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 7.2](../week07_2/README.md) · [Week 7.4 →](../week07_4/README.md)
