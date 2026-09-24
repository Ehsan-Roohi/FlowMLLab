# Week 6 — Physical validation and final evidence

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 5](../week05/README.md) · [Week 7 →](../week07/README.md)

Closure testing, physical validation and reproducibility.

![Independent cavity pressure-recovery validation](../../results/article_figures/fig08_pressure_recovery.png)

## Lecture and notebooks

**Lecture:** [Shared Weeks 5–6 guide](../../lectures/week05_06_project_guide.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| P0 setup and data audit | [Open notebook](../../notebooks/week05_06/P0_Project_Setup.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P0_Project_Setup.ipynb) |
| P1 Reynolds-number generalization | [Open notebook](../../notebooks/week05_06/P1_Re_Generalization.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P1_Re_Generalization.ipynb) |
| P2 physics-guided DNN/PINN objectives | [Open notebook](../../notebooks/week05_06/P2_Physics_Guided_DNN.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P2_Physics_Guided_DNN.ipynb) |
| P3 POD study | [Open notebook](../../notebooks/week05_06/P3_POD_Study.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P3_POD_Study.ipynb) |
| P4 uncertainty and data sufficiency | [Open notebook](../../notebooks/week05_06/P4_Uncertainty_Study.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P4_Uncertainty_Study.ipynb) |
| P5 rarefied cavity | [Open notebook](../../notebooks/week05_06/P5_Rarefied_Cavity.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P5_Rarefied_Cavity.ipynb) |
| P6 Fokker-Planck closure (CUDA GPU) | [Open notebook](../../notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/P6_FP_Cavity_Closure.ipynb) |

**Module guides:** [All tracks](../../notebooks/week05_06/README.md)

## What you will work on

- Fokker–Planck closure
- A-posteriori testing
- Reproducibility and communication

## Suggested route

Continue the track selected in Week 5; finish its physical checks, reproducible results and report. The lecture and project notebooks are shared across Weeks 5–6; you do not need to complete every track. P6 requires CUDA. Take the sparse-sensing companion after Week 7.

## Experiment and results

**Problem:** Recover and validate cavity pressure.<br>
**CFD / data:** FlowMLLab cavity CFD with least-squares pressure-gradient reconstruction.<br>
**Learning method:** No neural model in this pressure-recovery figure.



This existing pressure benchmark illustrates the independent physical checks
expected in a final evidence bundle; it is not a learned Fokker–Planck result.
Week 6 completes the selected Week-5 track, including optional closure testing.
[Project completion guide](../../notebooks/week05_06/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 5](../week05/README.md) · [Week 7 →](../week07/README.md)
