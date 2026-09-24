# Week 5 — Physics-guided projects

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4.2](../week04_2/README.md) · [Week 6 →](../week06/README.md)

POD, physics-guided learning and frozen project protocols.

![Animated cavity comparison for three retained blind POD–DeepONet cases](../../assets/flowmllab_blind_demo.gif)

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
| Sparse sensing and dynamics (cylinder wake) | [Open notebook](../../notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb) |

**Module guides:** [Week 5 project setup and tracks](../../notebooks/week05_06/README.md)

## What you will work on

- POD and reduced-order learning
- Physics-guided objectives and PINNs
- Research protocol

## Suggested route

Start with P0, choose one project track, and freeze its question, baseline and evaluation plan. The lecture and project notebooks are shared across Weeks 5–6; you do not need to complete every track. P6 requires CUDA. Take the sparse-sensing companion after Week 7.

## Experiment and results

**Problem:** Build a cavity surrogate or reconstruct a wake from sparse sensors.<br>
**CFD / data:** FlowMLLab cavity CFD for the animation; D2Q9–TRT LBM for the wake extension.<br>
**Learning method:** POD–DeepONet for the animation; gappy POD, sensor placement and SINDy in the extension.



Use this retained cavity experiment as a project starting point: freeze a baseline,
change one modeling choice and evaluate complete unseen cases.
[Project pack](../../notebooks/week05_06/README.md) · [Interactive demo](../../demo/README.md)

Extend the project with the [sparse sensing lab](../../notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb):
reconstruct fields from limited measurements and inspect the
[retained modal-method comparisons](../../results/modal_labs/README.md).
This companion uses the Week 7 cylinder-wake data and should be taken after
the Week 7 module.

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 4.2](../week04_2/README.md) · [Week 6 →](../week06/README.md)
