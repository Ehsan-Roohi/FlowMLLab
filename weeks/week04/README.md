# Week 4 — Cavity surrogates and DeepONet

<!-- cavity-seven-re -->
Separate follow-on notebooks cover [all seven Ghia Reynolds numbers](../../notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) and [CPU profile-surrogate optimization and depth](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb). Earlier notebook cells and outputs are preserved. [Benchmark data](../../references/ghia/README.md) · [Measured results](../../results/cavity_ghia_allre/README.md) · [Research ideas and sources](../../docs/CAVITY_RESEARCH_ROADMAP.md).
<!-- /cavity-seven-re -->

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
| Lab 5: our CFD versus Ghia and seven-Re coverage | [Open notebook](../../notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) |
| Lab 6: profile surrogates, optimizers and depth | [Open notebook](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) |
| Lab 7: new high-Re CFD and mesh/time checks | [Open notebook](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) |
| Lab 8: full-field POD-DeepONet and interpolation | [Open notebook](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) |

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

## CFD validation and all-seven-Re profile learning

[Lab 5](../../notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) opens our CFD
solutions at Re=100,400,1000, shows grid changes and signed Ghia discrepancies,
and then displays the frozen neural predictions at Re=3200 and 7500.
[Lab 6](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) provides
the full seven-Re profile-regression protocol and optimizer/depth exercises.
The historical profile fit remains separate. New Labs 7 and 8 now add the actual high-Re CFD and full-field neural experiment.

<img src="../../results/cavity_cfd_ghia/cfd_ghia_validation.png" width="760" alt="FlowMLLab CFD versus Ghia at three Reynolds numbers">

[Native CFD results](../../results/cavity_cfd_ghia/README.md) ·
[Neural predictions and interpolation baseline](../../results/cavity_ghia_allre/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 3](../week03/README.md) · [Week 4.1 →](../week04_1/README.md)

[New CFD evidence](../../results/cavity_highre_ghia/README.md) · [Full-field neural tests](../../results/cavity_field_deeponet/README.md).
