# Week 4 notebook guide

[Course unit](../../weeks/week04/README.md) · [Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf) · [Setup](../../START_HERE.md)

Read the numbered lab section of the lecture alongside the corresponding notebook.
Begin with the aim, identify the input data and held-out cases, run the cells in
order, and complete the experiments at the end. A small training loss is not the
final result: compare fields, signed profiles, boundaries and independent checks.

| Lab | Notebook | Execution and learning task |
| --- | --- | --- |
| 1 | [CFD data production](W4_Lab1_CFD_Data_Production_Student.ipynb) | Qualify numerical fields before using them as labels; retain solver histories and metadata. |
| 2 | [Scalar and field surrogates](W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) | Fresh CPU learning from a retained matched cavity dataset; compare every trained method with physical outputs. |
| 3 | [POD-DeepONet](W4_Lab3_DeepONet_Cavity_Student.ipynb) | Inspect retained blind predictions and run the declared CPU refit when investigating the branch and rank. |
| 4 | [Stokes-to-Navier–Stokes](W4_Lab4_Stokes_to_Navier_Stokes.ipynb) | Learn a correction to the simplified equations, then test the complete corrected field. |
| 5 | [All-Re Ghia validation](W4_Lab5_All_Re_Ghia_Validation.ipynb) | Recompute errors from hash-checked CFD and distinguish centreline regression from full-field corner accuracy. |
| 6 | [Profile optimizers and depth](W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) | Inspect all seven cases and run optional CPU experiments without changing the frozen comparison. |
| 7 | [High-Re CFD and Ghia](W4_Lab7_High_Re_CFD_Ghia.ipynb) | Audit retained mesh and time-step evidence on CPU; full campaign generation is separate. |
| 8 | [High-Re field DeepONet](W4_Lab8_High_Re_Field_DeepONet.ipynb) | Inspect frozen complete-case field predictions and interpolation controls; optional fresh training is separate. |
| 9 | [Label-free Ghia PINN audit](W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb) | Advanced return after Week 13 Part A: inspect verified outcomes of the research campaign, including failures. |

Two companions retain separate roles:

- [Classical POD/Galerkin/DEIM ROM](W4_1_Classical_ROM_Cavity.ipynb), the Week 4.1 extension.
- [Lab 4 grid-51 validation](W4_Lab4_Grid51_Validation.ipynb), a historical audit of the correction study rather than a replacement for the main Lab 4.

Default evidence-reading paths run on CPU. They do not launch the high-Re CFD
or PINN research campaigns, and measured local execution is not a guarantee of
identical timing on Google Colab. Each optional training/solver switch explains
what new work it requests. Keep the retained test predictions unchanged when
experimenting; use development cases for architecture, optimizer or loss choices.
