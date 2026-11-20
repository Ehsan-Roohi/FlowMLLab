# Week 4 — Cavity surrogates and DeepONet

<!-- cavity-seven-re -->
Separate follow-on notebooks cover [all seven Ghia Reynolds numbers](../../notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) and [CPU profile-surrogate optimization and depth](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb). Earlier notebook cells and outputs are preserved. [Benchmark data](../../references/ghia/README.md) · [Measured results](../../results/cavity_ghia_allre/README.md) · [Research ideas and sources](../../docs/CAVITY_RESEARCH_ROADMAP.md).
<!-- /cavity-seven-re -->

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 3](../week03/README.md) · [Week 4.1 →](../week04_1/README.md)

Learn how a cavity simulation becomes a qualified learning dataset, how different
surrogates use that dataset, and how to test a prediction against both CFD and
published measurements. Every lab has a different question: a good fit to two
centreline profiles is not a verified two-dimensional flow field.

## Lab-by-lab route

Start with Labs 1–3. Lab 4 asks a separate correction-model question. Lab 5 is the
bridge from reference data to numerical qualification; then use Lab 7 before
the full-field neural study in Lab 8. Lab 6 is a parallel optimizer investigation.
Return to Lab 9 after learning PINNs in Week 13 Part A.

| Lab | Aim | What to inspect in the output |
| --- | --- | --- |
| 1 — CFD data production | Solve the cavity and decide whether a numerical field is suitable for learning. | Convergence history, physical fields, centreline comparisons and saved case metadata. |
| 2 — Scalar and coordinate surrogates | Compare interpolation, polynomial regression and neural networks on complete unseen Reynolds-number cases. | Scalar predictions, training/validation histories, matched velocity fields, wall errors and signed profiles for every held-out case. |
| 3 — POD-DeepONet | Separate spatial representation from the network that predicts its coefficients. | Training-only POD content, velocity and pressure predictions, interpolation controls and measured costs. |
| 4 — Stokes-to-Navier–Stokes correction | Learn the missing inertial correction and test the complete reconstructed flow. | Stokes, corrected and CFD fields, residuals and complete-case errors; the 51-grid companion is a historical audit. |
| 5 — All-Re Ghia validation | Distinguish reference coverage, CFD accuracy, centreline fitting and field prediction. | Our CFD against Ghia, original and screened sample scores, seven-Re availability and enlarged full-field corner diagnostics. |
| 6 — Profile optimizer/depth study | Test training choices on a small, fully specified profile-regression task. | Signed profiles, seed spread, whole-case selection, optimization histories and failures as well as gains. |
| 7 — High-Re CFD qualification | Assess whether high-Re CFD labels are converged enough for the intended comparison. | Native fields, grid/time sensitivity, residual histories, Ghia discrepancies and the limits of a steady solution. |
| 8 — High-Re field POD-DeepONet | Predict full velocity fields and compare with interpolation using the same training cases. | Held-out fields, local and global errors, corner structures, vorticity and separately reconstructed pressure. |
| 9 — Label-free PINN audit | Evaluate a neural PDE solution independently of its training loss. | Boundary and continuity identities, independent equation residuals, corner sampling, seeds and failed accuracy gates. |

The [lecture](../../lectures/week04_cavity_surrogates_deeponet.pdf) follows these
lab numbers and explains the variables, algorithms, experiments and interpretation.
The [notebook index](../../notebooks/week04/README.md) distinguishes fresh CPU
experiments from notebooks that inspect retained, verified research evidence.

![POD–DeepONet cavity fields, Ghia checks, blind errors and cost](../../results/pod_deeponet/pod_deeponet_ghia_validation.png)

## Lecture and notebooks

**Lecture:** [Lecture 4](../../lectures/week04_cavity_surrogates_deeponet.pdf)

| Notebook | Read | Run / setup |
| --- | --- | --- |
| Lab 1: CFD data production | [Open notebook](../../notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab1_CFD_Data_Production_Student.ipynb) |
| Lab 2: scalar and field surrogates | [Open notebook](../../notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab2_Scalar_and_Field_Surrogates_Student.ipynb) |
| Lab 3: POD-DeepONet cavity | [Open notebook](../../notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab3_DeepONet_Cavity_Student.ipynb) |
| Lab 4: Stokes-to-Navier–Stokes correction | [Open notebook](../../notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb) |
| Lab 5: our CFD versus Ghia and seven-Re coverage | [Open notebook](../../notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb) |
| Lab 6: profile surrogates, optimizers and depth | [Open notebook](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) |
| Lab 7: new high-Re CFD and mesh/time checks | [Open notebook](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb) |
| Lab 8: full-field POD-DeepONet and interpolation | [Open notebook](../../notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) | [Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab8_High_Re_Field_DeepONet.ipynb) |
| Advanced Lab 9: label-free PINN audit (after Week 13 Part A) | [Open notebook](../../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb) | [Prerequisites and CPU setup](../../START_HERE.md#fresh-cavity-pinn-studies) |

**Module guides:** [Week 4 labs](../../notebooks/week04/)

## What you will work on

- Data qualification
- Scalar and coordinate surrogates
- Operator learning with an interpretable trunk

## Suggested route

Begin with Labs 1–3: qualify CFD, build simple surrogates, then learn the POD coefficients. Lab 5 audits seven-Re coverage; take Lab 7 mesh/time checks before Lab 8 high-Re field learning. Lab 6 is a separate profile-regression and optimizer branch, with Ghia samples as labels. Weeks 4.1/4.2 are optional ROM and Stokes-correction extensions. **Lab 9 is an advanced return visit after [Week 13 Part A](../week13/README.md), not the next introductory lesson.**

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
before the higher-Re CFD checks in [Lab 7](../../notebooks/week04/W4_Lab7_High_Re_CFD_Ghia.ipynb).
Its frozen profile predictions at Re=3200 and 7500 are a separate learning example,
not the CFD reference. Complete those CFD checks before the Lab 8 full-field tests.
[Lab 6](../../notebooks/week04/W4_Lab6_Ghia_Profile_Optimizer_Depth.ipynb) provides
the full seven-Re profile-regression protocol and optimizer/depth exercises.
The historical profile fit remains separate. New Labs 7 and 8 now add the actual high-Re CFD and full-field neural experiment.

<img src="../../results/cavity_cfd_ghia/cfd_ghia_validation.png" width="760" alt="FlowMLLab CFD versus Ghia at three Reynolds numbers">

[Native CFD results](../../results/cavity_cfd_ghia/README.md) ·
[Neural predictions and interpolation baseline](../../results/cavity_ghia_allre/README.md)

---

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 3](../week03/README.md) · [Week 4.1 →](../week04_1/README.md)

[New CFD evidence](../../results/cavity_highre_ghia/README.md) · [Full-field neural tests](../../results/cavity_field_deeponet/README.md).

## Measured solver experience

Labs 7 and 8 now end with a measured CPU solver comparison: why fewer Krylov
iterations did not reduce cavity solve time. The appendix includes successful
and failed trials, signed physical profiles, setup-cost analysis and exercises
using saved records. [Evidence and lesson links](../../results/cavity_solver_benchmark/README.md).

## Label-free PINN audit

[Lab 9](../../notebooks/week04/W4_Lab9_Label_Free_Ghia_PINN_Audit.ipynb)
examines a fresh 42-run experiment across all seven Ghia Reynolds numbers.
It separates exact boundary/continuity identities from field accuracy,
independent residuals, seed sensitivity and actual corner-sampling coverage.
The finite baseline failed its accuracy gates; the supervised field-model
results remain separate. [All outcomes and physical figures](../../results/cavity_pinn_campaign/README.md).
The existing Week 4 labs are preserved, and the lecture adds theory and results.

## Corner structures and field accuracy

Lab 5 now connects its reference audit to the frozen Lab 8 field model through
a separate [upper-left-corner diagnostic](../../results/week04_corner_audit/README.md).
The centreline MLP cannot predict a vortex map. For the field model, examine
the local velocity error, native-grid vortex centre and signed streamfunction
strength difference as well as the global norm. Identical contour levels and
fixed off-centre profiles expose differences that automatic streamline seeding
or a whole-field average can miss. This diagnostic does not refit or select a
model on its held-out targets.
