# Week 13 - Physics-informed neural networks for the lid-driven cavity

[Notebook](W13_Rectangular_Cavity_PINN_Research.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb) ·
[Lecture](../../lectures/week13_rectangular_cavity_pinn.pdf) ·
[PINN foundations reading](../../lectures/week04_2_pinn_cavity.pdf)

Prerequisites: Weeks 1 and 2 and the PINN foundations reading. Allow 75 minutes.
CPU only; the one training cell takes about a minute.

## What the notebook does

| Part | Content | Evidence it reads |
| --- | --- | --- |
| A. Learn | Every component of a streamfunction PINN is built in one cell (network, hard wall constraints, autodiff residual, masked loss, Adam then L-BFGS). The trained field is checked three ways: training loss, held-out residual, and velocity error against the Week 1 CFD reference at Re = 100. The short run reaches a small training loss and a large field error, which is the point: the held-out residual and the field comparison are what judge a PINN. | `data/cavity_data.npz` |
| B. Apply | The retained Re = 1000, D/W = 2.2 deep-cavity field (checkpoint `restart-55118.ckpt`, 301 x 661 grid) is hash-checked, its derived fields are rebuilt, the recovered continuation loss history is plotted, and the field is shown beside Nektar++ CFD at t = 120 on a common grid (velocity relative L2 difference about 3.6%, a near-matched comparison because the lid profiles differ). | `data/week13_deep_cavity/`, `results/week13_deep_cavity/` |
| C. Audit | The four retained float64 A100 runs (Re = 100/400, depth-to-width ratio D = 1/2; Adam 1000 steps then SSBroyden2 to step 4000) are read through their optimizer histories, independent full-domain and top-corner residuals, exact-wall checks and, for the square cases, frozen CFD gates. | `results/week13_rectangular_pinn/` |

Notation used throughout: L is the cavity width, H its depth, D = H/L the
depth-to-width ratio (the lecture and older figure titles also write H/L, and
the deep case of Part B is quoted as D/W = 2.2; all three mean the same ratio).

## What the evidence supports

- The two square cases pass every frozen CFD gate (interior velocity error 3.3%
  at Re = 100 and 10.6% at Re = 400 against the Week 1 reference).
- The independent full-domain residual is two to three orders of magnitude
  larger than the masked training residual in all four cases, and the Re = 100
  case has both the largest residual and the smallest field error. Residual
  magnitude and field error answer different questions; the notebook explains why.
- The deep cases (D = 2) have no matched CFD field in this repository and are
  residual-audited hypotheses. The Part B comparison at D/W = 2.2 is the closest
  available field check and is near-matched, not matched.
- The recovered `loss_continuation.dat` ends at checkpoint 65711, later than the
  field's checkpoint 55118; it is a resumed training history, not an independent
  test loss, a CFD error, or a certificate of final convergence.

## Retained loss curves of the four-case matrix

Each plot shows the momentum residual RMS (not squared loss, not CFD error).
Blue/orange distinguish Adam and SSBroyden2; black dashed and green dotted
curves track the held-out full-domain and top-corner residuals. A low masked
training residual does not certify the full domain.

### Re = 100, D = 1

![Re100 square cavity loss history](../../results/week13_rectangular_pinn/re100-d1/loss.png)

### Re = 100, D = 2

![Re100 deep cavity loss history](../../results/week13_rectangular_pinn/re100-d2/loss.png)

### Re = 400, D = 1

![Re400 square cavity loss history](../../results/week13_rectangular_pinn/re400-d1/loss.png)

### Re = 400, D = 2

![Re400 deep cavity loss history](../../results/week13_rectangular_pinn/re400-d2/loss.png)

## Rebuilding and reproduction

- `python qa/build_week13_notebook.py --execute` rebuilds and executes the
  notebook (PyTorch required); `--keep-outputs` rewrites the text while keeping
  the outputs of unchanged code cells.
- `python qa/build_week13_materials.py` rebuilds the lecture PDF and calls the
  notebook builder.
- The training runner, restartable `gpu-preempt` batch file, harvest gate and
  protocol for the four-case matrix are in `qa/` (`WEEK13_PINN_MATRIX_PROTOCOL.md`).
  Optimizer checkpoints stay on the cluster; the public evidence retains
  histories, audits, figures, the dependency lock, upstream commit/digest and
  job identifiers.

The four-case matrix is a feasibility pilot, not a publication comparison. A
paper must add matched multi-aspect-ratio CFD/FEM references, several fixed
seeds and a factorial comparison between primitive/FOSLS and streamfunction
representations at equal precision, capacity and residual-evaluation budget.
