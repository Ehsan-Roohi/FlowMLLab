# Week 10.2 - One-dimensional kinetic shocks

[Course home](../../README.md) · [Week guide](../../weeks/week10_2/README.md) · [Lecture](../../lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf)

Study why accurate density and temperature can coexist with inaccurate signed
heat flux, stress and higher moments. This separate lesson uses Ehsan Roohi's
[BGK/Shakhov PINN paper, revised 13 November 2026](https://arxiv.org/abs/2605.29211v2)
and a small, hash-verified extract from its public scientific repository.

[Open the notebook](W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb) ·
[Run in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb)

## Before running

Use Python 3.10-3.13 and the CPU PyTorch teaching extra:

```sh
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[kinetic-pinn]"
```

Run the notebook from top to bottom on CPU. It uses NumPy, pandas, Matplotlib and PyTorch;
it does not need a GPU or a neural-network checkpoint. A fresh Colab session uses
a sparse clone of the package and this module's data, then installs the package
constraints. Existing edited checkouts are preserved. Local fresh-kernel execution
is recorded in [the execution receipt](../../results/week10_2_kinetic_shock/execution.json);
that receipt is not a hosted-Colab timing certificate.

Prerequisites are Week 2 data roles, Week 3 Maxwellians/moments, Week 8 normal-shock
jumps and Week 10 profile validation. The accompanying lecture introduces the
kinetic PINN formulations. Its transient source network and characteristic
integral differ from a network that directly predicts every plotted moment.

## What runs, and what is read

**Physical conditions:** the tube figures use a monatomic gas (R_g = 1, γ = 5/3),
x ∈ [−0.5, 0.5], t = 0.02, and constant τ = Kn = 1 in the source scaling.
The initial left/right (ρ, u, T) states are (1, 0, 1) and (0.125, 0, 0.8),
joined by H = [1 + tanh(20x)]/2. BGK has Pr = 1; the Shakhov tube has Pr = 2/3.
Both reservoirs start at rest, so **the tubes have no prescribed stationary
upstream Mach number**. Local Mach evolves with the solution. The separate
stationary experiment is **M₁ = 2, Shakhov, Pr = 2/3, τ = 1/ρ**, with upstream
ρ₁ = T₁ = 1. Section 1 puts this distinction beside the figure/table registry.

| Tables | Work | Evidence |
| --- | --- | --- |
| A | Verify case identities, coordinates and hashes | Published source extract |
| B | Compute monatomic Rankine-Hugoniot states and steady fluxes | Fresh analytic/numerical calculation |
| C | Construct two positive distributions with the same g and h but different R and Delta | Fresh velocity-moment calculation |
| D | Vary quadrature order, velocity cutoff and Prandtl number | Fresh homogeneous BGK/Shakhov collision calculation |
| E | Compare transient BGK and Shakhov profiles with their matched DVM | Saved PINN predictions, freshly rescored |
| F | Compare three independent Mach-2 initializations and optimizer stages | Saved PINN/DVM profiles and source assessment limits |
| G | Compare equation-trained, five-field and eight-moment BGK states | A declared common-overlap reference grid and explicit data roles |
| H | Train a new CPU BGK PINN; compare initial and trained nine-moment profiles | Random weights, physics-only loss and an independently computed DVM |
| I | Refine the DVM in space/time and velocity; reevaluate frozen PINN quadrature | Fresh numerical sensitivity, no retraining or convergence-order claim |
| J | Change depth, optimizer and energy-loss weight in seven declared configurations | Optional fresh sweep, with published protocol and measured results available |

Run All trains one small BGK model from scratch with two 24-neuron hidden layers,
100 Adam updates followed by 20 L-BFGS outer steps, and a 120-second training
ceiling. It solves independent 400/800-cell references and checks velocity order
48/64. The seven-model sweep is optional. Actual work, stop reason and CPU timing
are printed; first-use imports and hardware affect runtime. Full historical
training budgets remain separate. A saved final state is not an automatic
convergence certificate.

## Student work

Seven research-audit and five fresh-training investigations finish their respective
sections. They require parameter sweeps,
measured tables, signed physical profiles, seed comparisons, density-thickness
sampling sensitivity, and an audit of which reference information was used during
training and selection. Each exercise names its corresponding table A-J.

In Section 10, edit `DEPTH`, `WIDTH`, `OPTIMIZER`, `LOSS_WEIGHTS`, point count and
training budget, then rerun from the training cell. `kp.train(config)` always
creates a fresh initialization; it never resumes a previous fit. The independent
physics audit is not used for updates. DVM scores are computed after fitting;
using them to choose hyperparameters turns that reference into development data.
Zero loss weights are explicitly incomplete-consistency ablations.

Keep model identities, relaxation laws, velocity quadrature and reference windows
fixed when making a matched comparison. Report both improvements and regressions.
For stationary shocks, use one upstream Mach number per figure. For transient
tubes, state the initial states and time instead of assigning an unmeasured
shock Mach. Put legends outside the plotted curves.

## Sources and rebuild

[Data provenance and reuse notice](../../data/week10_2_kinetic_shock/README.md) ·
[Results and metrics](../../results/week10_2_kinetic_shock/README.md) ·
[Teaching implementation](../../flowmllab/kinetic_shock.py) ·
[Fresh PINN](../../flowmllab/kinetic_pinn.py) ·
[Independent DVM](../../flowmllab/kinetic_dvm.py) ·
[Bounded tests](../../tests/test_kinetic_shock.py)

The additive builder [qa/build_week10_2_materials.py](../../qa/build_week10_2_materials.py)
can regenerate this new notebook, figures and lecture. It never rebuilds the
earlier course notebooks. PDF authoring requires the genuine Times New Roman
fonts specified in the [lecture typography contract](../../qa/LECTURE_TYPOGRAPHY.md).
