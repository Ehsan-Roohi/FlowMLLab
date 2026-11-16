# Week 10.2 - One-dimensional kinetic shocks: BGK, Shakhov, PINNs and DVM

[Course home](../../README.md) · [All weeks](../README.md) · [← Week 10.1](../week10_1/README.md) · [Week 11 →](../week11/README.md)

How much of a distribution has a network learned when its density looks right?
Use signed heat flux, stress and fourth-order moments to answer that question
with matched one-dimensional kinetic-shock data.

## Lecture and notebook

[Lecture PDF](../../lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf) ·
[Notebook](../../notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb) ·
[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb) ·
[CPU setup and exercise map](../../notebooks/week10_2/README.md)

Read after Weeks 2, 3, 8 and 10. Start with the collision and moment calculations
before interpreting the saved neural results. The full Week 13 PINN research
audit is an optional continuation, not a prerequisite for this self-contained
introduction.

## Computation and physical interpretation

The fresh CPU exercises compute jump conditions, collision invariants, the
different BGK/Shakhov heat-flux relaxation rates and a positive-distribution
counterexample showing why g and h cannot determine separate R and Delta.
The retained experiments then compare transient shock tubes, three stationary
Mach-2 initializations and moment-supervised BGK states with matched DVM profiles.

![BGK shock-tube density, temperature and signed nonequilibrium moments for matched DVM and three learning routes](../../results/week10_2_kinetic_shock/supervised_bgk_profiles.png)

All displayed methods use the same declared comparison nodes and 6400-cell
reference. The five-field state can reproduce lower moments while missing
fourth-order moments. Adding moment supervision improves that state, but the
equation-trained model remains more accurate for several nonequilibrium moments.
The reference's training, scaling and assessment roles remain explicit; this
comparison is not an independent held-out generalization test.

[Physical profiles, metrics and limitations](../../results/week10_2_kinetic_shock/README.md) ·
[Paper v2](https://arxiv.org/abs/2605.29211v2) ·
[Pinned source repository](https://github.com/Ehsan-Roohi/PINNs-DVM/tree/add5388f7da5f7eace95fff97d34d6ed58b90169)

## Assignment

Complete the notebook's seven investigations linked to tables A-G. Vary one
numerical control at a time, retain signed physical curves, quantify seed effects,
and distinguish reference-data availability from an independent test. Report
the source's failed criteria alongside low global error norms.
