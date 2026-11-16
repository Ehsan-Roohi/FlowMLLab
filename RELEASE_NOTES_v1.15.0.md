# FlowMLLab v1.15.0 - Trainable kinetic shock PINN laboratory

Week 10.2 now trains a BGK physics-informed neural network from random initialization
on the CPU. Students can change network depth and width, choose Adam, L-BFGS or a
hybrid, and adjust the density, momentum and energy consistency weights. The
default uses two 24-unit tanh layers, 100 Adam updates followed by 20 L-BFGS outer
updates, and a 120-second training safeguard. Actual completed work and stop
reasons are recorded.

The characteristic construction enforces the single-Maxwellian initial condition
and incoming reservoir populations. It predicts a collision source, reconstructs
the reduced distributions g, h and k, and integrates the reported physical
moments. Training uses physics consistency without DVM targets, reference-derived
scales or reference-based model selection. The short-time case has tau = 1 and
final time 0.02; initialized predictions are shown alongside trained predictions
because much of the transport is already supplied by the construction.

An independent NumPy reduced-BGK DVM supplies the primary post-training comparison.
It uses MUSCL/minmod transport and SSPRK2, analytic transverse moments, and explicit
boundary, conservation and positivity diagnostics. Paired spatial resolutions and
separate velocity/quadrature checks measure numerical sensitivity. They do not
provide a formal convergence certificate.

Seven declared demonstrations vary the optimizer, hidden depth and energy-loss
weight. Their measured losses, signed profiles, reference errors, gradient and
diagnostic evaluations, elapsed times and stop reasons remain available together.
Different optimizer phase lengths are not presented as an equal-work ranking.
The report and metrics retain improvements and deteriorations without selecting
the best run by its DVM score.

For the declared default, relative L2 heat-flux/stress errors decrease from
0.3011%/0.9829% to 0.2364%/0.5510%, while velocity and temperature errors increase.
The 400-to-800-cell fixed-CFL reference changes are 0.1213%/0.1374% for heat flux
and stress; these sensitivity measures are not uncertainty bounds. The complete
22-cell notebook ran in a fresh local CPU kernel in 94.31 seconds and retained
nine inline figures. This timing includes imports, plotting and reference solves;
it is not a hosted-Colab benchmark or a general speed claim.

The expanded English lecture explains the trainable model, optimizer choices,
reference solver and controlled exercises. Earlier BGK/Shakhov and stationary
Mach-2 research comparisons remain available with their original provenance and
failed checks. The new CPU experiment is separate from the paper's historical
training and finite-transverse-velocity reference calculations.

PyTorch is optional through the `kinetic-pinn` extra (`torch>=2.5,<3`).
Importing the core package and using the NumPy DVM does not require Torch. The
release workflow checks Python 3.12 with CPU Torch 2.14.1, executes the published
Week 10.2 notebook in a fresh kernel, and verifies an isolated core wheel without
Torch. The notebook execution preserves tracked lesson and evidence files.

[Notebook](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.15.0/notebooks/week10_2/W10_2_BGK_Shakhov_Shock_PINN_DVM.ipynb)
and [lecture](https://github.com/Ehsan-Roohi/FlowMLLab/blob/v1.15.0/lectures/week10_2_bgk_shakhov_shock_pinn_dvm.pdf).
The release assets include the package, lesson, frozen protocol, report, metrics,
physical-profile figures and verification receipts.

The teaching material cites [Ehsan Roohi's paper revision](https://arxiv.org/abs/2605.29211v2)
and the [pinned scientific repository](https://github.com/Ehsan-Roohi/PINNs-DVM/tree/add5388f7da5f7eace95fff97d34d6ed58b90169).
New teaching software uses the project license; upstream research inputs retain
their own attribution and license scope. The known
[Zenodo concept DOI](https://doi.org/10.5281/zenodo.22074169) identifies the series.
No new version DOI is assigned in this metadata.
