# Checks performed for the initial SPARTA step pilot

Checked on 2026-09-05 in the development runtime. **No Unity job was submitted
from this runtime.** SPARTA source: `95b9abaa8bd548991cc3c3f1c58b34722f7ade74`,
unmodified. The serial executable was compiled with g++ and used for actual
solver tests, not a Python surrogate or fabricated output.

| Check | Result |
|---|---|
| Serial compile | Passed |
| h/H=0.25 smoke, 300 steps | Passed; 925 fluid cells, zero stuck particles |
| h/H=0.50 smoke, 300 steps | Passed; 850 fluid cells, zero stuck particles |
| h/H=0.75 smoke, 300 steps | Passed; 775 fluid cells, zero stuck particles |
| Equal-pressure control, 40 reference PPC, 2000 steps | Passed finite-particle equilibrium screening |
| Full 1000x200 mesh initialization and one move/check step | Passed; 5,100,000 initial particles, zero stuck |
| Fluid geometry area and empty solid cells | Checked against analytic step area |
| Thermal p=n*kB*T, finite fields, exact final step | Passed |
| Truncated final output | Correctly rejected |
| Mock Slurm pipeline | Correct `afterok` build → pilot → collect; single-node/export rules |
| Partial Slurm submission failure | Previously submitted job ID retained |
| Python compilation and Bash syntax | Passed |
| MPI executable compilation against MPICH 4.3.2 | Passed locally |
| Local 2-rank MPI execution | **Blocked at MPI initialization** by runtime socket restrictions; no MPI pass claimed |
| Unity OpenMPI 5.0.3 execution | Pending mandatory in-cluster build/preflight job |

For the equal-pressure control, the first/last-cell mean pressures were about
30.244 and 31.779 kPa versus a 32.081 kPa reservoir, and mean bulk velocity was
about -4.00 m/s. This short coarse test detects large setup mistakes; it is not
a precise equilibrium or transport-coefficient validation.

An additional **deliberately coarse** 100x40 pressure-driven serial run used
8 outlet-reference PPC, 40,000 warmup and 10,000 sampling steps. It completed
with zero stuck particles and 3400 fluid cells. Its diagnostic results were:

| Diagnostic | Observed |
|---|---:|
| Net inlet mass flow per unit depth | 8.85338e-4 kg/(m s) |
| Net outlet mass flow per unit depth | 9.25426e-4 kg/(m s) |
| Relative mass imbalance | 4.33% |
| First/second-half bulk-velocity drift | 0.99% |
| Inlet-adjacent pressure error | 8.18% |
| Outlet-adjacent pressure error | **16.41%** |
| Largest cell / minimum sampled lambda | **10.70** |
| Maximum dt / sampled collision time | 0.218 |

That run **fails the pressure and spatial-resolution screens**. It establishes
that the implementation evolves a pressure-driven flow and records useful
diagnostics; it does not establish agreement with Bird or justify using these
fields as training truth. The proposed Unity pilot has ten times finer x cells,
five times finer y cells and higher particle population, but still requires
measured verification. No improvement is presumed before it runs.

Remaining scientific work: recover the exact Bird Kn reference state and gas
relaxation settings, check reservoir treatment, establish stationarity and
uncertainty, perform mesh/dt/PPC studies, and compare raw profiles at matched
physical parameters. `training_data_approved=false` is intentional.
