# Matched cavity solver benchmark

The original sparse-direct solver was faster in all four matched warm-case/grid comparisons. The tested GMRES/multigrid implementation took 1.95 to 2.17 times as long, with the same accepted numerical solution. The original production solver is retained.

Hardware: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz, node uri-cpu074; Python 3.12.3, NumPy 2.2.6, SciPy 1.15.3. The two grid studies ran in separate one-core Slurm steps on the same node; paired methods within each study ran sequentially.

The existing damped Newton solver with a sparse direct linear solve is compared with the same outer Newton iteration using GMRES and a coupled geometric Galerkin multigrid preconditioner. No PDE, grid, boundary condition, line search, or nonlinear acceptance tolerance was changed.

Warm cases use the same retained Re=6500 field for Re=7500 and the same retained Re=9000 field for Re=10000. Continuation starts from the same retained 128-cell Re=1000 field and follows the same 20-root path. These are not cold-start timings.

Solve wall time includes Jacobian assembly and preconditioner construction; it excludes queueing, imports, initial input I/O, seed generation, verification, and checkpoint writing. One CPU thread per trial; methods run sequentially on the same allocation with fresh processes. Warm comparisons use AB, BA, AB order.

Each successful solution must have original nonlinear residual <1e-8, combined velocity relative L2 difference <1e-8, and vorticity relative L2 difference <1e-7 against the retained same-grid CFD field. These are agreement checks, not error against Ghia or a continuum exact solution.

## Pilot

| Trial | ILUT fill / drop | Passed | Solve seconds | Failure |
|---|---|---|---:|---|
| pilot/n128/00_warm7500_direct_r0 | — | True | 1.8064331739442423 |  |
| pilot/n128/01_warm10000_direct_r0 | — | True | 1.8022622839780524 |  |
| pilot/n128/02_warm7500_gmres_mg_r0 | 4.0 / 0.001 | False | — | RuntimeError: Factor is exactly singular |
| pilot/n128/03_warm10000_gmres_mg_r0 | 4.0 / 0.001 | False | — | RuntimeError: Factor is exactly singular |
| pilot/n128/04_warm7500_gmres_mg_r0 | 8.0 / 0.0001 | False | — | RuntimeError: Factor is exactly singular |
| pilot/n128/05_warm10000_gmres_mg_r0 | 8.0 / 0.0001 | False | — | ArithmeticError: Krylov solve did not meet the original-equation gate |
| pilot/n128/06_warm7500_gmres_mg_r0 | 12.0 / 1e-05 | False | — | ArithmeticError: Krylov solve did not meet the original-equation gate |
| pilot/n128/07_warm10000_gmres_mg_r0 | 12.0 / 1e-05 | False | — | ArithmeticError: Krylov solve did not meet the original-equation gate |
| pilot_strong/n128/00_warm7500_direct_r0 | — | True | 1.9421047098003328 |  |
| pilot_strong/n128/01_warm10000_direct_r0 | — | True | 1.9514145799912512 |  |
| pilot_strong/n128/02_warm7500_gmres_mg_r0 | 20.0 / 1e-06 | True | 4.898684271145612 |  |
| pilot_strong/n128/03_warm10000_gmres_mg_r0 | 20.0 / 1e-06 | True | 4.8349206410348415 |  |
| pilot_strong/n128/04_warm7500_gmres_mg_r0 | 40.0 / 1e-08 | True | 3.849818138871342 |  |
| pilot_strong/n128/05_warm10000_gmres_mg_r0 | 40.0 / 1e-08 | True | 3.7622614642605186 |  |

## Matched results

| Grid nodes | Case | Method | Passed | Median seconds [range] | Peak RSS MiB |
|---|---|---|---|---:|---:|
| 257 × 257 | continuation | direct | 1/1 | 247.442 [247.442, 247.442] | 517.1 |
| 257 × 257 | continuation | gmres_mg | 1/1 | 561.950 [561.950, 561.950] | 613.7 |
| 257 × 257 | warm10000 | direct | 3/3 | 14.855 [14.523, 15.257] | 533.7 |
| 257 × 257 | warm10000 | gmres_mg | 3/3 | 28.998 [28.541, 29.299] | 572.4 |
| 257 × 257 | warm7500 | direct | 3/3 | 14.248 [14.224, 14.680] | 525.8 |
| 257 × 257 | warm7500 | gmres_mg | 3/3 | 29.270 [29.229, 29.544] | 590.0 |
| 513 × 513 | warm10000 | direct | 3/3 | 133.804 [133.436, 136.642] | 2279.0 |
| 513 × 513 | warm10000 | gmres_mg | 3/3 | 273.301 [269.880, 273.742] | 2385.4 |
| 513 × 513 | warm7500 | direct | 3/3 | 130.158 [127.245, 134.148] | 2234.7 |
| 513 × 513 | warm7500 | gmres_mg | 3/3 | 283.005 [279.991, 293.304] | 2363.3 |

The continuation comparison has one repetition per method; the warm cases have three. A ratio from one continuation pair should not be treated as a hardware-independent speedup.

- 257 nodes, warm7500: GMRES-multigrid / direct median wall-time ratio = 2.0544 (above 1 means multigrid is slower).
- 257 nodes, warm10000: GMRES-multigrid / direct median wall-time ratio = 1.9521 (above 1 means multigrid is slower).
- 257 nodes, continuation: GMRES-multigrid / direct median wall-time ratio = 2.2710 (above 1 means multigrid is slower).
- 513 nodes, warm7500: GMRES-multigrid / direct median wall-time ratio = 2.1743 (above 1 means multigrid is slower).
- 513 nodes, warm10000: GMRES-multigrid / direct median wall-time ratio = 2.0425 (above 1 means multigrid is slower).

Profiling explains the observed slowdown: construction of the candidate preconditioner accounted for approximately 89.3% to 95.2% of warm-case solve time (ratios of median timings). Both methods used [4] Newton steps in these cases. This implementation rebuilds the strong ILUT-based hierarchy at every Newton step; that setup cost dominates, even though GMRES converges in few iterations.


## Direct comparison of complete fields

Across the 5 representative repeat-0 field pairs, the largest relative L2 difference is 2.11031e-14 for combined velocity and 1.71166e-14 for vorticity. These compare the two methods directly over the entire grid; they are not differences from Ghia data.

Velocity difference is sqrt(sum((u_MG-u_direct)^2 + (v_MG-v_direct)^2)) / sqrt(sum(u_direct^2 + v_direct^2)). Vorticity uses the analogous relative L2 norm.


## Limits

This benchmark assesses one experimental preconditioner, not all multigrid or Newton–Krylov methods. It is not a reimplementation of Ghia et al.’s coupled strongly implicit multigrid algorithm. Both methods compute the same two-dimensional steady branch. Converging that branch does not establish temporal stability of a high-Reynolds-number physical flow. Memory includes the complete Python child process, not just the factorization.

Historical Ghia timings are not a matched benchmark: the hardware, solver, starting field, discretization and stopping conditions differ. The present paired measurements support algorithm comparisons only on the recorded hardware and inputs.

The three original lightweight 128-cell configurations failed. A bounded follow-up increased ILUT strength while keeping the equations and all acceptance gates unchanged. The successful setting was frozen before any 256/512-cell candidate evaluation.

Exact source hashes, input array hashes, NPZ container hashes, host/CPU model, per-linear-solve histories, failures, and numerical gates are in `evidence.json` and `verified_summary.json`. Complete recovery states and representative full fields remain in the recorded Unity Workspace.
