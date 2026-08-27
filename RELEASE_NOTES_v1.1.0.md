# FlowMLLab v1.1.0

FlowMLLab v1.1.0 adds a validated classical reduced-order-model pathway for the
same lid-driven cavity used throughout the course and research examples.

## New in v1.1.0

- adds the installable `flowmllab.cavity_rom` module and `flowmllab rom` command;
- adds `W4_1_Classical_ROM_Cavity.ipynb` without modifying earlier lectures or
  notebooks;
- implements boundary lifting, mean centering, weighted POD, intrusive
  POD--Galerkin integration, and POD--DEIM hyper-reduction;
- selects POD and DEIM dimensions before opening three blind Reynolds-number
  cases;
- validates exact FOM recovery, grid and time-step refinement, wall conditions,
  divergence, velocity, vorticity, timing, and break-even query count;
- records machine-readable validation protocols, metrics, timing, and the ROM
  model under `results/cavity_rom/`; and
- retains the existing continuum, scientific-ML, neural-operator, and DSMC
  workflows under the MIT License.

## Recorded validation gates

The selected rank-16 POD--Galerkin and rank-16 POD--DEIM models were evaluated
on blind cases at Reynolds numbers 175, 275, and 375. Across those cases, the
maximum time-dependent relative velocity error was 0.4941% for POD--Galerkin and
0.6338% for POD--DEIM. The maximum final-vorticity error was 0.5582%, the wall
velocity error was zero to machine precision, and the maximum divergence norm
was 1.43e-16. The recorded online comparison reports 0.93x POD--Galerkin speedup,
9.24x POD--DEIM speedup, and a 7.84-query break-even point after including the
offline cost.

Before publication, the tagged archive is installed in a clean environment and
the unit tests, smoke test, repository QA, notebook execution, and Python
3.10--3.12 GitHub Actions matrix must pass.

Version 1.1.0 supersedes v1.0.2 for current use and citation.
