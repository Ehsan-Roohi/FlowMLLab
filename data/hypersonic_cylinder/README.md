# Week 7.1 hypersonic-cylinder teaching data

`cylinder_teaching_subset.npz` is a compact, deterministic derivative of the
author-supplied archive `AllMachNNCylinder.zip`. It supports the incremental
Week-7.1 rarefied-cylinder lab without placing the 1.4 GB archive or roughly
4.9 GB of uncompressed research files in the course repository.

The source is associated with:

> E. Roohi et al., "Neural Networks for Rarefied Gas Dynamics: Relaxation
> Problem, Polyatomic Shock Waves, and Hypersonic Cylinder Flow," *Physics of
> Fluids* **38**, 057108 (2026), <https://doi.org/10.1063/5.0334590>.

## Contents and derivation

- 20 freestream-Mach cases from 5 through 15, including half- and quarter-Mach cases;
- a fixed 50 by 50 selection from each original 400 by 400 structured grid;
- finite points only, after removal of DSMC solid/sentinel entries;
- coordinates, freestream Mach, local Mach, source temperature (TOV), source pressure (P),
  original case identifier, and source-grid row;
- 44,500 retained points in about 0.5 MB.

The exact archive hash, derived-file hash, source entries, and retained count
for every case are recorded in `manifest.json`. Rebuild only from the reviewed
archive:

```bash
python qa/build_hypersonic_cylinder_subset.py \
  --archive /path/to/AllMachNNCylinder.zip
```

## Scope

The corresponding author supplied and released this derivative for FlowMLLab
teaching use. It is not a general relicensing of every script, checkpoint, log,
or collaborator artifact in the source archive. The classroom split and the
trained 3x96 tanh MLP are teaching designs, not a reproduction of the
published full-resolution Fusion-DeepONet accuracy.

The quoted teaching-use permission is not a standardized open-data license.
CC BY 4.0 has **not** been assigned to this derivative. Broader redistribution
or PhysicsNeMo adoption requires documented approval from the relevant rights
holders, including coauthors where applicable. The software's MIT license does
not establish those data rights. See [case provenance](../../DATA_PROVENANCE.md).

The historical NPZ keys `temperature_ratio` and `pressure_ratio` remain for
byte-level compatibility only; source normalization is not verified by those
names. Classroom displays use TOV and P without asserting a freestream ratio.

## Reading the article comparison correctly

In the [publisher-provided full text](https://www.researchgate.net/publication/404728916_Neural_networks_for_rarefied_gas_dynamics_Relaxation_problem_polyatomic_shock_waves_and_hypersonic_cylinder_flow),
Sec. V.E and Fig. 31 compare DSMC, DeepONet and **linear extrapolation** for
Mach 15 along the stagnation line and cylinder surface. It would be incorrect
to say the paper has no linear baseline. That profile comparison is not the
course's full-field two-bracketing-case interpolation test. A directly matching
three-target global relative-L2 table was not established in this review;
do not substitute the course metrics for published model errors. Exact
quantitative reproduction requires the corresponding checkpoint, masks,
preprocessing and per-figure split.
