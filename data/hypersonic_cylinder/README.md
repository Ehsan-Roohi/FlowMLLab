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
- coordinates, freestream Mach, local Mach, temperature ratio, pressure ratio,
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
fast separable ridge ensemble are teaching designs, not a reproduction of the
published full-resolution Fusion-DeepONet accuracy.
