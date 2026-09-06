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

The committed derivative is licensed **CC BY 4.0**, authorized by Ehsan Roohi
on 2026-09-05 after confirming ownership of the cylinder data. See the
[data-specific grant, attribution and scope](DATA_LICENSE.md). The paper's
open-access status and the software's MIT license are separate from this grant.
The cylinder licensing gate is resolved; scientific reproducibility and any
PhysicsNeMo integration still require their own checks.
See [case provenance](../../DATA_PROVENANCE.md).

## Verified import semantics; original reference conditions still needed

The [source audit](source_audit.json) matches **all 44,500 retained rows in all
20 cases** exactly to the archived `MA`, `TOV` and `P` columns after the recorded
float32 conversion. The course import performs **no freestream division**.
Manual inspection of the three hashed grid-conversion scripts found spatial
interpolation and solid masking, not a freestream normalization. The historical
NPZ keys `temperature_ratio` and `pressure_ratio` are compatibility aliases,
not verified physical ratios; the NPZ bytes and stored model remain unchanged.

The upstream-edge medians are approximately 199.46–200.92 for TOV and
1.1331–1.1386 for P. A check of `P` against `ND * k_B * TTR` using the SI
Boltzmann constant gives per-case median relative discrepancies of
3.48e-5–3.90e-5, with individual discrepancies as large as 6.66%. These are
**consistency evidence for kelvin/pascal units**, not authoritative solver-unit
metadata or recovered freestream constants. Independently interpolated fields
need not preserve products. The full audit retains median, 99th percentile and
maximum discrepancies; it is not a solver-conservation validation.

The archive search found no named Fortran source or DS2V/DS2VD input deck under
the recorded filename patterns. The exact producing input and reference
conditions therefore remain missing. Do not silently divide by guessed values,
label plots T/T∞ or p/p∞, or interpret low interpolation error alone as proof
of Mach independence. Displays use the source labels TOV and P.

Reproduce the read-only comparison without executing any archived script:

```bash
python qa/audit_cylinder_source.py --archive /path/to/AllMachNNCylinder.zip \
  --output tmp/cylinder-source-audit.json
```

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
