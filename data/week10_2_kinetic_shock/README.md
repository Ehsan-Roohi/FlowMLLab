# Week 10.2: kinetic shock profiles and collision moments

This extract supports the CPU lesson in `flowmllab.kinetic_shock`. It contains
saved numerical evidence, not trained models or an executable research solver.
The numerical helper code is an independent FlowMLLab implementation of the
equations. No upstream software code was copied.

Source: Ehsan Roohi, *Physics-informed neural networks (PINNs) for kinetic shock
waves: simultaneous accuracy of lower- and higher-order moments, with and
without moment supervision*, [arXiv:2605.29211v2](https://arxiv.org/abs/2605.29211v2),
revised 13 November 2026. The immutable source repository is
[PINNs-DVM at add5388](https://github.com/Ehsan-Roohi/PINNs-DVM/tree/add5388f7da5f7eace95fff97d34d6ed58b90169).
See NOTICE.md for attribution and licensing boundaries.

## Contents and provenance

`source_files.json` lists every selected public path, byte length, Git blob SHA1,
SHA256, and immutable URL. `upstream/` contains those exact source bytes.
`manifest.json` binds the teaching case, physical assumptions, columns, sample
coverage, observation roles, and hashes. The preparation script checks **both**
Git blob identity and SHA256 before reusing any local source. It never reads a
checkpoint. The source extract is 24 files and 7,483,249 bytes.

The 12 case IDs are:

- `tube_bgk`, `tube_shakhov`: transient smoothed Sod-type shock tubes at
  `t=0.02`, constant `tau=Kn=1`, on `[-0.5,0.5]`. Each has the unchanged 3,200-node
  supplied comparison grid and `parent`, `final`, `reference` fields. The first
  two are equation-trained characteristic PINN endpoints, before and after
  200 further SSB updates. Their references are collision-matched DVM profiles.
- `m2_seed141001_adam20000`, `m2_seed141001_ssb500`,
  `m2_seed141001_fixed500`, with the same three stages for seeds 141002 and
  141003: steady Mach-2 Shakhov shocks, `Pr=2/3`, `tau=1/rho`, 81 actual common
  core nodes across `[-10,10]`. These are three independent initializations;
  stages are sequential and the split-half stage changes quadrature. They
  are not an equal-budget optimizer benchmark. At least one physical
  acceptance tolerance remains unmet for each final endpoint.
- `supervised_s0_current`: the actual five-field S0 baseline, current v2
  eight-moment data-assisted BGK shock tube, an equation-trained BGK endpoint,
  and their 6,400-cell DVM reference. The five-field S0 comes from the exact
  public `STAGE0_FIELDS.npz`, never from the eight-moment 170-update state.
  Their source grids and reference values are checked for exact equality.
  It also retains the correctly named original-envelope and common-tail initial
  states for physical interpretation. It has **399** existing equation-trained
  nodes in actual common coverage. Assessment fields and DVM are linearly
  interpolated to these nodes. Two endpoint nodes outside cell-center coverage
  are excluded. There is no fitted shift, amplitude adjustment, smoothing, or
  extrapolation. Original source arrays remain available to audit this choice.

The supervised extract is a transparent teaching transformation, not a claim to
reproduce the paper's unweighted 401-node Euclidean score. The module's metrics
use a spatial trapezoidal integral on the declared common grid. A relative L2
requires a nonzero reference norm. Signed percentage profiles use one positive
whole-profile amplitude per field, never a pointwise near-zero denominator.

## Data roles and limitations

### Physical conditions and Mach-number interpretation

All tube profiles, including `supervised_s0_current` and the fresh CPU lesson,
use x ∈ [−0.5, 0.5], t = 0.02, constant τ = Kn = 1 in the source scaling,
R_g = 1 and γ = 5/3. The initial density, velocity and temperature are
(ρ, u, T) = (1 − H)(1, 0, 1) + H(0.125, 0, 0.8),
where H(x) = [1 + tanh(20x)]/2. The initial distribution is the local Maxwellian
of those fields. Boundary reservoirs prescribe only incoming molecules;
outgoing molecules leave freely. BGK has Pr = 1; Shakhov uses Pr = 2/3.

Thus p_left/p_right = 10 and ρ_left/ρ_right = 8. Both reservoirs start at rest;
there is **no prescribed stationary upstream Mach number**. Local Mach
|u|/sqrt(γR_gT) evolves from zero. A moving shock's speed-based Mach would
require a separate front-tracking measurement. Neither should be confused with
the stationary M₁ = 2 Shakhov case, whose upstream ρ₁ = T₁ = 1 and
u₁ = 2 sqrt(5/3), with τ = 1/ρ and Pr = 2/3.

### Reference information used by each method

The data-assisted lineage uses 160 locations for three macro fields and 42
locations for five moment fields: 690 scalar reference observations. Exact
locations are retained in `OBSERVATION_LOCATIONS_FROZEN.npz` and accessible
through `load_observation_locations`. Reference-derived scales and some
acceptance guards also used the same reference. Removing observed locations or
their interpolation neighbors does not create an independent test set.

Reference roles are method-specific. The five-field S0 was trained on a
400-cell DVM reference and is reevaluated on a 48x20x20 velocity grid against
the 6,400-cell comparison here. The final eight-moment data-assisted refinement
uses 6,400-cell targets, scales and some guards. The equation-trained method
uses no DVM field targets; its original comparison is against 3,200 cells and
the 6,400-cell reference here is post-solve only. The case-wide contamination
and admission flags explicitly refer to the final data-assisted lineage.

The original `paper_admission: false` and `all_criteria_pass: false` records
remain unchanged. The author elected to report the final state with the
limitations: E4 failed a predeclared 0.1% tolerance on 64 separate physics
points; E8 also failed 0.1% on 64 points unused by all stages, and that criterion
was added after the first assessment. `STATUS_NOTE.txt` provides the exact
provenance. The script-level `held_out_reference_read_by_this_script: false`
must not be generalized to the full data-assisted lineage. None of this extract
certifies continuum convergence or a new independent validation experiment.

## Units and moments

All fields use nondimensional `R=1`, `gamma=5/3` monatomic conventions with three
molecular velocity components. Peculiar velocity is `c=v-u*e_x`.

- `q = 1/2 integral |c|^2 c_x f dv`.
- `sigma = integral (c_x^2-|c|^2/3) f dv` is the kinetic deviatoric stress sign.
- `m = integral (c_x^3-3 c_x |c|^2/5) f dv`.
- `R = integral (|c|^2-7T)(c_x^2-|c|^2/3) f dv`.
- `Delta = integral |c|^4 f dv - 15 rho T^2`; `S=R+Delta/3`.

Columns in original source files are mapped **by name**; the assessment's
`S_raw` is the same `S` combination used here. Eight moment fields comprise
`rho,u,T,q,sigma,m,R,Delta`; dependent `S` is an additional diagnostic.

For steady shocks only, conserved fluxes are `rho*u`,
`rho*u^2+rho*T+sigma`, and `rho*u*(u^2/2+5T/2)+u*sigma+q`.
Constant steady flux diagnostics do not apply to transient shock-tube snapshots.

## Reproduction

From the repository root, with NumPy installed:

```sh
python -B data/week10_2_kinetic_shock/prepare_extract.py
python -B -m pytest tests/test_kinetic_shock.py
```

Preparation downloads only missing enumerated files. An optional
`--local-release PATH` reuses source files only after both identities match the
public pinned bytes. Outputs stay beside this script. This performs no research
training or shock solve. `load_case` verifies each selected file before reading.

The fresh `collision_experiment` is separately labeled: a positive two-Maxwellian
mixture on tensor Gauss-Legendre 3D nodes, not a saved paper result. Its analytic
bulk moments are rho=1, u=0.2, T=1.19, sigma=0.56 and q=-1.071. Increasing order
and cutoff are distinct controls. Invariant defects, heat/stress rate errors,
negative Shakhov target nodes and integrated negative mass are reported.
No target clipping, conservation rescaling or positivity repair is applied.
