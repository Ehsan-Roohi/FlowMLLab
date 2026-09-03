# Week 8 retained gas-dynamics evidence

This directory is the small, offline teaching snapshot used by the FlowMLLab
Week-8 lecture and notebooks. The authoritative implementations and complete
evidence remain in
[`GasDynamicsSciML`](https://github.com/Ehsan-Roohi/GasDynamicsSciML) at commit
`374431a1033138f56e2752bf8bbf9b75a454d80c`.

The copied CSV and JSON files are byte-identical to the source files listed in
`provenance.json`; their SHA-256 values are recorded there. The FlowMLLab QA
gate verifies those hashes and the headline numerical claims.

The teaching interpretation is also aligned with the author-supplied revised
manuscript *Physics-Guided Neural Surrogates for Canonical Compressible
Thermal-Fluid Relations* (`AITF-D-26-00044R1`). Its PDF and Overleaf-source
hashes are recorded in `provenance.json`; the files themselves are not
redistributed. FlowMLLab does not infer editorial acceptance from a revision
identifier.

## Interpretation

- All five physics-guided inverse models have retained blind relative-L2 error
  below `3.5e-3` and full physical-domain coverage.
- Classical interpolation remains the preferred baseline on covered
  one-dimensional tables when it is both faster and more accurate.
- At a matched roughly 4096-state budget, the five-input shock-tube audit has
  `4.88496%` interpolation relative-L2 error versus `0.177106%` for the MLP.
- The 100,000-state shock-tube application audit records `16.53x` speedup over
  one bracketed root solve per state with `0.09148%` relative-L2 error.
- These results do not validate reacting flow, variable heat capacity,
  multidimensional shocks, unrestricted extrapolation, or high-fidelity CFD.

The separate `SU2-Diamond-Airfoil-Verification` repository is cited only as a
bridge to multidimensional CFD. At its frozen Week-8 source commit, only the
sharp-wall `euler_alpha0` case is a qualified teaching reference; the other
eight cases remain unverified.
