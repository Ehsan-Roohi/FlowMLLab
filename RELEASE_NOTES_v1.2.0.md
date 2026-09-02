# FlowMLLab v1.2.0

FlowMLLab v1.2.0 adds a complete educational pathway from lattice-Boltzmann
mechanics to Reynolds-dependent cylinder wakes and leakage-safe field learning.

## New in v1.2.0

- adds the installable `flowmllab.cylinder_lbm` D2Q9 solver, with transparent
  BGK collision and a stable TRT default;
- adds Zou--He inflow, a convective outlet, periodic transverse boundaries,
  Bouzidi interpolated bounce-back on the analytical circular cylinder,
  momentum-exchange forces, pressure, vorticity,
  recirculation-length, and Strouhal diagnostics;
- adds quick classroom and expensive qualification parameter profiles;
- retains executed `Re=5,20,40,100,180` regime evidence, reference ranges,
  metadata, and publication-quality PNG/PDF figures;
- adds `W5_Lattice_Boltzmann_Cylinder_Student.ipynb`, including D2Q9/TRT
  derivations, regime classification, validation ladders, and refinement tasks;
- adds `flowmllab.cylinder_ml`: centered multi-field POD, complete-Re splits,
  a deterministic Reynolds/phase baseline, reconstruction metrics, and physical
  diagnostics;
- adds a two-layer neural POD exercise with `Re=100` withheld as a complete
  physical case, conceptually motivated by Lee & You (JFM 2019); and
- retains a 1080p LBM-versus-neural vorticity animation for a completely
  withheld `Re=100` trajectory, with training Reynolds cases, phase
  conditioning, errors, the non-neural baseline, and claim limits recorded;
- adds `flowmllab cylinder --root .` to verify retained Week-5 evidence.

## Validation contract

The retained quick sweep must remain finite and low-Mach, keep mean-density
drift below 1%, and distinguish attached flow (`Re=5`), steady recirculating
wakes (`Re=20,40`), and periodic shedding (`Re=100,180`).  Quantitative drag,
recirculation, lift-RMS, and Strouhal reference bands are reported but are not a
hard gate for the quick grid.  They become mandatory only for the separate
qualification profile after grid, domain, Mach, and sampling-time refinement.

This distinction is deliberate.  Finite curved-wall resolution, finite
blockage, periodic transverse images, and short classroom runs can preserve the
regime while biasing force and recirculation values.  `Re=180` is explicitly a
two-dimensional teaching case; Mode A/B are outside this solver.

## Release checks

Before tagging, run:

```bash
python -m unittest discover -s tests -v
flowmllab smoke --root .
flowmllab cylinder --root .
flowmllab qa --root .
python -m build
python -m twine check dist/*
```

Version 1.2.0 supersedes v1.1.0 for current source use.  The v1.1.0 Zenodo DOI
remains the latest archived identifier until the v1.2.0 GitHub release is
archived; no new DOI is claimed here.
