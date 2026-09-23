# Week 16 assignment: design, predict, recompute

**Time:** two 75-minute sessions plus 3-5 hours of independent work. CFD data generation is instructor preparation; students use the retained dataset and may reproduce CFD as an extension.

**Prerequisites:** compressible flow and oblique shocks (Week 8), supervised regression, train/validation/test separation, and NumPy. POD from Week 4 is helpful; the notebook includes a short derivation.

## Scientific question

Can a learned model propose a fixed-volume body of revolution with a lower maximum off-body pressure coefficient at r/L=0.5 without increasing pressure drag by more than 2%? Does that proposal survive a fresh Euler calculation and mesh refinement?

## Deliverables and rubric (100 points)

1. **Geometry and governing equations (15).** Derive the volume normalization of the radius function and the axisymmetric Euler source term. Explain why a planar 2D solution changes the physical problem. Plot three admissible shapes with identical axis scales. Check volume by independent quadrature.
2. **CFD evidence (20).** Explain the slip wall, symmetry axis and farfield conditions. Reproduce the cone benchmark comparison from retained data. Tabulate baseline mesh and domain differences. Explain why the number of iterations alone is not a convergence measure.
3. **Learning (20).** Train ridge, POD-MLP and POD-GP models using the frozen training geometry IDs. Use validation only for selecting the design model. Report waveform relative L2, peak error and pressure-drag error separately for test and extrapolation. Explain any case in which a neural model loses to a simpler baseline.
4. **Design (20).** Reproduce constrained optimization, plot the baseline and proposed shapes, and tabulate predicted versus recomputed objectives. Report pressure drag with the documented reference area. A surrogate-only improvement earns no CFD-validation credit.
5. **Stress test (15).** Inspect M=1.7 and M=1.9 reruns and all three extraction radii. Explain which conclusions transfer and which remain condition-specific. Do not call radial-distance checks an atmospheric uncertainty study.
6. **Interpretation and reproducibility (10).** Supply software versions, seeds and the dataset SHA-256. Explain why near-field peak reduction cannot be translated into a PLdB benefit without propagation and a validated loudness calculation.

## Suggested investigations

- Reduce training labels to 6, 12 and 24, keeping validation and test unchanged. Plot error versus number of actual CFD runs, including training cost.
- Replace pressure-peak optimization with pressure-signature L2 optimization. Compare both using their own CFD-confirmed designs.
- Find two parameter vectors with similar predicted pressure signatures and different drag. Check the forward solutions before describing them as equivalent designs.
- Repeat the optimization with a stricter shape-slope or radius constraint.

## Submission

One executed notebook, a two-page engineering memo, and a small CSV of baseline/predicted/recomputed metrics. State unsuccessful designs and failed checks explicitly. Do not upload the paper PDF or claim reproduction of the Beihang aircraft.
