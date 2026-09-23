# What this lab reproduces, and what it does not

The original journal papers are:

1. Zheng, Q., Liang, Y., Yang, Y., and Pan, C. (2026). *Research on low-drag low-boom supersonic transport configuration using an MDO framework and deep learning methods*. Aerospace Science and Technology 178, 113218. https://doi.org/10.1016/j.ast.2026.113218
2. The earlier Chinese aerodynamic optimization paper: https://doi.org/10.7638/kqdlxxb-2025.0081 . This is a distinct publication; do not attribute the later paper's neural dataset to the earlier paper.
3. NASA First Sonic Boom Prediction Workshop SEEB-ALR benchmark: https://lbpw.larc.nasa.gov/sbpw1/test-cases/seeb-alr/ ; public original files: https://lbpw-ftp.larc.nasa.gov/lbpw1/ .

## Compare the scientific tasks

| Item | Zheng et al., 2026 | FlowMLLab Week 16 |
|---|---|---|
| Geometry | Full aircraft, wing/fuselage/canard/V-tail; 36 descriptors | Two-parameter axisymmetric body with fixed volume |
| Design condition | Mach 1.8, lift controlled through incidence | Mach 1.8, zero incidence |
| CFD | FAST01 Euler with boundary-layer correction; STAR-CCM+ SA RANS benchmark comparison | Gmsh and SU2 axisymmetric Euler |
| Learning dataset | 3,480 Latin hypercube samples | 44 cases: 24 training, 6 validation, 8 test, 6 extrapolation |
| Learned maps | Two forward MLPs, geometry to near field to ground; two inverse MLPs | POD coefficients and pressure drag learned from two geometry parameters |
| Acoustic propagation | Augmented Burgers equation and Mark VII PLdB | Not implemented; near-field pressure only |
| Shared validation idea | SEEB-ALR reference case in the CFD methodology | Original SEEB-ALR data provided separately from the learning dataset |
| Data availability | Author data available on request | Original compact evidence, code and audit predictions distributed |

The paper's 80/20 training/validation split is not the same as the lab's separate training, validation and test sets. Our architecture audit uses eight held-out geometries recomputed on a finer mesh. It is a test of a refit of our published architecture, not the original authors' neural weights.

## What a faithful reproduction still requires

A faithful aircraft reproduction needs the author's geometry generator/CAD, precise design-variable definitions and bounds, all CFD and near-field records, train/validation identities, trained weights or complete training configuration, and atmospheric propagation settings. The paper's parameter table contains definitions that should be clarified rather than guessed. A surrogate that reproduces our axisymmetric CFD cannot establish accuracy for their complete aircraft.

No result in this lab establishes a reduction in ground perceived loudness, validates a flight demonstrator, or establishes that a reported TMS-10 test used the identical research implementation. The lab is a reproducible teaching study inspired by the methodology.

## Why three comparisons are necessary

Taylor-Maccoll checks the conical surface pressure. NASA's SEEB-ALR compares an off-body pressure wave with experimental measurements and a separate numerical solver. The neural test compares predictions with CFD at unseen geometries. Passing one comparison does not imply that the other two pass. See [NASA guide](NASA_REFERENCE_GUIDE.md) and [model guide](MODEL_VALIDATION_GUIDE.md).
