# Matched D/W=5 cavity fields from the authors' retained assessment

Source: Ehsan Roohi and Christopher J. McDevitt,
[Regional accuracy of PINNs for recirculation in a deep lid-driven cavity, v2](https://arxiv.org/abs/2610.38814v2).
The article's ancillary-file list provides the original analysis and figure code.

`reynolds_common_fields.npz` contains x, h and CFD/PINN u,v,p,omega for Re=100,
250,500,1000. All retained arrays are exactly equal to the original author
assessment arrays. Only redundant speed arrays were omitted. The original file
and the compact teaching file have different whole-file hashes for this reason.
`manifest.json` records both hashes and every retained array hash.

Coordinates are x/W and height h/W above the floor, on a common 121x605 grid;
arrays are [h,x]. Velocities use U, pressure uses rho U squared and one frozen
whole-interior gauge, and vorticity uses U/W. The Gaussian-regularized lid and
geometry differ from Ghia's square cavity. No exact wall values are sampled.

The Re=1000 field is the control trajectory. The separate detailed 161x801
workflow comparison is tabulated in `results/week13_d5/workflow_metrics.csv`,
not represented by substituting another field into this NPZ. Preserve the two
sampling conventions when reproducing the manuscript numbers.

This is an author-authorized educational field-assessment bundle. It is not a
complete training restart, optimizer/RNG-state archive or independent-seed
validation. The original data and unique model checkpoints remain in their
research archive. No new blanket license is asserted for inherited third-party
training code. The new teaching/postprocessing code follows the repository license.

[CPU notebook](../../notebooks/week13/W13_Deep_Cavity_D5_Regional_Accuracy.ipynb) ·
[Recomputation code](../../flowmllab/deep_cavity_audit.py) ·
[Results and verification](../../results/week13_d5/README.md).
