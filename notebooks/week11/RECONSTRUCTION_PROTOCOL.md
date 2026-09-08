# Week 11 extension: reconstruct first, identify second

This experiment implements the user's selected interpretation of the Ricardo
course material: reconstruct velocity before applying a physical structure
diagnostic. It does not implement SHAP or claim to reproduce Ricardo's research.

## Scientific question

Does a reduction in velocity reconstruction error also improve recovery of
vortex regions? These are different objectives. Spatial derivatives amplify
small-scale errors, and thresholding can turn small diagnostic changes into
large mask changes. A visually smoother contour is not sufficient evidence.

All methods receive the same two-component velocity field reduced by area
averaging from 32 x 78 to 8 x 19, then bilinearly interpolated to the original
ROI. The horizontal reduction is 78/19, not exactly four. The baseline applies
the physical diagnostic directly to this interpolated field. A compact U-Net
learns normalized high-resolution u and v with MSE, after which the same
diagnostic is applied. A second U-Net predicts the vortex mask directly with
binary cross entropy. This last model has a different output head and objective;
parameter counts and losses are reported separately.

## What constitutes a vortex reference?

For A = grad(u,v), the swirling strength is the imaginary part of its complex
eigenvalues: lambda_ci = sqrt(max(-(u_x-v_y)^2/4 - u_y*v_x, 0)). Pure rotation
has positive lambda_ci; simple shear does not, despite nonzero vorticity.
Derivatives use the same second-order finite-difference operator for native
ROI and reconstructed velocities. We deliberately do not mix this operator
with the archive's vorticity, which was computed before spatial subsampling.

The binary reference uses a fixed threshold: the 90th percentile of interior
training-field lambda_ci. The two-cell ROI margin is omitted from scores.
This is agreement with a velocity-derived weak reference, not an independent
human annotation or proof that all dynamically important structures are found.
The crop boundary is not a wall. Coordinates are x/D,y/D; velocities u/U,v/U;
lambda_ci, vorticity and divergence have units U/D.

## Frozen comparison

Complete Re90 and Re110 cases train the models. Re100 selects the checkpoint
with minimum validation loss and the direct model's probability cutoff on a
fixed 0.1--0.9 grid. Re105 is evaluated only after these choices. It has been
inspected in earlier course work: call it a retained test, not a blind test.
Normalization uses training fields only. Three seeds (17,29,43), 60 epochs,
batch size 32 and Adam at 0.001 are fixed for both tasks. Epoch losses and
selected checkpoints are retained. No best-test seed is selected.

Report velocity relative L2, vorticity relative L2, divergence RMS and
frame-macro Dice/IoU. Direct segmentation has no predicted velocity: field
errors are not applicable. Temporal frames are correlated; seed spread is
optimization variability, not a confidence interval over independent flows.
The display uses the midpoint frame and first seed, chosen in advance.

## Provenance and limits

The data are previously generated author LBM fields, described in
[the data provenance](../../data/modal_labs/README.md). The source has only
12 lattice nodes per diameter and is not grid-converged DNS. These are
incompressible wake fields without shocks. This extension assesses vortex
recovery, not shock detection. The original Week 11 shock controls and separate
research checkpoint gallery remain distinct evidence categories.

The user-supplied course archive `Lesson 5 - Unets and XAI` contains a
single-component turbulent-field super-resolution notebook with area reduction,
bilinear interpolation, training normalization and a U-Net. We independently
implement that methodological idea in PyTorch with two velocity components
and a smaller two-level network. We do not redistribute the course code,
checkpoint or figures, nor claim its numerical results transfer to this case.
Archive SHA-256:
`7cea9030eea601cd952ce3e24566e579b550029c69c4ec95493e3f57a5c0855f`.
Attribution within that archive is not independently established; the folder
name is not evidence of sole authorship or an open-source license.

The new implementation and explanatory text are AI-assisted. Analytic controls,
complete-case separation and saved-checkpoint score recomputation are technical
verification steps; they do not constitute independent human validation of masks.

Architecture reference: Ronneberger, Fischer and Brox (2015),
[U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).
U-Net is the architecture, not a separate competing 'Ricardo method'. The
meaningful comparison is reconstruction followed by physics versus direct
segmentation, using matched observations and case splits.

## Reproduction

Install CPU PyTorch into the project environment, then from the repository root:

```sh
python qa/run_week11_reconstruction.py --output output/week11_reconstruction_new
```

The destination must not exist. Retained evidence is never overwritten. This
is a spatial reconstruction experiment, not temporal forecasting. Before
research claims, repeat across independent higher-resolution flows, degradation
operators and physically justified diagnostic thresholds, with an independent
structure reference and shock-containing compressible fields where appropriate.
