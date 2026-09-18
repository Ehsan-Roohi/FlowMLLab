# Week 15 — Neural operators under geometry change

## 1. The scientific question

FlowMLLab / Ehsan Roohi

Can a neural operator trained on several separated-flow geometries predict velocity, pressure, and recirculation when the geometry changes? This lecture uses retained OpenFOAM fields and three-seed predictions. It does not replace missing runs with synthetic contours.

Learning outcomes: formulate DeepONet, Geo-DeepONet, FNO, and U-FNO; design whole-geometry and whole-family splits; read CFD/prediction contours with common scales; distinguish velocity, centered-pressure, and reverse-flow metrics; and recognize topology extrapolation failure.

## 1.1 Definitions and notation

Operator learning: learning a map from an input function or parameter set (here the geometry and the Reynolds number) to an output field (velocity and pressure everywhere in the channel), as opposed to a map from one vector to one number. Signed distance field (SDF): at each grid point, the distance to the nearest wall, negative inside the solid and positive in the fluid; it is one way to give a network the geometry as a field. Mask: a 0/1 field marking solid and fluid cells.

DeepONet: q-hat(xi) = sum over k of b_k(a) t_k(xi) + c. The branch network turns the input a (a parameter vector, or a function sampled at fixed sensors) into p numbers b_k; the trunk network turns a query point xi = (x, y) into p numbers t_k; the prediction is their inner product. The rank p (48 in this project) is the number of terms. Worked example with p = 2 on a 2-by-2 grid: if the branch returns b = (1, 0.5) for one geometry and the trunk returns t = (x, y) at each of the four points (0,0), (1,0), (0,1), (1,1), the prediction is 1*x + 0.5*y, that is 0, 1, 0.5, 1.5 at the four points. The trunk therefore fixes a spatial basis that is the same for every input; only the coefficients b_k change with the geometry. That is the origin of the fixed-domain limitation discussed in Section 3.

Fourier neural operator (FNO): a network whose layers act on a whole field v_l on a regular grid: v_(l+1) = sigma(W_l v_l + F^-1(R_l F(v_l))). F and F^-1 are the discrete Fourier transform and its inverse over the grid, R_l multiplies the lowest Fourier modes by learned complex weights (a learned convolution), W_l is a pointwise linear map across channels, and sigma is a nonlinearity. U-FNO adds a U-Net-style local branch beside the Fourier branch.

Error metrics: velocity relative L2 error, 100 ||u-hat - u|| / ||u|| over fluid cells; centered-pressure error, the same quantity after the mean of each pressure field over the fluid has been removed separately; reverse-flow IoU, the intersection-over-union of the sets where u < 0 in prediction and CFD (1 is perfect agreement, 0 is no overlap). A relative error above 100% is possible whenever the denominator is small, which is the case for pressure fluctuations in low-speed separated flow: an error of 299% means the prediction error is three times the size of the pressure variation itself, not that a percentage was miscomputed. Vortex-region error is the velocity error restricted to the recirculation zone; the same remark applies.

## 2. Data and split discipline

[SPLIT]

The dataset contains 130 accepted sampled OpenFOAM fields, 51 distinct masks, Reynolds numbers 25, 50, and 100, and a 60×300 common grid. Every case stores u, v, pressure, coordinates, a fluid mask, and signed distance. All Reynolds-number variants of a geometry must remain in one split.

The earlier g011 audit uses 106 training, 21 validation, and 3 test cases and compares Geo-DeepONet, FNO, and U-FNO at seed 17. It is useful for metric verification but no longer counts as unopened evidence.

## 3. Vanilla DeepONet and its geometry limitation

A DeepONet writes q-hat(xi) = sum_k b_k(a)t_k(xi)+c. The branch embeds an input function or parameter vector; the trunk embeds the query coordinate. Their inner product forms the output field.

If a contains only Reynolds number or boundary values, the trunk basis remains tied to the training domain. A changed obstacle changes the valid coordinate set and boundary conditions. Ordinary DeepONet has no automatic geometry equivariance, so good fixed-geometry interpolation is not evidence of shape transfer.

Geo-DeepONet supplies a mask, signed-distance field, or learned geometry representation to the operator. This makes geometry visible; it does not guarantee extrapolation to a missing topology.

[VANILLA]

FlowMLLab already contains the ordinary DeepONet implementation in `qa/step_architecture_v5.py`: step height enters a two-layer 128-wide tanh branch, normalized x/y enter a two-layer 128-wide tanh trunk, rank is 48, and the contraction returns u and v. The notebook prints and structurally verifies that exact builder instead of presenting only an equation.

The retained V5 DSMC height experiment used three seeds and matched sampling schedules. Ordinary DeepONet reached mean terminal global errors of 14.01% (uniform) and 23.31% (zonal), with vortex-region errors of 190.44% and 93.41%; no seed passed the predeclared checkpoint ceiling. This is direct project evidence of the limitation, but it is a different rarefied-flow dataset and is not relabeled as an OpenFOAM result.

Read Figure 2 carefully before drawing the lesson from it: on that DSMC height family the plain coordinate MLP, which receives eight known geometry features together with the coordinates, has the lowest global error of the three models, and Geo-DeepONet is not better than ordinary DeepONet on the zonal sampler. Two things follow. First, "geometry-aware" is a property of the inputs, not of the name: an MLP fed explicit geometry features is geometry-aware in the sense that matters here, and a branch/trunk operator fed only a scalar height is not. Second, the V5 study varies one parameter (step height) within one family, which is a task a coordinate MLP can interpolate; the OpenFOAM protocols of Sections 5 to 8 vary the mask itself, which is where the operator formulations are tested and where the MLP was not run. The figure supports the narrow claim that a height-only branch fails the checkpoint gate; it does not show that operator architectures beat simpler models, and this lecture does not claim it.

## 4. FNO and U-FNO

An FNO layer has the form v_(l+1) = sigma(W_l v_l + F^-1(R_l F(v_l))). The learned low-frequency spectral multiplier is efficient on a common grid. Masks, sharp walls, pressure gradients, and new topology still produce out-of-distribution structure.

U-FNO adds a local U-shaped pathway to Fourier blocks, improving access to multiscale local features. In the retained archive U-FNO is available only for g011. It was not run in the two three-seed OpenFOAM generalization protocols; the lecture does not invent that comparison. Ordinary DeepONet is now trained separately on the OpenFOAM geometry-holdout split with Reynolds number in the branch and fixed x/y in the trunk, but without mask, SDF, or geometry ID.

## 5. What was trained and what was tested

[POINTS]

Protocol A, geometry holdout: 107 training cases, 11 validation cases, and 12 tests from unseen g009, g023, g036, and g048. Protocol B, family holdout: 103 training, 8 validation, and 19 tests from the excluded double-step family g012 and g045 through g051. Geo-DeepONet and FNO were run for 400 epochs with seeds 17, 29, and 43. The new ordinary-DeepONet baseline uses the same Protocol-A counts, tests, epochs and seeds; its explicit 11-case validation grouping is recorded with the run.

Across the 12 geometry-holdout tests, ordinary DeepONet gives 28.96% mean velocity error, 299.57% mean centered-pressure error, and 0.406 mean reverse-flow IoU. At g009/Re100, all three seeds give about 46.6–47.1% velocity error. These are executed OpenFOAM-dataset results, not the older DSMC V5 scores.

Exact test identities are committed in case_metrics.csv. The separately frozen train-versus-validation identity lists and checkpoints were not recovered, so they are not guessed. The development pool is the complement of the named tests, but membership of its train and validation subsets must be restored before claiming full retraining reproducibility.

## 6. Read fields, not only scores

[TRAIN]

The blue panels are confirmed ordinary-DeepONet training geometries; red g009 is excluded at every Reynolds number. Exact historical Geo-DeepONet/FNO train-versus-validation identities are unavailable. The full CFD/model fields with streamlines follow immediately below.

[G009]

Rows are CFD, ordinary DeepONet, Geo-DeepONet, and FNO. Columns show speed with each row's own streamlines and independently mean-removed pressure. Common column scales prevent each model from choosing flattering limits, and every model row reports its seed-17 velocity and centered-pressure errors. For g009/Re100, ordinary DeepONet has 46.69% velocity error and 92.25% centered-pressure error, versus 10.25%/19.36% for Geo-DeepONet and 2.81%/10.69% for FNO. It cannot distinguish two unseen masks at the same Reynolds number because geometry is absent from its inputs.

## 7. A velocity win can hide pressure failure

[G048]

At g048/Re50, FNO's velocity error is about 3.62%, while its centered-pressure error is about 142%. Geo-DeepONet's corresponding errors are about 4.58% and 49.0%. A plausible speed field does not certify momentum balance or pressure transfer.

Reverse-flow IoU compares u<0 footprints. It does not identify vortex centers, circulation, reattachment from wall shear, or topological equivalence. Contours and local diagnostics must accompany global norms.

## 8. Whole-family extrapolation

[FAMILY]

When the whole double-step family is withheld, mean velocity errors rise to 18.56% for Geo-DeepONet and 9.16% for FNO. Mean centered-pressure errors reach 74.58% and 255.61%; reverse-flow IoU falls to 0.410 and 0.578. Several Geo-DeepONet cases also show large seed spread.

This is the intended lesson: neither model establishes reliable topology extrapolation. FNO is better on mean velocity here, Geo-DeepONet is far better on mean pressure, and neither ranking is universal.

## 9. Validation checklist

Use whole-geometry splits; fit scalers on training cases only; preserve one pressure-gauge convention; compare identical cases, seeds, and optimization budgets; report per-case mean and standard deviation; inspect worst cases; test flux and boundary consistency; and keep an unopened final family.

The notebook rechecks data hashes, shapes, geometry grouping, metric formulas, pressure-offset invariance, vorticity stencils, and the representative raw fields. The two committed casebooks expose every test case. The original OpenFOAM case directories and solver convergence logs are still needed for independent CFD validation.

## 10. Take-home message and references

Do not reduce geometry generalization to one bar. Show CFD and every model in rows, speed plus streamlines and pressure in columns, shared limits, exact split identities, and failure cases. A model that interpolates Reynolds number on a fixed mask can still fail when the wall moves; a geometry-conditioned model can still fail when topology changes.

Lu et al. (2021), Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators, Nature Machine Intelligence 3, 218–229. DOI: 10.1038/s42256-021-00302-5.

Li et al. (2021), Fourier Neural Operator for Parametric Partial Differential Equations, ICLR 2021, arXiv:2010.08895.

Li et al. (2022), Fourier Neural Operator with Learned Deformations for PDEs on General Geometries, arXiv:2207.05209.

## 11. In-class diagnostic questions

Why can a branch that receives only Reynolds number or step height not distinguish two masks at the same flow condition? Which additional information must be available before calling a model geometry-aware? Explain why adding signed distance helps within represented shape families but cannot guarantee a new topology.

At g048/Re50, decide whether the FNO result is acceptable if the application needs velocity only, surface force, or pressure-driven loading. Use the velocity and centered-pressure errors separately. Then identify what reverse-flow IoU can and cannot say about vortex structure.

Compare ordinary DeepONet with Geo-DeepONet on the same OpenFOAM test geometry. Which field differences follow directly from hiding mask and SDF? Then compare with the older V5 DSMC study and explain why results from the two datasets must remain separate.

## 12. Required student submission

Submit the split map; the exact ordinary-DeepONet code path and architecture table; one g009 and one g048 CFD/model comparison; the geometry- and family-holdout metric table; and a paragraph distinguishing interpolation, unseen geometry, and unseen topology. State every missing artifact needed for a full retraining claim: frozen train/validation identities, scalers, source hash, checkpoints, histories, OpenFOAM case directories, solver convergence, and grid-sensitivity evidence.
