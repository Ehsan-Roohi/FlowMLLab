# Week 15 — Neural operators under geometry change

## 1. The scientific question

FlowMLLab / Ehsan Roohi

Can a neural operator trained on several separated-flow geometries predict velocity, pressure, and recirculation when the geometry changes? This lecture uses retained OpenFOAM fields and three-seed predictions. It does not replace missing runs with synthetic contours.

Learning outcomes: formulate DeepONet, Geo-DeepONet, FNO, and U-FNO; design whole-geometry and whole-family splits; read CFD/prediction contours with common scales; distinguish velocity, centered-pressure, and reverse-flow metrics; and recognize topology extrapolation failure.

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

## 4. FNO and U-FNO

An FNO layer has the form v_(l+1) = sigma(W_l v_l + F^-1(R_l F(v_l))). The learned low-frequency spectral multiplier is efficient on a common grid. Masks, sharp walls, pressure gradients, and new topology still produce out-of-distribution structure.

U-FNO adds a local U-shaped pathway to Fourier blocks, improving access to multiscale local features. In the retained archive U-FNO is available only for g011. It was not run in the two three-seed OpenFOAM generalization protocols; the lecture does not invent that comparison. Ordinary DeepONet is represented by its exact V5 code and retained DSMC results above, while the OpenFOAM field rows remain the models actually present in that prediction archive.

## 5. What was trained and what was tested

[POINTS]

Protocol A, geometry holdout: 107 training cases, 11 validation cases, and 12 tests from unseen g009, g023, g036, and g048. Protocol B, family holdout: 103 training, 8 validation, and 19 tests from the excluded double-step family g012 and g045 through g051. Geo-DeepONet and FNO were run for 400 epochs with seeds 17, 29, and 43.

Exact test identities are committed in case_metrics.csv. The separately frozen train-versus-validation identity lists and checkpoints were not recovered, so they are not guessed. The development pool is the complement of the named tests, but membership of its train and validation subsets must be restored before claiming full retraining reproducibility.

## 6. Read fields, not only scores

[G009]

Rows are CFD, Geo-DeepONet, and FNO. Columns show speed with each row's own streamlines and independently mean-removed pressure. Common column scales prevent each model from choosing flattering limits. At g009/Re100, FNO has the stronger velocity score, but pressure is a separate question.

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

Compare the V5 ordinary DeepONet with Geo-DeepONet using the frozen checkpoint rule, not only terminal means. Why is “no eligible checkpoint” different from a crashed training run? Why must those DSMC results remain separate from the OpenFOAM casebooks?

## 12. Required student submission

Submit the split map; the exact ordinary-DeepONet code path and architecture table; one g009 and one g048 CFD/model comparison; the geometry- and family-holdout metric table; and a paragraph distinguishing interpolation, unseen geometry, and unseen topology. State every missing artifact needed for a full retraining claim: frozen train/validation identities, scalers, source hash, checkpoints, histories, OpenFOAM case directories, solver convergence, and grid-sensitivity evidence.
