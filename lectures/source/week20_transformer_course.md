# Week 20 | Sensor-to-field learning with missing observations

## Formulate a comparison before fitting

The scientific question is whether a set-based attention model is useful when the available sensor set changes. A comparison on one fixed, noiseless sensor vector mainly tests function fitting on that particular vector. It does not exercise the cardinality flexibility often used to motivate attention.

We therefore compare complete observations with a predetermined half-sensor dropout condition. The protocol fixes the training cases, validation case, retained evaluation case, sixteen QR/D-optimal sensors and observation noise. The evaluation condition is declared before looking at its results. These are already-inspected teaching cases; the report does not label Re105 a newly blind test.

## Four estimators, one target representation

All learned reconstructions use the same training-fitted rank-eight POD decoder. The Sensor-set model embeds normalized value-coordinate triples, applies self-attention and uses learned invariant pooling to produce modal coefficients. The release POD-DeepONet uses a fixed sixteen-input branch network with a 208-unit hidden layer and the same fixed POD trunk. Its 5208 parameters approximately match the attention model's 5192. Ridge maps the same sensor vector linearly to coefficients. Gappy POD solves the observed basis directly.

$$ \hat u=\bar u+\Phi\,\mathrm{diag}(s)\,f_\theta(y,x) $$

This classroom POD-DeepONet is not a reproduction of every published DeepONet variant. Its shared fixed decoder makes representation capacity explicit. Neither model predicts pressure, enforces a no-slip boundary nor learns a general full-domain Navier-Stokes solution. The wake ROI excludes the cylinder.

| Estimator | Handles available coordinates directly? | Missing-observation policy |
| --- | --- | --- |
| Sensor-set attention | Yes | Remove value-coordinate tokens together |
| Fixed POD-DeepONet branch | No | Impute missing normalized values by zero |
| Ridge branch | No | Same training-mean imputation |
| Gappy POD | Yes | Solve only the available basis rows |

## What the dropout experiment can establish

The set model is trained with variable-cardinality augmentation. Fixed-vector controls retain their original interfaces and receive declared imputation at evaluation. This is a comparison of systems with different adaptation mechanisms, not a capacity-matched architecture-only ablation. Parameter count, training time and the augmentation rule are part of the record.

A classical method remains a strong comparator because gappy POD naturally accepts a subset of available sensors. Missing observations do not imply that attention should win. A network may be permutation invariant yet inaccurate, or robust to dropout yet worse than a linear estimator at every cardinality. Those outcomes should remain in the table.

The current release tests complete and fixed half-sensor conditions. Arbitrarily relocated sensors and irregular geometries are separate extensions, not implied successes. To study relocation, train with coordinate variation and freeze a new-location evaluation protocol; do not relabel the existing dropout experiment.

## Optimization is an experimental variable

The old 180-step comparison selected all neural checkpoints at its final evaluation point. The revised protocol allows a larger ceiling, records full training and validation traces, and uses validation patience. Each record includes selected step, executed steps, maximum allowed steps, parameter count and wall-clock fit seconds.

A selected checkpoint at the ceiling is a budget warning. It does not logically prove undertraining: the final point might be the true optimum of the observed interval. Nor does selecting an earlier point prove convergence. Inspect curve shape, then perform a predeclared budget sensitivity study using validation for all choices. Never extend only the model whose evaluation score looked disappointing.

$$ \theta^*=\arg\min_{\theta_k\in\mathcal{C}}\mathcal{L}_{validation}(\theta_k) $$

The set C consists of recorded candidate checkpoints. Evaluation fields are absent from the fitting interface. A poisoned-evaluation test should leave training weights and selected checkpoint unchanged. This is stronger evidence than checking that a constant dictionary contains the intended case numbers.

## Interpret field error relative to the decoder floor

For an orthonormal basis and a prediction inside its affine span, the error decomposes into orthogonal representation and in-subspace components. Their squared norms add; their unsquared norms generally do not. Use the same truth norm as denominator for all three quantities.

$$ \|\hat u-u\|^2=\|u_r-u\|^2+\|\hat u-u_r\|^2 $$

Suppose the representation floor is 4% and the learned in-subspace error is 3%. Total field error is 5%, not 7%. Reducing the learned component to 1% changes the total only to about 4.12%. A small change in field error can therefore conceal a large improvement in coefficient prediction, while a near-floor result leaves little room for further benefit without changing representation.

The equality relies on orthogonality in the metric being used. Here the spatial grid has uniform cell area, so unweighted Euclidean norms have the same relative weighting as area-weighted L2. On a nonuniform mesh, use the corresponding weighted orthonormal basis and inner product.

## Worked implementation and student responsibilities

The notebook trains a short 120-step model as an interactive exercise and labels that budget explicitly. It then loads the full retained comparison, reconstructs a stored predicted field and independently recomputes its relative error. The short classroom run must not be mistaken for the release evidence.

Four tasks implement consistent token dropout, count trainable parameters, diagnose a selected-at-ceiling record and select a validation winner under a time budget. The parameter-count check uses a 16-64-8 branch with biases: 16 times 64 plus 64 plus 64 times 8 plus 8, giving 1608 parameters. The selection task intentionally contains no evaluation score.

## Reporting and reading

Report every seed and condition; show validation curves; distinguish model size, update count and measured time. Add the decoder floor and a separate error color scale to field figures. State whether dropout occurred during training. A conclusion should name the exact condition and comparator rather than saying that attention is universally better or worse.

Course background: the Week 4/5 POD-DeepONet and modal-sensing laboratories. Architecture context: Lu et al., Learning nonlinear operators via DeepONet, https://doi.org/10.1038/s42256-021-00302-5; Wu et al., Transolver (2024), https://arxiv.org/abs/2402.02366. The latter is a research comparison for discussion, not an architecture claimed reproduced by this compact POD-decoder experiment.

## Printable laboratory worksheet

### Task 1: Cardinality-preserving token dropout

Remove every other sensor from values and coordinates together; never reinterpret a missing value as an observed zero.

Show the retained indices and verify that coordinates and observations use exactly the same subset. Explain why a missing normalized observation imputed as zero corresponds to the training mean, not physical zero velocity. Distinguish that fixed-vector baseline from dropping a token in the set model.

### Task 2: Capacity accounting

Count trainable parameters; compare the attention model with the 16-64-8 branch network.

Count biases as well as weights, and distinguish all parameters from trainable parameters. Verify the illustrative 16-64-8 branch count of 1608. Then inspect the release models: their approximately matched parameter counts use a wider branch. Parameter matching alone does not match training augmentation or FLOPs.

### Task 3: Training budget diagnosis

Flag a record whose selected checkpoint reaches its ceiling. This diagnoses an unresolved budget limit, not proof of convergence or its absence.

Plot training and validation losses against executed steps, marking the selected checkpoint and the ceiling separately. A selected step of zero means adaptation was rejected by validation; it is a legitimate outcome to report. Examine the separate doubled-budget audit before claiming that a small field-error change proves convergence.

### Task 4: Compute-aware ranking

Select the best validation record under a wall-time budget; evaluation metrics must not enter selection.

Choose the eligible checkpoint using only validation scores and the stated time budget. Report hardware and timing conditions. Do not compare an isolated process with a heavily contended run as if their speed difference came from the architecture. Retained test scores may describe the chosen model but cannot choose it.

## Before submitting

Restart the kernel and run the worked examples and completed tasks. Confirm that all four coding checks pass and that no retained data or results file changed. Include the requested interpretation, identify every quantity fitted from training data, and separate test observations from decisions made using validation. Record unresolved budget or representation limits rather than removing inconvenient runs.
