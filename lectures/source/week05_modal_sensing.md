# Week 5 companion: sparse sensing and sparse dynamics

## Learning objectives and provenance
Reconstruct a wake from a few velocity observations; distinguish sensor placement from model fitting; identify a low-dimensional ODE without claiming to discover Navier-Stokes. Prerequisites: SVD/POD, least squares, training/validation/test splits. Suggested lesson: 75 minutes plus the executable notebook.

The fields are original FlowMLLab LBM data released in cylinder-cfd-v1. They were generated before these teaching experiments. They are not new article results and are not copied from the four inspirational repositories. The low-Mach wake is different from the rarefied hypersonic cylinder study. There is no shock in this example.

## From a basis to observations
For a centered snapshot q, approximate q = mean + Phi a. Phi has r orthonormal columns fitted only on training fields. A selector C samples k rows: y = C(mean + Phi a) + noise. Recover a by least squares with C Phi, then reconstruct the entire field. Full column rank is necessary, but not enough: conditioning controls noise amplification.

For a state in the POD span, coefficient perturbation is bounded by norm(pseudoinverse(C Phi)) times the measurement perturbation. With truncation residual r, include its sampled contribution C r as well: field error is bounded by norm(r) + norm(pseudoinverse(C Phi)) times (norm(C r) + norm(noise)). More sensors need not remove representation error.

## Position selection
Pivoted QR of Phi-transpose selects the first r independent rows. For k greater than r, this implementation adds rows by a D-optimal determinant-gain criterion. It does NOT use arbitrary trailing QR pivots as an oversampling rule. Compare against five seeded random layouts and report every seed.

---
# Week 5: the controlled sensing experiment

## What is held out?
Fit rank-eight POD to complete Re90 and Re110 trajectories of transverse velocity v/U. Use Re100 only to select among 8, 16 and 32 sensors. Select the smallest budget whose mean validation relative L2 is at most 5%; if none qualify, select the best validation budget. Evaluate the frozen choice on retained Re105.

Noise is artificial measurement noise with standard deviation 1% of training-field RMS. The CFD was not rerun with five physical noise realizations. Generate a full-grid noise array per seed and share it between optimized and random layouts, so overlapping observations have identical perturbations. Seeds are 10 through 14.

## Reading the results
The retained run chooses 16 sensors. Compare optimized and random placements at the SAME budget; show all five scores, not only their mean. Inspect the condition number and full-field projection floor. That oracle sees the complete target field and is therefore a representation diagnostic, not a deployable sensor method.

ROI-edge error is measured on the edge of the extracted wake rectangle. Calling it no-slip error would be incorrect. Integrating v does not produce a mass-flux balance. The equal sample-area weights are a declared quadrature convention, not a conservative finite-volume budget.

## Student checkpoint
Explain why selecting positions using the test POD leaks information. Predict how a poorly conditioned 8-sensor system responds to 1% observation noise. In a separate exploratory run, remove one sensor and report the change in rank, condition number and error; do not overwrite the retained comparison.

---
# Week 5: an interpretable ODE, with limits

## Sparse identification in POD coordinates
Use only the first two Re110 training POD coefficients and divide by their training standard deviations. The candidate library contains a constant and all monomials through degree three. Seek dz/dt = Theta(z) Xi with a sparse coefficient matrix Xi.

Avoid pointwise differentiation here: integrate each library term over four training intervals using the trapezoidal rule, and regress the observed coefficient increments. Normalize regression columns, solve least squares, remove physical-coordinate coefficients below a threshold, and refit retained terms. Repeat until stable or 20 iterations. Thresholds 0.01, 0.05 and 0.1 are selected only by validation field error.

## Autonomous validation, not teacher forcing
Initialize at the last training frame and integrate forward without future observations. Test integration starts from that SAME training state, continuing through the validation interval; it never resets to a true test state. A validation-failed candidate is rejected; a test-failed selected candidate remains a test failure, never replaced by a better test candidate.

## Why a good orbit can give a poor field
The retained rank-two field projection floor is about 36%. The selected sparse ODE is close to that floor, not competitive with an eight-mode field model. One threshold diverges. Keep the failure, the discovered equations, the field error and the truncation floor together.

A single periodic orbit cannot uniquely constrain all cubic terms away from that orbit. Sparse coefficients are not evidence of unique physical governing equations. New initial conditions, parameter cases, noise experiments and longer horizons would be needed before making broader claims.

---
# Week 5: assessment and sources

## Deliverables
Submit the train/validation/test inventory, sensor coordinates, all seeded errors, condition numbers, ODE terms, validation selections, and a limitation paragraph. Separate observation noise, truncation error and dynamics error. Keep retained evidence read-only and save exploratory variants under tmp/.

## Questions
1. Derive the linear least-squares sensor reconstruction and its sensitivity bound.
2. Why can 32 sensors still have nonzero error even without measurement noise?
3. Why does a two-mode orbit not identify unique off-orbit cubic dynamics?
4. What additional experiment would test transfer across Reynolds number?
5. Why are five observation-noise seeds not five independent CFD cases?

## Original implementation and attribution
This companion uses independently written textbook algorithms, not wrappers or a full reimplementation of the cited packages. PySensors inspired the sensor-selection comparison; PySINDy inspired sparse dynamical-system identification. No code or images were copied. Refer to the upstream packages for maintained advanced implementations.

PySensors: https://github.com/dynamicslab/pysensors

PySINDy: https://github.com/dynamicslab/pysindy

Related DMD methods: https://github.com/PyDMD/PyDMD

Evaluation inspiration: https://github.com/pdebench/PDEBench

Data: https://github.com/Ehsan-Roohi/FlowMLLab/releases/tag/cylinder-cfd-v1

Executable companion: notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb
