# FlowMLLab v1.13.0

Actual cavity CFD now covers all seven Ghia Reynolds numbers on three grids.
The release adds 45 native fields over 15 Reynolds values, a new executed
CFD-validation lab, and a separate full-field POD-DeepONet lab with concrete
architecture, optimizer, weighting and physical-error exercises. Original
notebooks and original lecture pages are preserved.

The model holds out Re=3200 and 7500 in full. The validation-selected two-layer
Adam-to-L-BFGS ensemble has vector errors 0.0951% and 0.0844% against 512-cell CFD,
versus 0.5394% and 0.3139% for linear interpolation in log Re. Both interpolation
conventions and all candidate seeds are retained. Depth and per-mode weighting
also produce deterioration; the course reports it.

A documented validation-only amendment added training Re=500 and 750 after
the first POD basis showed a 3.251% projection floor at validation Re=400.
Test scoring occurred only after the amended predictions were frozen.

The reference fields are converged discrete steady solutions. At the two
tests, 256-to-512 mesh differences remain 2.03% and 4.72%; tiny neural error
does not establish mesh-independent physical accuracy or stability.
The nonlinear transport residual remains distinct from exact discrete
continuity. No pressure prediction is claimed.

Six Week-4 lecture pages explain these methods, results and limitations.
The homepage retains the earlier course gallery and adds the new CFD/Ghia
and held-out physical-field comparisons. The literature review identifies
FFV-PINN's exact seven-case coverage and distinguishes it from parametric
learning and the Roohi-McDevitt deep-cavity assessment.

Validation includes independent native-field equations and SHA256 checks,
analytic Jacobian/neural-gradient/POD tests, preserved source-page checks,
new notebook execution, and the release CI/wheel gates. CI-generated receipts
are attached to the release. Remaining numerical limitations are documented
in the protocols and result pages.
