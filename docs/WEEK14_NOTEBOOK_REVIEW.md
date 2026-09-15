# Week 14: source audit before classroom release

The supplied `W14_pyCALC_RANS_ML_Closure_Audit.ipynb` is a useful **offline
interpolation exercise**, not an exact reproduction of Davidson's PINN/NN
turbulence-model pipeline. Its input file was preserved unchanged.

## Confirmed corrections

1. **The purported baseline is already corrected.** Its embedded RANS payload
   has SHA-256 `6ee9b045274441fc38c2f43c95c2a2d9f3783a25234812aa0faa3b555eb70bd5`.
   This exactly matches the profile in the upstream
   `channel-5200-half-channel-PINN-vist-over-y-uv_tot-2nd-submission` directory.
   The actual reference baseline is in `channel-5200-half-channel-yfac1.1`.
   The former gives approximately 8.95% pointwise relative L2 error in k;
   the latter approximately 42.85%. These are archived-profile comparisons,
   not evidence of a new solver execution.
2. **The supplied MLP is an adaptation.** It uses log(nu_t/y), StandardScaler,
   tanh, L-BFGS and a contiguous gap. The original c_k script uses raw nu_t/y,
   separate MinMaxScalers, a 2-10-10-1 ReLU network, SGD at 0.04, batch size 1,
   1,000 epochs and an 80/20 random split with random_state=42. The original
   constructs a learning-rate scheduler but never calls scheduler.step().
   Do not silently add a schedule when reproducing that script.
3. **Derivative endpoints differ.** The original training feature uses NumPy's
   default first-order gradient endpoints; the supplied notebook uses order 2.
4. **A gap is not an unopened case.** Targets are displayed before fitting and
   the historical results have already been inspected. Call this a held-out
   interpolation exercise, not a prospective blind validation campaign.
5. **Units need precision.** The stored omega is nondimensionalized with
   delta/u_tau, not the viscous wall-unit omega+. With u_tau=delta=1,
   omega+ equals stored omega / Re_tau. Likewise nu_t/y assumes u_tau=1;
   the deployment feature is nu_t/(y*u_tau).
6. **A coefficient error is not a flow error.** C_k multiplies 0.09*k*omega,
   but the complete method also modifies sigma_k and C_omega2 and couples
   them back into the solver. A frozen-field product cannot establish
   a posteriori stability, accuracy, or geometry transfer.
7. **Credit the primary data producers.** pyCALC-RANS and the model workflow
   are by Lars Davidson; the Re_tau=5200 DNS is by Myoungkyu Lee and Robert
   D. Moser. The lecture explanations, new figures and audit are FlowMLLab
   additions, not original Davidson lecture slides.
8. **Permission wording was too specific.** The shared conversation records
   the instructor saying permission was granted, but does not show the
   original written authorization or its precise date/terms. Do not invent
   these details or apply FlowMLLab's software license to upstream material.

## Upstream packaging issue

The public archive linked from Davidson's 11 September 2026 update has SHA-256
`7aae30d0e990e78ac008c03d008ddc7609ce3a4aae9cc55fada65f1be91d774f`.
Its `nn10000/modify_case.py` equivalent includes the two documented fixes,
whereas the distributed `exec-pyCALC-RANS.py` still contains the old `y_0`
column and `prand__k_ML` spelling. Reassemble from the corrected source files;
do not execute the stale concatenated file and assume the fixes are active.

## Additional execution finding: released targets are not regenerated

### Distinguish a package-stage comparison from the paper's final model

The reported 1.726% to 4.988% velocity-error change compares the classical
baseline with the distributed **table-based PINN** stage. The final
**PINN-NN** model is a different deployed closure. Davidson's Section 5.1
and Figure 8 describe good velocity profiles and improved k at Re_tau=5200;
the explicitly mentioned velocity overprediction is at Re_tau=550.
Our percentages are newly calculated metrics, not numbers quoted by the paper.
They must not be presented as verified agreement with, or contradiction of,
the final-model claims without matching the figure, model and run state.

See the [paper-alignment ledger](WEEK14_PAPER_ALIGNMENT.md), including the
observed difference between released scaler bounds and current training inputs.

Executing the original balance script against the bundled inputs does not
recreate the released coefficient tables. Maximum absolute differences are
approximately 0.524585 for the two-column C_k table, 0.019786 for C_omega2,
and 0.298867 for smoothed sigma_k. The rerun is isolated from the retained
published targets and checkpoints. This is a reproducibility gap in the
distributed state, not proof that the paper's physical conclusions are wrong.
Do not silently replace the targets, retrain, and call that the same experiment.

## Primary references

- Davidson's [source distribution and update notes](https://www.cfd-sweden.se/lada/Using-Physical-Informed-Neural-Network-PINN-and-NN-improve-a-k-omega-turbulence-model.html).
- Davidson, [2026 paper, preprint v3](https://arxiv.org/abs/2511.12493v3),
  Journal of Turbulence 27(7), 187-208,
  DOI: 10.1080/14685248.2026.2665148.
- Davidson, [pyCALC-RANS report](https://www.tfd.chalmers.se/~lada/postscript_files/py-calc-rans.pdf).
- Lee and Moser, JFM 774 (2015), 395-415, DOI: 10.1017/jfm.2015.268.

Execution results and remaining limitations are recorded separately in
`results/week14_validation/`; a source-code audit is not a completed validation.
