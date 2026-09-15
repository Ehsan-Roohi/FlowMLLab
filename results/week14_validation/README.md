# Week 14 validation ledger

## What is established

- The supplied notebook ran without a cell error. Its data hashes are valid,
  but its RANS baseline label is scientifically incorrect: it embeds the
  PINN-corrected profile. The input in Downloads was not changed.
- The original c_k training script ran for all 1000 epochs, with an explicitly
  added PyTorch seed of 42. Test MSE: **0.00010168597646855102**.
- The compact teaching implementation produced exactly the same test MSE
  on the executed environment; test relative L2 was **2.30415%**. This is a
  within-profile random-point test, not whole-case transfer.
- Fresh source-assembled classical and PINN-corrected 5200 solver restarts
  reproduce their archived profiles closely. They use the original 1 x 70 grid.
- Four diagnostic unit tests pass. The notebook checks hashes, dimensions,
  finite arrays, feature reconstruction, and numerical result gates.

## Important negative findings

The unmodified RANS restart has U relative L2 **1.726%**, k relative L2
**42.849%**. The PINN-corrected restart has U relative L2 **4.988%**, k relative
L2 **8.945%**. These are unweighted norms on the stored stretched grid.
The k improvement is accompanied by a larger velocity error.

**Paper-claim boundary:** this pair uses the distributed table-based PINN stage,
not a verified reproduction of the final PINN-NN comparison in Figure 8 of
Davidson's 2026 paper. Section 5.1 describes good mean-velocity predictions
with both final models and improved k. The 1.726% to 4.988% change is our
particular archived-file metric, not a percentage quoted by the paper or
evidence refuting its final-model claim. Figure-level agreement remains a
separate verification step.

The baseline completes 20,000 iterations with residual **3.119e-13** and unit
wall shear, but does **not** meet its original **1e-14** stopping criterion.
The corrected 5200 restart meets its **1e-6** criterion after two iterations
(zero-based final iteration 1); this is continuation of an already converged
source state, not a cold start.

The original balance script does **not** reproduce the released coefficient
tables from the bundled inputs. Maximum absolute differences: C_k two-column
table **0.524585**, C_omega2 **0.019786**, sigma_k **0.298867**. We retain the
published targets for source training and keep regenerated versions separately
in the local source run tree. No undocumented input substitution is used to
force agreement. **Full end-to-end target reproduction is not established.**

The NN-10000 preassembled executable is stale relative to its corrected
`modify_case.py`. Our runner reassembles the source and records the resulting
hash. The two documented initialization fixes are thereby active.

The historical gap test gives PCHIP **1.467%** relative L2 versus **4.123%** for
the adapted tanh MLP. This already-inspected interval is held out from fitting,
not a prospective blind benchmark. The two methods have different deployment
roles: PCHIP uses y; the NN uses local flow features.

## Extended executions

The NN-10000 and full 200,000-epoch inverse-PINN runs have completed with separate JSON records.
See `summary.json` and the final execution status below; never
infer completion from the presence of a partial log or a figure.

Completed NN-10000: 40,000 iterations, residual 1.566e-5 versus the requested
1e-6 (gate not met), wall shear 0.998672, finite fields, 1 x 120 cells. The
paper's Table 1 uses 150 wall-normal cells; this is not a mesh-identical
reproduction of its Figure 9.

Completed inverse PINN: all 200,000 epochs from the original supplied
checkpoint. Minimum printed total loss about 1.24, but **final** printed loss
about 231 (interior loss 230.633789). Do not report the minimum as the final
checkpoint's loss. The exported diffusion profile differs from the bundled
profile by approximately 0.7555% relative L2. Its downstream effects have not
been retrained and re-solved as a new end-to-end pipeline. This is execution
of the source optimization, not replication of the paper's final loss of 4.

The public paper mentions a scheduler milestone at 80,000; the released load
script uses 800,000 in that position. The source is retained unchanged in this
run, and the discrepancy is not silently corrected.

## Files and interpretation

Completed constructed NN-5200: 40,000 iterations, residual **2.386e-3** versus
**1e-6** (gate not met), wall shear **1.001233**, finite fields, 1 x 70 cells.
It combines the original 5200 setup/restart with the NN deployment code and
m=500 averaging; it is not an unchanged archive case. Its final-iterate metrics
are diagnostics, not a converged confirmation of Figure 8.

- `manifest.json`: compact evidence hashes, source archive hash, software versions.
- `summary.json`: recomputed metrics and run outcomes.
- `baseline.npz`, `pinn.npz`, and any completed `nn10000.npz`: newly executed
  solver fields. They are source-checkpoint restarts, not new DNS.
- `train_ck.npz`: fresh training history, predictions and source split.
- `teaching_data.npz`: attributed source tables, not generated flow fields.
- PNG files: new FlowMLLab scientific plots, not copied paper figures.
- `supplied_notebook_executed.ipynb` (local only; excluded from publication): historical audit execution, **not** the
  corrected teaching notebook. It retains the original scientific labeling
  errors to preserve the input audit trail.
- Full local logs are retained beside these files but excluded by the existing
  repository log-ignore rule. Reproduce them with the original-script runner.

## Provenance and reuse

Source archive:
https://www.cfd-sweden.se/lada/pythons-rans-code-RANS-open.tar.gz

SHA-256: `7aae30d0e990e78ac008c03d008ddc7609ce3a4aae9cc55fada65f1be91d774f`.
Public update page dated 11 September 2026. Individual upstream file hashes
are retained locally in `tmp/w14/source_manifest.json`.

Solver, PINN/NN workflow, trained source checkpoints: Lars Davidson.
DNS channel statistics: Myoungkyu Lee and Robert D. Moser, JFM 774 (2015),
DOI 10.1017/jfm.2015.268. Method paper: Davidson, Journal of Turbulence 27(7)
(2026), DOI 10.1080/14685248.2026.2665148.

The instructor reports permission in the shared conversation. The written
authorization is not included; no exact permission date or blanket relicensing
is asserted. This local teaching adaptation must not assign FlowMLLab's MIT
license to third-party data, solver code or checkpoints. Original source stays
in the local download directory. Publication of this attributed teaching audit
was requested by the instructor, who reports permission. Further reuse of
third-party materials remains subject to their original terms.

The code was run using Python 3.12.14 and PyTorch 2.14.0 CPU, rather than the
author's recorded Python 3.12.2 / PyTorch 2.5.1 environment. Numerical differences
are therefore possible; environment equality is not claimed.
