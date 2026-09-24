> **Week 16 scope (24 September 2026):** the release covers the verified teaching-body CFD, neural model, design checks and Taylor–Maccoll study. NASA SEEB-ALR CFD failed convergence/physical checks and remains a deferred research extension; no successful NASA validation is claimed.

# Week 16: retained computational evidence

Publication status is recorded in [PUBLICATION_STATUS.md](../../PUBLICATION_STATUS.md); the educational core is released separately from the failed NASA research extension. New computations use Gmsh 4.15.2 and the checksum-pinned official SU2 8.0.1 executable. Original SU2 8.5.0 results are retained separately as historical and failure-analysis evidence. The scientific quantity is off-body pressure, with the primary objective at r/L=0.5. Ground loudness is not calculated.

## Current acceptance evidence

- `reference/clean_dataset_v801.npz` and `clean_campaign_audit.json`: separately regenerated 44-case student dataset; every case must pass numerical and full-field physical checks.
- `reference/clean_model_v801.npz`, `clean_model_training_v801.json`, `clean_model_audit_v801.json`: one separately identified fixed-architecture fit on the 24 clean training cases; no fitting occurs during checkpoint auditing.
- `reference/weakwall_checkpoint_audit.json`: unchanged historical portable weights versus eight new finer CFD references. Wave/peak/drag errors are 7.1499%/2.6343%/2.0615%, with worst-case waveform error 15.9701%.
- `reference/weakwall_design_audit.json`: ten fresh fixed-geometry evaluations. Peak reductions on levels 1.5 and 2 are 21.1876% and 20.6646%; drag changes are -3.7971% and -3.5218%. Optimized waveform mesh change is 5.3946%; the declared mesh gates apply to peak and drag.
- `reference/cone_refinement_v801.json`: 24,000/96,000/216,000-cell cone pressure errors 4.5632%/2.1563%/1.1616%. The coarse failure is preserved; the last-two Cp change is 1.0064% and the refined-family criterion passes.
- NASA three-mesh experimental comparison: deferred. The retained coarse replay failed, and no accepted NASA comparison report is claimed.

Read each report's `passed`, `checks`, provenance and limitations. The clean dataset/model files appear only after their actual runs and audit finish. A historical optimizer's geometry is not attributed to the new clean model.

## Historical SU2 8.5.0 results — not accepted full-field physical validation

| Quantity | Approx. 45,000 cells | Approx. 81,000 cells |
| --- | ---: | ---: |
| Baseline peak Cp | 0.0328684 | 0.0336963 |
| Optimized peak Cp | 0.0257889 | 0.0266733 |
| Peak reduction | 21.54% | 20.84% |
| Pressure-drag change | -5.28% | -4.54% |

In the historical calculation, the optimized body satisfies the fixed-volume condition and the 2% drag-increase limit on both meshes. The independent Taylor-Maccoll cone pressure error is 1.00%. Enlarging the baseline domain changes peak Cp by 0.14%. This is an engineering teaching-level resolution check; the solution is not described as exact or fully grid-independent.

One surrogate-selected alternative (`alternative_1`) violates the drag constraint after fresh CFD. It remains in `design_comparison.csv` as an explicitly rejected alternative. The other alternative and principal design satisfy the constraint. Comparable scalar objectives do not prove identical waveforms or uniqueness of an inverse design.

The validation-selected POD-MLP has test waveform relative L2 error 4.83%, mean peak error 2.39%, and mean pressure-drag error 2.42%. Extrapolation is harder: its waveform error is 32.39% and drag error is 21.25%. The MLP uses a fixed seed and the campaign uses one training set; the original automatic PCA solver was not explicitly seeded. Refits may differ, and statistical architecture superiority is not claimed. These historical scores refer to the original model evaluation, not the later portable checkpoint.

## Historical contents

- `dataset.npz`: all 44 pressure signatures, parameters, drag coefficients, geometry IDs and split labels.
- `design_plan.json`: 24 train, six validation, eight test and six extrapolation geometries, frozen before their CFD generation.
- `software_versions.json`: actual package and solver versions, including the official binary SHA-256.
- `case_metrics.csv`: per-case numerical checks and solver wall time.
- `learning_metrics.json`, `model_comparison.csv`: all model and split results.
- `design_candidates.json`, `design_comparison.csv`: proposals and independent forward checks.
- `summary.json`: mesh/domain checks, Mach stress tests, volume checks and dataset hash.
- `release_check.json`: scientific and educational completion checks.
- `numerical_evidence.zip`: solver configurations, full iteration histories, solver logs, extracted pressure signals, numerical metadata and the cone benchmark, under `runs/`.
- Figures: actual CFD fields, shape/signature comparisons, convergence, learning and off-design checks.

Full Gmsh meshes, restart files and volume fields have been retained separately in `FlowMLLab_Week16_Raw_CFD.zip` (328,376,876 bytes; SHA-256 `6a99f8e0a99d2e0a66f1391760eb0a25fef0f58d2ed422e009c221c6620adde8`). They are not duplicated in Git history. The repository's executable scripts regenerate them; the instructor also retains the archive. Extract either archive into this directory, preserving `runs/`.

## Reproduction and limits

Follow [the CFD guide](../../cases/week16_lowboom/README.md). Use a separate Python 3.12 environment with `qa/week16/requirements.txt`; the recorded Week 16 numerical packages are newer than the core package's pinned range. The notebook's original execution used IPython in a fresh process because this execution environment prohibits local kernel sockets; real text, tables and figures were captured. The normal build path uses nbclient on systems supporting Jupyter kernels.

The body family has two parameters, no lift or propulsion and no viscous drag. There is no atmospheric propagation, perceptual loudness computation or full-aircraft result. The article by Zheng et al. motivates the design question; no article data, figures or PDF are redistributed. The new data, code, figures and educational text were authored for FlowMLLab with AI assistance and checked through the retained calculations.

## Reference extension

`reference/` retains unchanged NASA sources through the separate source directory, a historical frozen-prediction audit and a distinct portable neural checkpoint. `qa/week16/independent_audit_report.py` recomputes the historical audit; `qa/week16/freeze_model.py` loads and audits the later checkpoint without retraining. Read the model guide before comparing their different error tables.

New NASA raw runs are retained as GitHub Actions artifacts and, after successful publication, release assets. `reference/reference_evidence.zip` packages their compact configuration, histories, signatures and metadata. The experimental comparison is recomputed by `reference_report.py`; full raw-file verification is performed by `validate_reference.py`. The original `source_sha256.json` remains a historical provenance record, while `reference/release_source_sha256.json` records the reference-extension sources at publication.

## Failed physical checks are retained

The eight old 8.5.0 finer test fields failed the maximum total-enthalpy allowance at body/axis tips despite converged residuals. The original underresolved NASA nose family was also rejected. See `reference/neural_physical_plausibility.json`, the pointed-body physics guide and the nose physics guide. Later accepted runs do not retroactively validate these fields. Full raw-data acceptance is checked separately from compact, laptop-friendly notebook comparisons.
