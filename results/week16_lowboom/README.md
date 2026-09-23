# Week 16: retained computational evidence

This module was executed before classroom publication. All flow data were generated with Gmsh 4.15.2 and SU2 8.5.0 (axisymmetric Euler, zero incidence). The scientific quantity is off-body pressure, with the primary objective at r/L=0.5. Ground loudness is not calculated.

## Main result

| Quantity | Approx. 45,000 cells | Approx. 81,000 cells |
| --- | ---: | ---: |
| Baseline peak Cp | 0.0328684 | 0.0336963 |
| Optimized peak Cp | 0.0257889 | 0.0266733 |
| Peak reduction | 21.54% | 20.84% |
| Pressure-drag change | -5.28% | -4.54% |

The optimized body satisfies the fixed-volume condition and the 2% drag-increase limit on both meshes. The independent Taylor-Maccoll cone pressure error is 1.00%. Enlarging the baseline domain changes peak Cp by 0.14%. This is an engineering teaching-level resolution check; the solution is not described as exact or fully grid-independent.

One surrogate-selected alternative (`alternative_1`) violates the drag constraint after fresh CFD. It remains in `design_comparison.csv` as an explicitly rejected alternative. The other alternative and principal design satisfy the constraint. Comparable scalar objectives do not prove identical waveforms or uniqueness of an inverse design.

The validation-selected POD-MLP has test waveform relative L2 error 4.83%, mean peak error 2.39%, and mean pressure-drag error 2.42%. Extrapolation is harder: its waveform error is 32.39% and drag error is 21.25%. All models use one fixed seed and one training sample set; statistical architecture superiority is not claimed.

## Contents

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
