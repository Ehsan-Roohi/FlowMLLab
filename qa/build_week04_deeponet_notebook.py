#!/usr/bin/env python3
"""Build the expanded Week-4 POD-DeepONet teaching notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "week04" / "W4_Lab3_DeepONet_Cavity_Student.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""# Week 4 Lab 3 — POD-DeepONet for the Ghia cavity

This lab turns the validated lid-driven-cavity family into a compact **parameter-to-field neural operator**. The branch network receives Reynolds number; a development-only POD basis acts as an interpretable spatial trunk. You will verify blind-field accuracy, incompressibility, Ghia centerlines, seed sensitivity, and measured inference speed.

### Learning outcomes

By the end, you should be able to:

1. distinguish coordinate regression from branch–trunk operator factorization;
2. build a POD trunk without using blind fields;
3. select rank and branch width using a complete validation case;
4. refit with three predeclared seeds and open blind Reynolds cases once;
5. compare the neural field with CFD and Ghia centerlines; and
6. explain why fast inference is a valid advantage even when the CFD labels remain the physical reference.
"""
    ),
    markdown(
        r"""## Concept map

\[
\text{validated CFD snapshots}
\rightarrow \text{POD trunk }t_k(x,y)
\rightarrow \text{branch MLP }b_k(Re)
\rightarrow \widehat{\mathbf q}(Re,x,y)
\rightarrow \text{blind fields + Ghia + timing}.
\]

The model is

\[
\widehat{\mathbf q}(Re,x,y)=\overline{\mathbf q}(x,y)+
\sum_{k=1}^{r} b_k(Re)t_k(x,y), \qquad \mathbf q=[u,v].
\]

This is a restricted scalar-branch **POD-DeepONet**. It does not claim to learn an arbitrary function-to-function map; a true boundary-function operator is proposed in the exercises.
"""
    ),
    markdown(
        r"""## Reproducibility contract

- Development cases: `Re = [100, 150, 200, 225, 250, 300, 350, 400]`
- Architecture-selection validation case: complete `Re = 225` field
- Blind cases: `Re = [175, 275, 375]`
- Candidate ranks: `r = [3, 4]`
- Candidate branch widths: `(16,16)`, `(32,32)`, `(64,64)`
- Predeclared seeds: `690, 691, 692`
- Metric: velocity-vector relative \(L_2\), plus wall and divergence checks

Do not change the candidate set after inspecting the blind metrics. If you intentionally run a new experiment, create a new untouched test family and record the change.
"""
    ),
    code(
        """from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Works from the repository root or from notebooks/week04.
ROOT = next(
    candidate for candidate in [Path.cwd(), *Path.cwd().parents]
    if (candidate / "data" / "cavity_data.npz").exists()
)
sys.path.insert(0, str(ROOT / "common"))
import run_pod_deeponet_validation as poddon
import w4utils

RESULTS = ROOT / "results" / "pod_deeponet"
data = poddon.load_data()
print("course root:", ROOT)
print("dataset version:", str(data["package_version"]))
print("development Re:", data["Re"][data["split"] == "train"])
print("blind Re:", data["Re"][data["split"] == "test"])
assert np.all(data["accepted"]), "One or more CFD cases failed quality control"
"""
    ),
    markdown(
        r"""## 1. Start from physical validation, not from the network

The labels come from the educational streamfunction–vorticity solver. Its velocity centerlines at \(Re=100\) and 400 were compared with Ghia, Ghia, and Shin (1982). The neural operator is useful only if it retains that benchmark fidelity.

Pressure is not included in this POD-DeepONet output. It is recovered separately from the steady momentum-gradient field in Lab 1, so a network that predicts stored pressure accurately would still inherit the label-generation limitations.
"""
    ),
    code(
        """def centerlines(u, v):
    mid = len(data["x"]) // 2
    return u[:, mid], v[mid, :]

idx400 = poddon.case_index(data, 400)
u400, v400 = data["u"][idx400], data["v"][idx400]
uc, vc = centerlines(u400, v400)
ghia = w4utils.GHIA[400]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(uc, data["y"], label="course CFD")
axes[0].scatter(ghia["u"], ghia["y"], label="Ghia et al.")
axes[0].set(xlabel="u/U_lid", ylabel="y/L", title="Vertical centerline")
axes[1].plot(data["x"], vc, label="course CFD")
axes[1].scatter(ghia["x"], ghia["v"], label="Ghia et al.")
axes[1].set(xlabel="x/L", ylabel="v/U_lid", title="Horizontal centerline")
for ax in axes: ax.grid(alpha=.25); ax.legend()
plt.tight_layout()
"""
    ),
    markdown(
        r"""### Prediction prompt

Before computing the POD, answer:

1. Will the first mode describe the mean primary vortex, a Reynolds-dependent correction, or a wall discontinuity?
2. Which blind case should be hardest: 175, 275, or 375? Why?
3. Why can a linear combination of divergence-free CFD snapshots preserve divergence more naturally than an unconstrained coordinate MLP?
"""
    ),
    markdown(
        r"""## 2. Construct a development-only spatial trunk

Concatenate each development case as \([u(:),v(:)]\), subtract the development mean, and compute an SVD. The right singular vectors are spatial trunk modes. No blind snapshot may enter the mean or SVD.
"""
    ),
    code(
        """development = np.where(data["split"] == "train")[0]
states = poddon.state_matrix(data, development)
state_mean = states.mean(axis=0)
_, singular_values, spatial_modes = np.linalg.svd(states - state_mean, full_matrices=False)
energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)

display(pd.DataFrame({
    "mode": np.arange(1, len(singular_values)+1),
    "singular_value": singular_values,
    "cumulative_energy": energy,
}))

plt.semilogy(np.arange(1, len(singular_values)+1), singular_values, "o-")
plt.xlabel("mode"); plt.ylabel("singular value"); plt.grid(alpha=.3)
"""
    ),
    markdown(
        r"""Energy is a representation diagnostic, not the final selection rule. A low-energy mode can still affect a wall layer or weak secondary vortex. Therefore rank is selected by complete-case validation, not by choosing a visually attractive cumulative-energy threshold after seeing the test fields.
"""
    ),
    markdown(
        r"""## 3. Branch network and leakage-free selection

For a fixed rank, the coefficients are exact projections onto the development trunk. A small tanh MLP maps standardized Reynolds number to standardized coefficients. L-BFGS is appropriate here because only seven physical cases are used during architecture selection.

For each candidate, train seeds 690–692 on the selection-training cases and compute the mean error on the complete \(Re=225\) field. This uses Reynolds cases—not random grid points—as the statistical units.
"""
    ),
    code(
        """RUN_SELECTION = False  # Set True to regenerate the six-candidate table.

if RUN_SELECTION:
    selected_rank, selected_hidden, selection = poddon.select_model(data)
else:
    selection = pd.read_csv(RESULTS / "deeponet_selection.csv")
    best = selection.sort_values("mean_validation_relative_L2_uv").iloc[0]
    selected_rank = int(best["rank"])
    selected_hidden = tuple(int(v) for v in best["hidden"].split("x"))

display(selection[[
    "rank", "hidden", "trunk_energy_fraction",
    "mean_validation_relative_L2_uv",
    "min_validation_relative_L2_uv",
    "max_validation_relative_L2_uv",
]])
print("selected rank:", selected_rank, "selected branch:", selected_hidden)
"""
    ),
    markdown(
        r"""### Frozen selection result

The development-only rule selected rank \(r=3\) and branch widths `(32,32)`:

- mean validation velocity error: **0.235%**;
- three-seed validation range: **0.189%–0.261%**;
- final rank-3 trunk energy after refitting on allowed development cases: **99.988%**.

The blind cases have not been used in any statement above.
"""
    ),
    markdown(
        r"""## 4. Refit after selection

Now rebuild the trunk using all allowed development cases and fit the selected branch architecture with all three predeclared seeds. The output transform imposes the known no-slip and moving-lid velocity values exactly. This transform uses prescribed boundary conditions, not blind CFD labels.
"""
    ),
    code(
        """RUN_REFIT = False  # True retrains the three branches; the saved result is loaded otherwise.

if RUN_REFIT:
    trunk = poddon.make_trunk(data, development, selected_rank)
    bundles = [
        poddon.fit_branch(data, development, trunk, selected_hidden, seed)
        for seed in poddon.SEEDS
    ]
    print("refit complete")
else:
    print("Using the verified result package. Set RUN_REFIT=True for an in-memory refit.")
"""
    ),
    markdown(
        r"""# Stop: blind-test gate

Before running the next cell, write down:

- selected rank and branch widths;
- development and blind Reynolds lists;
- seeds;
- primary and physical metrics; and
- the failure criterion you would report.

Do not return to model selection after opening the blind table. A new model requires a new untouched test.
"""
    ),
    code(
        """metrics = pd.read_csv(RESULTS / "deeponet_metrics.csv")
blind = metrics[metrics["method"] == "three-seed POD-DeepONet ensemble"].copy()
blind["E_uv_percent"] = 100 * blind["relative_L2_uv"]
display(blind[["Re", "E_uv_percent", "div_l2_pred", "wall_rms_error"]])

assert blind["wall_rms_error"].max() == 0.0
assert blind["div_l2_pred"].max() < 1e-12
assert blind["relative_L2_uv"].max() < 0.005
"""
    ),
    markdown(
        r"""### Blind result—every case reported

| blind \(Re\) | ensemble \(E_{uv}\) | individual-seed range | divergence \(L_2\) |
|---:|---:|---:|---:|
| 175 | **0.4525%** | 0.2997%–0.6041% | \(1.33\times10^{-15}\) |
| 275 | **0.0718%** | 0.0678%–0.1030% | \(1.34\times10^{-15}\) |
| 375 | **0.0928%** | 0.0892%–0.1452% | \(1.35\times10^{-15}\) |

This is not a best-seed table. The pointwise ensemble and the full seed range are both retained.
"""
    ),
    markdown(
        r"""## 5. Field evidence

The top row checks that the POD-DeepONet retains the \(Re=400\) Ghia centerlines. The bottom row shows CFD, blind POD-DeepONet, and error fields at \(Re=275\).

![Executed POD-DeepONet result](../../results/pod_deeponet/pod_deeponet_ghia_validation.svg)
"""
    ),
    markdown(
        r"""## 6. What is the neural-network advantage?

The claim is not that a surrogate improves the Ghia data. The reference CFD solve establishes the physics label; the network should retain that fidelity while avoiding a new iterative PDE solve for each query.

The reported timing includes reconstruction of all \(65\times65\) values from a three-seed ensemble. The CFD timing is an independently rerun streamfunction–vorticity solve including pressure recovery, stopped at residual \(8.95\times10^{-7}\).
"""
    ),
    code(
        """ghia_metrics = pd.read_csv(RESULTS / "deeponet_ghia_metrics.csv")
timing = json.loads((RESULTS / "deeponet_protocol_and_timing.json").read_text())
display(ghia_metrics)
display(pd.DataFrame([{
    "POD-DeepONet ensemble inference (ms)": timing["POD_DeepONet_ensemble_inference_ms"],
    "CFD solve (s)": timing["CFD_Re275_seconds"],
    "amortized speedup": timing["speedup"],
    "CFD residual": timing["CFD_final_residual"],
}]))
"""
    ),
    markdown(
        r"""### Ghia and timing result

| \(Re\) | CFD \(E_u\) | POD-DeepONet \(E_u\) | CFD \(E_v\) | POD-DeepONet \(E_v\) |
|---:|---:|---:|---:|---:|
| 100 | 0.008594 | 0.008659 | 0.024205 | 0.024346 |
| 400 | 0.052393 | 0.052478 | 0.154259 | 0.154486 |

The neural field preserves the benchmark fidelity of the educational CFD labels. In the recorded CPU run, ensemble inference was about **0.8 ms** and the CFD solve about **8 s**, an amortized per-case advantage near **\(10^4\times\)**. Report timing with hardware/software context and never include training cost in a per-query speedup without stating how many queries amortize it.
"""
    ),
    markdown(
        r"""## 7. Interpretation and limitations

1. **Why divergence is near round-off:** the fixed trunk is a linear basis of divergence-free CFD snapshots; the output transform also restores exact wall values.
2. **Why the first blind case is harder:** \(Re=175\) lies in a part of coefficient space where the branch mapping is more sensitive to sparse parameter sampling.
3. **Why this is still restricted:** a scalar Reynolds input does not describe a new lid-velocity function, geometry, or boundary shape.
4. **Why Ghia errors do not vanish:** the neural operator reproduces the educational CFD solver, including its grid/discretization error. It cannot become more accurate than its labels without additional physical information.
5. **Why speedup is useful:** parametric studies, uncertainty propagation, optimization, and interactive teaching require many field queries after training.
"""
    ),
    markdown(
        r"""## Common failure modes and fixes

| Symptom | Likely cause | Evidence-based fix |
|---|---|---|
| Very small training error, poor blind fields | coefficient overfit from only a few cases | reduce rank/branch width; add complete development cases |
| Nonzero wall error | boundary transform omitted or corner convention changed | inspect all four walls and preserve the CFD corner convention |
| Large divergence | predicted \(u,v\) directly with an unconstrained trunk | use a streamfunction/POD trunk or add a carefully scaled divergence loss |
| Excellent global error, wrong centerline | global norm hides a local feature | retain Ghia and vortex/centerline diagnostics |
| Unrealistic speedup | data loading or solver convergence excluded inconsistently | time matched end-to-end calls; report warm-up and repetitions |
| Better test result after several redesigns | test set became a validation set | create a new untouched physical test family |
"""
    ),
    markdown(
        r"""## Exercises

1. Plot the first three shared trunk modes and identify the flow structure carried by each.
2. Repeat selection with rank 2 included. Does validation—not energy alone—reject it?
3. Compare one seed, the three-seed ensemble, and the seed spread at every blind case.
4. Remove the boundary output transform and quantify wall, global, and divergence changes.
5. Build a true operator dataset in which the branch input is \([U_{lid}(\xi_1),\ldots,U_{lid}(\xi_m)]\), not a scalar Reynolds number.
6. Repeat the timing for 1, 10, 100, and 1000 queries. At what query count does offline work amortize?
7. Add pressure as a separate branch–trunk output, then explain why matching stored pressure does not independently validate pressure recovery.
"""
    ),
    markdown(
        r"""## Required deliverables

- selection table with complete case lists and three seeds;
- blind table with every Reynolds case, seed range, wall error, and divergence;
- one field/error figure and both centerline comparisons;
- Ghia table for \(Re=100\) and 400;
- matched timing table with residual and software/hardware context;
- a paragraph stating the restricted scope of scalar-branch POD-DeepONet; and
- one failure mode or limitation that remains after the final model.

### Suggested report claim

> After development-only rank/architecture selection, the POD-DeepONet preserved the Ghia fidelity of the reference solver, maintained round-off-level divergence, achieved sub-0.5% error on all complete blind fields, and reduced amortized field-query time from seconds to milliseconds. The result applies to the fixed-geometry, scalar-Reynolds family and is not evidence of arbitrary boundary-function generalization.
"""
    ),
    markdown(
        r"""## References and next steps

- Ghia, Ghia & Shin (1982), *Journal of Computational Physics* 48, 387–411.
- Lu et al. (2021), “Learning nonlinear operators via DeepONet,” *Nature Machine Intelligence* 3, 218–229.
- Berkooz, Holmes & Lumley (1993), POD in turbulent flows.
- Hesthaven & Ubbiali (2018), non-intrusive neural reduced-order modeling.

Continue with the Week-5 POD, physics-guided-loss, or uncertainty track. For a research extension, replace the scalar branch with a sampled boundary function or a geometry encoding and create new complete blind operators.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "course": {
            "title": "MIE 690A AI in Fluid Mechanics",
            "lab": "Week 4 Lab 3 POD-DeepONet",
            "protocol_version": "2026.08",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT} with {len(cells)} cells")
