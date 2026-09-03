"""Build the two Week-9 Roohi--Mahdavi research-to-classroom notebooks."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:  # Keep release generation dependency-free.
    class _V4:
        @staticmethod
        def new_markdown_cell(source):
            return {"cell_type": "markdown", "metadata": {}, "source": source}

        @staticmethod
        def new_code_cell(source):
            return {
                "cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": source,
            }

        @staticmethod
        def new_notebook(cells, metadata):
            return {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}

    class _NotebookFormat:
        v4 = _V4()

        @staticmethod
        def write(notebook, target):
            Path(target).write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    nbf = _NotebookFormat()


HERE = Path(__file__).resolve().parent


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


BOOTSTRAP = r"""
# FLOWMLLAB_COLAB_BOOTSTRAP_V1
from pathlib import Path as _FlowMLLabPath
import os as _flowmllab_os
import subprocess as _flowmllab_subprocess
import sys as _flowmllab_sys

if "google.colab" in _flowmllab_sys.modules or _flowmllab_os.environ.get("COLAB_RELEASE_TAG"):
    _flowmllab_root = _FlowMLLabPath("/content/FlowMLLab")
    if not (_flowmllab_root / ".git").is_dir():
        _flowmllab_subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(_flowmllab_root)],
            check=True,
        )
    _flowmllab_subprocess.run(
        [_flowmllab_sys.executable, "-m", "pip", "install", "-q", "-e", str(_flowmllab_root)],
        check=True,
    )
    _flowmllab_os.chdir(_flowmllab_root / "notebooks/week09")

from pathlib import Path
import json
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ModuleNotFoundError:
    display = print

REPO_ROOT = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "results/mahdavi_deeponet").is_dir()
)
RESULTS = REPO_ROOT / "results/mahdavi_deeponet"
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9})
print("Python:", platform.python_version())
print("FlowMLLab root:", REPO_ROOT)
"""


def lab1_cells():
    return [
        md(r"""
# Week 9 Lab 1 — Micro-step DeepONet and physics-guided zonal loss

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb)

**Runtime:** CPU, normally under 2 minutes. **Prerequisites:** case-wise data
splits, relative error, and the basic idea of a neural operator.

This lab turns the Roohi--Mahdavi article *Analysis of the rarefied flow at
micro-step using a DeepONet surrogate model with a physics-guided zonal loss
function* (Microfluidics and Nanofluidics 30:44, published 11 May 2026) into a
controlled classroom experiment.

### Learning outcomes

By the end, you should be able to:

1. distinguish a parameter-to-field operator from pointwise regression;
2. keep every geometry case entirely inside one split;
3. explain why a global mean loss can hide recirculation failure;
4. select a zonal-loss weight using validation cases only; and
5. separate retained article evidence from a manufactured method demonstration.
"""),
        md(r"""
## Evidence and claim contract — read before code

The paper's DSMC micro-step fields and trained checkpoint are not in the public
repositories audited for this lesson. Therefore:

- the table loaded below is **retained paper evidence**;
- the velocity fields generated in this notebook are **manufactured teaching
  fields**, not DSMC and not a reproduction of the paper;
- the fitted model below is a linearized branch--trunk operator used to expose
  the loss tradeoff, not the paper's neural checkpoint; and
- the held-out ratios 44% and 67% mirror reported article tests, but the
  notebook numbers must never be quoted as article accuracy.

The useful scientific question remains real: when the reverse-flow region is
small, can a globally good operator still be locally unacceptable?
"""),
        code(BOOTSTRAP),
        code(r"""
from flowmllab.mahdavi_deeponet import (
    manufactured_step_velocity,
    zonal_velocity_metrics,
)

paper = pd.read_csv(RESULTS / "step_paper_evidence.csv")
display(paper.pivot(index="objective", columns="scope", values="reported_error_percent"))
"""),
        md(r"""
## 1. DeepONet as a parameter-to-field map

For a geometry parameter $h/H$ and query coordinate $\mathbf{y}=(x,y)$, a
DeepONet has the separable form

$$
\widehat{G}(h/H)(\mathbf{y})=
\sum_{k=1}^{r} b_k(h/H)\,t_k(\mathbf{y})+b_0.
$$

The **branch** network encodes the input case; the **trunk** network encodes the
query coordinate. All points from one height are correlated parts of a single
operator sample. Randomly splitting points would put the same geometry in both
training and test sets and produce leakage.

The paper defines the recirculation zone from the reference streamwise
velocity, $U<0$, and balances two separately normalized regional errors:

$$
\mathcal L_{\rm zonal}=\alpha\mathcal L_{U<0}+
(1-\alpha)\mathcal L_{U\ge 0}.
$$

Predict before running: as $\alpha$ increases, which error should decrease,
and what global tradeoff might appear?
"""),
        code(r"""
x = np.linspace(0.0, 5.0, 72)
y = np.linspace(0.0, 1.0, 32)
xx, yy = np.meshgrid(x, y)
u_demo, v_demo, solid_demo = manufactured_step_velocity(xx, yy, 0.44)

masked_u = np.ma.array(u_demo, mask=solid_demo)
fig, ax = plt.subplots(figsize=(10.5, 3.0), constrained_layout=True)
levels = np.linspace(-0.7, 1.55, 24)
contour = ax.contourf(xx, yy, masked_u, levels=levels, cmap="coolwarm", extend="both")
ax.contour(xx, yy, masked_u, levels=[0.0], colors="black", linewidths=1.1)
ax.fill_between([0, 1], 0, 0.44, color="0.2", label="solid step")
ax.set(xlabel=r"$x/H$", ylabel=r"$y/H$", title="Manufactured teaching field, $h/H=0.44$")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(contour, ax=ax, label=r"$U/U_0$")
plt.show()

print("This plot is pedagogical, not article DSMC data.")
"""),
        md(r"""
## 2. A reviewable linear branch--trunk operator

To isolate the effect of the objective, we use polynomial branch features of
$h/H$ and a fixed, low-rank spatial trunk. Their outer products form a
separable operator basis. Weighted least squares then changes only the loss,
not the data, basis, optimizer tolerance, or split.

This is the linear limit of the branch--trunk idea. An assignment extension at
the end replaces both feature maps with small neural networks while preserving
the same split and metrics.
"""),
        code(r"""
def operator_design(height_ratio):
    xn = xx / x.max()
    local = [
        np.exp(-((xx - xc) / sx) ** 2 - (yy / sy) ** 2)
        for xc, sx, sy in ((1.25, .45, .16), (1.70, .65, .22), (2.20, .90, .28))
    ]
    trunk = np.stack([
        np.ones_like(xx), xn, yy, xn**2, yy**2, xn * yy,
        np.sin(np.pi * yy), np.cos(np.pi * yy), *local,
        xx * local[1], yy * local[1],
    ], axis=-1)
    branch = np.array([1.0, height_ratio, height_ratio**2, height_ratio**3])
    return (trunk[..., None, :] * branch[None, None, :, None]).reshape(*xx.shape, -1)


def fit_operator(case_heights, alpha=None):
    designs, u_targets, v_targets, vortex_flags = [], [], [], []
    for height in case_heights:
        u, v, solid = manufactured_step_velocity(xx, yy, height)
        valid = ~solid
        designs.append(operator_design(height)[valid])
        u_targets.append(u[valid])
        v_targets.append(v[valid])
        vortex_flags.append(u[valid] < 0.0)
    design = np.vstack(designs)
    u_target = np.concatenate(u_targets)
    v_target = np.concatenate(v_targets)
    vortex = np.concatenate(vortex_flags)
    if alpha is None:
        weight = np.full(len(u_target), 1.0 / len(u_target))
    else:
        weight = np.where(
            vortex, alpha / vortex.sum(), (1.0 - alpha) / (~vortex).sum()
        )
    normal = design.T @ (weight[:, None] * design) + 1e-7 * np.eye(design.shape[1])
    coef_u = np.linalg.solve(normal, design.T @ (weight * u_target))
    coef_v = np.linalg.solve(normal, design.T @ (weight * v_target))
    return coef_u, coef_v


def evaluate_operator(coefficients, case_heights):
    coef_u, coef_v = coefficients
    rows = []
    for height in case_heights:
        u, v, solid = manufactured_step_velocity(xx, yy, height)
        design = operator_design(height)
        metrics = zonal_velocity_metrics(
            u, v, design @ coef_u, design @ coef_v,
            alpha=0.7, valid_mask=~solid,
        )
        rows.append({"h_over_H": height, **metrics})
    return pd.DataFrame(rows)
"""),
        md(r"""
## 3. Freeze the split and selection rule

We reserve 44% and 67% for the final teaching test. Heights 40% and 60% are
validation cases. The remaining six cases fit the operator. No spatial point
from a reserved geometry enters fitting.

**Predeclared rule:** among $\alpha\in\{0.3,0.5,0.6,0.7,0.8\}$, choose the
largest vortex improvement whose validation global relative error is no more
than two percentage points worse than the unweighted fit. This prevents a
zonal win purchased by an unlimited global failure.
"""),
        code(r"""
all_heights = np.array([.25, .30, .35, .40, .44, .50, .55, .60, .67, .72])
test_heights = np.array([.44, .67])
validation_heights = np.array([.40, .60])
development_heights = np.array([
    value for value in all_heights
    if value not in set(test_heights) | set(validation_heights)
])
print("development:", development_heights)
print("validation:", validation_heights)
print("unopened teaching test:", test_heights)

baseline_validation = evaluate_operator(
    fit_operator(development_heights, alpha=None), validation_heights
)
baseline_global = 100 * baseline_validation["full_relative_l2"].mean()
selection_rows = []
for alpha in (.3, .5, .6, .7, .8):
    metrics = evaluate_operator(fit_operator(development_heights, alpha), validation_heights)
    selection_rows.append({
        "alpha": alpha,
        "validation_global_percent": 100 * metrics["full_relative_l2"].mean(),
        "validation_vortex_percent": 100 * metrics["vortex_relative_l2"].mean(),
    })
selection = pd.DataFrame(selection_rows)
eligible = selection[
    selection["validation_global_percent"] <= baseline_global + 2.0
]
selected_alpha = float(eligible.sort_values("validation_vortex_percent").iloc[0]["alpha"])
display(selection)
print(f"unweighted validation global error: {baseline_global:.3f}%")
print("selected alpha:", selected_alpha)
assert selected_alpha == 0.6
"""),
        md(r"""
## Stop: teaching-test gate

At this point the split, basis, regularization, candidate weights, and selection
rule are frozen. Write your prediction for the two reserved heights. Only then
run the next cell. If you change a choice after viewing the result, these cases
become development data and must no longer be called held out.
"""),
        code(r"""
fit_heights = np.concatenate([development_heights, validation_heights])
global_test = evaluate_operator(fit_operator(fit_heights, None), test_heights)
zonal_test = evaluate_operator(fit_operator(fit_heights, selected_alpha), test_heights)

comparison = pd.DataFrame({
    "model": ["unweighted", "zonal"],
    "mean_global_percent": [
        100 * global_test["full_relative_l2"].mean(),
        100 * zonal_test["full_relative_l2"].mean(),
    ],
    "mean_vortex_percent": [
        100 * global_test["vortex_relative_l2"].mean(),
        100 * zonal_test["vortex_relative_l2"].mean(),
    ],
})
display(comparison)

fig, ax = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
locations = np.arange(2)
width = 0.34
ax.bar(locations - width / 2, comparison["mean_global_percent"], width, label="global")
ax.bar(locations + width / 2, comparison["mean_vortex_percent"], width, label="recirculation")
ax.set_xticks(locations, comparison["model"])
ax.set(ylabel="mean relative L2 error (%)", title="Manufactured held-out cases: objective tradeoff")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=.25)
plt.show()
"""),
        md(r"""
## 4. Interpret without mixing evidence levels

The manufactured experiment should show the intended mechanism: the selected
zonal objective reduces reverse-flow error while accepting a modest increase
in whole-field error. The paper reports the same qualitative tradeoff on its
research data: zonal loss changes the reported recirculation-zone error from
14.6135% (MSE) to 11.9413%, while the full-domain value changes from 2.1739%
to 2.2254%.

Those percentages came from the article table, not the bars above. The
notebook's role is to let you inspect *why* the tradeoff occurs and *how* to
select a weight without opening the final cases.

### DeepONet implementation exercise

Replace `operator_design` with two small Keras networks:

- branch input: one scalar, $h/H$;
- trunk input: two scalars, $(x/H,y/H)$;
- output: dot product of equal-width branch and trunk vectors, with separate
  heads for $U$ and $V$.

Keep complete geometry cases together. Implement the regional means before
mixing them with $\alpha$; a pointwise weight without regional normalization is
not the same objective. Compare a standard DeepONet with a larger fusion model
under the same sparse case budget—the paper reports that additional capacity
can overfit when only seven cases are available.

### Required submission

1. a signed split table;
2. the validation-only $\alpha$ sweep;
3. global and reverse-flow metrics for every held-out geometry;
4. one contour locating the largest local error; and
5. one paragraph stating which outputs are manufactured and which are retained
   article evidence.
"""),
    ]


def lab2_cells():
    return [
        md(r"""
# Week 9 Lab 2 — Shock-aligned DeepONet for a rarefied micro-nozzle

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb)

**Runtime:** CPU, normally under 4 minutes. **Prerequisites:** DSMC sampling,
POD/SVD, case-wise validation, and DeepONet branch--trunk notation.

This lab uses a compact derivative of **all 15 real public DSMC snapshots**
from the Roohi--Mahdavi micro-nozzle study. It asks why a moving shock is
high-rank in laboratory coordinates and low-rank in shock-centered coordinates.

### Learning outcomes

1. audit snapshot provenance and source hashes before learning;
2. detect a centerline compression station and identify noisy outliers;
3. reproduce the 15-snapshot physical versus shock-centered density POD audit;
4. select POD rank by leave-one-case-out development error;
5. open pressures 16, 25, and 30 kPa only after freezing the model; and
6. distinguish a compact centerline teaching model from the article's full 2-D
   six-output Fusion--DeepONet.
"""),
        md(r"""
## Evidence and claim contract

- `nozzle_centerline_15cases.npz` is derived from the public Tecplot DSMC files
  at pinned commit `e1b234ba499408d3b6224633972f939f3b2301d6` and remains
  CC BY 4.0.
- The POD spectrum is directly reproducible from those data.
- The small model in this notebook predicts **jump-normalized centerline
  density profiles**. It is a POD trunk plus a neural branch and is not the
  article's trained full-domain, six-field Fusion--DeepONet checkpoint.
- The article metric tables are immutable retained evidence with a different
  output domain and must not be merged numerically with notebook errors.
- Shock centering uses target-derived locations in the structural POD and
  normalized-profile audit. For deployment, a shock-location model using only
  input pressure and training cases is separately tested below.
"""),
        code(BOOTSTRAP),
        code(r"""
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from flowmllab.mahdavi_deeponet import (
    NOZZLE_HELD_OUT_KPA,
    density_snapshot_matrix,
    load_nozzle_centerlines,
    pod_spectrum,
    relative_l2,
    validate_week9_evidence,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
report = validate_week9_evidence(REPO_ROOT)
print(json.dumps(report, indent=2))
"""),
        md(r"""
## 1. Data audit before model fitting

The source files span back pressures from 15 to 33 kPa. Each main zone is
101 by 31; this teaching archive retains the max-$y$ symmetry centerline and
seven recorded fields. `pressure_tecplot` keeps the original file values
without inferring a unit not stated by the Tecplot header.

The compact file is not a new simulation. Its provenance records every source
filename and SHA-256, the exact source commit, the extraction rule, the derived
file hash, and the licensing change notice.
"""),
        code(r"""
data = load_nozzle_centerlines(REPO_ROOT)
pressure = data["pressure_kpa"]
x_um = data["x_m"] * 1e6
held_out = np.isin(pressure, NOZZLE_HELD_OUT_KPA)
development = ~held_out

audit = pd.DataFrame({
    "pressure_kPa": pressure.astype(int),
    "split": np.where(held_out, "held out", "development"),
    "shock_x_um": data["shock_x_m"] * 1e6,
    "delta_jump_um": data["delta_jump_m"] * 1e6,
})
display(audit)
print("density array:", data["density"].shape)
print("held-out pressures:", pressure[held_out].astype(int).tolist())
assert pressure[held_out].astype(int).tolist() == [16, 25, 30]
"""),
        code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0, 1, len(pressure)))
for index, (p_value, color) in enumerate(zip(pressure, colors)):
    axes[0].plot(x_um, data["density"][index], color=color, lw=1.4, label=f"{p_value:g}")
    axes[1].scatter(p_value, data["shock_x_m"][index] * 1e6, color=color, s=36)
axes[0].set(xlabel=r"$x$ ($\mu$m)", ylabel="density (source units)", title="Real DSMC centerline snapshots")
axes[1].set(xlabel="back pressure (kPa)", ylabel=r"detected $x_s$ ($\mu$m)", title="Moving compression station")
axes[0].grid(alpha=.25)
axes[1].grid(alpha=.25)
axes[0].legend(title="kPa", ncol=3, fontsize=7, frameon=False)
plt.show()
"""),
        md(r"""
The detector chooses the largest smoothed interior density gradient, then
estimates a jump width $\delta_j=|\rho_L-\rho_R|/|\partial\rho/\partial x|_{x_s}$.
At 22 kPa the automatic station is visibly non-monotone. Retaining that point
is important: a shock sensor is a model component that also needs validation,
not an infallible preprocessing oracle.
"""),
        md(r"""
## 2. Structural audit: moving discontinuity versus aligned structure

For each case, define

$$
\xi_j=\frac{x-x_s}{\delta_j}, \qquad
\widetilde\rho=\frac{\rho-\rho_R}{\rho_L-\rho_R}.
$$

First use all 15 cases only for the published **representation audit**, not for
model selection. This answers how compact the snapshot family is after a known
coordinate transform. It does not estimate held-out model accuracy.
"""),
        code(r"""
pod_rows = []
matrices = {}
grids = {}
for coordinate in ("physical", "shock_centered"):
    grid, matrix = density_snapshot_matrix(
        data["x_m"], data["density"], data["shock_x_m"], data["delta_jump_m"],
        coordinate=coordinate,
    )
    spectrum = pod_spectrum(matrix)
    grids[coordinate], matrices[coordinate] = grid, matrix
    pod_rows.append({
        "coordinate": coordinate,
        "E1_percent": spectrum["first_mode_percent"],
        "E12_percent": spectrum["first_two_percent"],
        "E123_percent": spectrum["first_three_percent"],
        "N99": spectrum["n99"],
    })
pod_observed = pd.DataFrame(pod_rows)
pod_reference = pd.read_csv(RESULTS / "nozzle_pod_reference.csv")
display(pod_observed)
display(pod_reference)

fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
for coordinate, matrix in matrices.items():
    cumulative = pod_spectrum(matrix)["cumulative_energy"] * 100
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, "o-", label=coordinate)
ax.axhline(99, color="black", ls="--", lw=1, label="99%")
ax.set(xlabel="number of POD modes", ylabel="cumulative energy (%)", ylim=(75, 101), xlim=(1, 10))
ax.grid(alpha=.25)
ax.legend(frameon=False)
plt.show()

assert pod_observed.set_index("coordinate").loc["physical", "N99"] == 8
assert pod_observed.set_index("coordinate").loc["shock_centered", "N99"] == 2
"""),
        md(r"""
## 3. POD--DeepONet teaching analog with a frozen split

The POD modes act as a fixed trunk, $t_k(\xi)$. A one-input neural branch maps
back pressure to modal coefficients, $b_k(P_b)$. The predicted normalized
centerline profile is

$$
\widehat{\widetilde\rho}(P_b,\xi)=\bar\rho(\xi)+
\sum_{k=1}^{r} b_k(P_b)t_k(\xi).
$$

We select rank separately in each coordinate system using leave-one-case-out
error across the 12 development pressures. Every fold recomputes the POD basis
without its validation case. Architecture, optimizer, random seed, and
candidate ranks are matched.
"""),
        code(r"""
def make_branch():
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
            alpha=0.01, max_iter=1500, random_state=690,
        ),
    )


def fit_predict_pod_branch(matrix, train_indices, query_indices, rank):
    train_matrix = matrix[train_indices]
    mean = train_matrix.mean(axis=0)
    _, _, modes = np.linalg.svd(train_matrix - mean, full_matrices=False)
    rank = min(rank, len(train_indices) - 1)
    coefficients = (train_matrix - mean) @ modes[:rank].T
    target = coefficients.ravel() if rank == 1 else coefficients
    branch = make_branch().fit(pressure[train_indices, None], target)
    predicted_coefficients = np.asarray(
        branch.predict(pressure[query_indices, None])
    ).reshape(len(query_indices), rank)
    return mean + predicted_coefficients @ modes[:rank]


development_indices = np.flatnonzero(development)
candidate_ranks = {
    "physical": [2, 4, 6, 8],
    "shock_centered": [1, 2, 3, 4],
}
selection_rows = []
for coordinate, ranks in candidate_ranks.items():
    matrix = matrices[coordinate]
    for rank in ranks:
        fold_errors = []
        for validation_index in development_indices:
            fold_train = development_indices[development_indices != validation_index]
            prediction = fit_predict_pod_branch(
                matrix, fold_train, np.array([validation_index]), rank
            )[0]
            fold_errors.append(100 * relative_l2(matrix[validation_index], prediction))
        selection_rows.append({
            "coordinate": coordinate,
            "rank": rank,
            "LOO_mean_percent": np.mean(fold_errors),
            "LOO_max_percent": np.max(fold_errors),
        })
selection = pd.DataFrame(selection_rows)
display(selection)
selected_rank = (
    selection.sort_values("LOO_mean_percent")
    .groupby("coordinate", as_index=False).first()
    .set_index("coordinate")["rank"].astype(int).to_dict()
)
print("selected ranks:", selected_rank)
"""),
        md(r"""
## Stop: held-out pressure gate

The representation, split, branch architecture, seed, rank candidates, and
selection statistic are now frozen. Predict which pressure will be hardest.
Then open 16, 25, and 30 kPa exactly once.
"""),
        code(r"""
held_out_indices = np.flatnonzero(held_out)
blind_rows = []
blind_predictions = {}
for coordinate in ("physical", "shock_centered"):
    prediction = fit_predict_pod_branch(
        matrices[coordinate], development_indices, held_out_indices,
        selected_rank[coordinate],
    )
    blind_predictions[coordinate] = prediction
    for local_index, case_index in enumerate(held_out_indices):
        blind_rows.append({
            "coordinate": coordinate,
            "rank": selected_rank[coordinate],
            "pressure_kPa": int(pressure[case_index]),
            "normalized_centerline_relative_L2_percent": 100 * relative_l2(
                matrices[coordinate][case_index], prediction[local_index]
            ),
        })
blind_metrics = pd.DataFrame(blind_rows)
display(blind_metrics)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True, sharey=True)
for axis, local_index, case_index in zip(axes, range(3), held_out_indices):
    axis.plot(grids["shock_centered"], matrices["shock_centered"][case_index],
              color="black", lw=2.2, label="DSMC target")
    axis.plot(grids["shock_centered"], blind_predictions["shock_centered"][local_index],
              color="#C44536", lw=1.8, ls="--", label="POD + neural branch")
    axis.set(title=f"held out: {pressure[case_index]:g} kPa", xlabel=r"$\xi_j$")
    axis.grid(alpha=.25)
axes[0].set_ylabel(r"jump-normalized $\rho$")
axes[0].legend(frameon=False)
plt.show()
"""),
        md(r"""
These are one-dimensional, jump-normalized representation errors. They are
useful for comparing coordinate choices under a matched teaching model, but
they are **not commensurate** with the paper's global 2-D raw-field errors.
The expected pattern is also scientifically richer than “alignment always
wins”: average aligned error decreases, yet the 30 kPa case can remain hard.
Inspecting individual cases prevents a favorable mean from hiding that failure.
"""),
        md(r"""
## 4. Can the shock location be predicted without target CFD?

The structural audit used target-derived $x_s$. A deployable operator cannot
read the unseen density gradient first. Fit an intentionally transparent
pressure-to-location baseline using the 12 development cases only, then test
the three frozen pressures. This is a separate component-level validation.
"""),
        code(r"""
shock_locator = make_pipeline(
    StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1e-6)
).fit(pressure[development, None], data["shock_x_m"][development] * 1e6)
predicted_shock_um = shock_locator.predict(pressure[held_out, None])
shock_table = pd.DataFrame({
    "pressure_kPa": pressure[held_out].astype(int),
    "DSMC_detected_xs_um": data["shock_x_m"][held_out] * 1e6,
    "training_only_predicted_xs_um": predicted_shock_um,
})
shock_table["absolute_error_um"] = abs(
    shock_table["training_only_predicted_xs_um"] - shock_table["DSMC_detected_xs_um"]
)
display(shock_table)
"""),
        md(r"""
The 22 kPa detector outlier remains in development data, so it can degrade this
simple locator. Do not delete it silently. Defensible next steps are to audit
the sensor window, quantify detector uncertainty, predeclare a robust fit, and
repeat the frozen test—not to tune directly on 16, 25, or 30 kPa.
"""),
        md(r"""
## 5. Retained full-model paper evidence

The next tables are read from immutable CSV transcriptions. They describe the
article's full two-dimensional held-out outputs and hard 16 kPa comparison;
the notebook did not regenerate them.
"""),
        code(r"""
paper_fields = pd.read_csv(RESULTS / "nozzle_paper_field_errors.csv")
paper_baselines = pd.read_csv(RESULTS / "nozzle_hard_case_baselines.csv")
display(paper_fields.pivot(
    index="field", columns="held_out_pressure_kpa",
    values="reported_relative_l2_percent",
))
display(paper_baselines)
"""),
        md(r"""
The retained hard-case table shows why global error alone is insufficient:
vanilla models can look less poor globally while missing the shock window. The
shock-aligned Fusion--DeepONet is evaluated with global, shock-window,
gradient-weighted, velocity-window, and shock-location metrics.

The method is physics-guided through representation, localized weighting,
Gaussian region envelopes, and a two-stage curriculum. It does **not** thereby
become a PDE-constrained model: the reported formulation does not explicitly
enforce a PDE residual, Rankine--Hugoniot jump, or global conservation law.

### Required submission

1. the data/provenance audit and held-out pressure list;
2. physical and aligned POD energy tables;
3. leave-one-case-out rank selection;
4. per-pressure held-out profile and shock-location errors;
5. one documented failure or detector ambiguity; and
6. an explicit sentence distinguishing notebook-generated evidence from the
   paper's retained full-model evidence.

### Extension

Use the full public Tecplot snapshots to build a two-dimensional trunk and add
$U,V,T,M, P$ outputs. Freeze a spatial shock window and conservation diagnostics
before opening test cases. A larger network is not automatically stronger when
the number of independent CFD cases remains small.
"""),
    ]


def write_notebook(filename: str, cells) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.write(notebook, HERE / filename)


def main() -> None:
    write_notebook("W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb", lab1_cells())
    write_notebook("W9_Lab2_Shock_Aligned_Nozzle_DeepONet_Student.ipynb", lab2_cells())
    print("Wrote two Week-9 student notebooks.")


if __name__ == "__main__":
    main()
