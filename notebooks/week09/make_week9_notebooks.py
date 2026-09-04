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
# Week 9 Lab 1 — Real micro-step DSMC data, zonal learning, and validation

<!-- MIE690A real-step-data validation v5 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week09/W9_Lab1_Microstep_Zonal_DeepONet_Student.ipynb)

**Runtime:** CPU, normally 1–3 minutes with the included compact data.
**Prerequisites:** case-wise splits, relative error, and neural operators.

This lab turns the Roohi--Mahdavi article *Analysis of the rarefied flow at
micro-step using a DeepONet surrogate model with a physics-guided zonal loss
function* (Microfluidics and Nanofluidics 30:44, published 11 May 2026) into a
controlled classroom experiment.

### Learning outcomes

By the end, you should be able to:

1. distinguish a parameter-to-field operator from pointwise regression;
2. keep every geometry case entirely inside one split;
3. explain why a global mean loss can hide recirculation failure;
4. select a zonal-loss weight using validation cases only;
5. compare article contours with an independently trained classroom model; and
6. distinguish data integrity from independent DSMC verification.
"""),
        md(r"""
## Data and validation contract — read before code

The nine real DSMC height fields originate in the authors'
[`roohi-step-dnn-mahdavi`](https://github.com/Ehsan-Roohi/roohi-step-dnn-mahdavi)
repository at commit `c3f211376b42b8dc30daad380eaef5e0ab800b5c`. With the
corresponding author's explicit permission, FlowMLLab includes two compact
derivatives whose hashes trace back to all nine source files. The seven
development/validation cases and two held-out tests live in different files.

Three evidence levels stay separate:

- **source data:** the nine published, smoothed DSMC fields;
- **retained article evidence:** the paper's MSE/GMSE/zonal table;
- **notebook result:** a new independent coordinate-network baseline trained
  here. It is not the paper's C-DeepONet checkpoint.

The notebook model uses only $h/H$, $(x,y)$, and known geometry. Development,
validation, and held-out geometry cases remain in separate case-wise groups.
"""),
        code(BOOTSTRAP),
        code(r"""
from flowmllab.mahdavi_deeponet import (
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_HELD_OUT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
    STEP_SOURCE_COMMIT,
    evaluate_step_coordinate_surrogate,
    fit_step_coordinate_surrogate,
    load_step_height_archive,
    predict_step_coordinate_surrogate,
)

paper = pd.read_csv(RESULTS / "step_paper_evidence.csv")
display(paper.pivot(index="objective", columns="scope", values="reported_error_percent"))

manifest = json.loads((RESULTS / "step_source_manifest.json").read_text())
learning_heights = np.concatenate([
    STEP_HEIGHT_DEVELOPMENT_PERCENT,
    STEP_HEIGHT_VALIDATION_PERCENT,
])
learning_cases = load_step_height_archive(REPO_ROOT, split="learning")
first_case = learning_cases[int(STEP_HEIGHT_DEVELOPMENT_PERCENT[0])]
bounds_m = (
    float(first_case["x"].min()), float(first_case["x"].max()),
    float(first_case["y"].min()), float(first_case["y"].max()),
)
print("source commit:", STEP_SOURCE_COMMIT)
print("opened archive: step_height_learning_7cases.npz only")
print("development:", STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist())
print("validation:", STEP_HEIGHT_VALIDATION_PERCENT.tolist())
print("sealed test:", STEP_HEIGHT_HELD_OUT_PERCENT.tolist())
"""),
        md(r"""
## 1. Start with the real flow and its geometry

The parent grid has 200 streamwise by 120 transverse locations. Points inside
the solid step are absent, so the number of fluid points changes with $h/H$.
The plot below uses equal physical axis scaling: the step is not stretched to
make the recirculation region look larger.
"""),
        code(r"""
def on_parent_grid(case, values):
    xs, ys = np.unique(case["x"]), np.unique(case["y"])
    field = np.full((len(ys), len(xs)), np.nan)
    ix = np.searchsorted(xs, case["x"])
    iy = np.searchsorted(ys, case["y"])
    field[iy, ix] = values
    return xs, ys, field

example = learning_cases[50]
xs, ys, u_grid = on_parent_grid(example, example["u"])
fig, ax = plt.subplots(figsize=(11.0, 3.1), constrained_layout=True)
image = ax.pcolormesh(xs * 1e9, ys * 1e9, np.ma.masked_invalid(u_grid),
                      shading="nearest", cmap="coolwarm")
ax.contour(xs * 1e9, ys * 1e9, np.ma.masked_invalid(u_grid),
           levels=[0.0], colors="black", linewidths=1.0)
ax.set(xlabel="x (nm)", ylabel="y (nm)", title=r"Real smoothed DSMC field, $h/H=0.50$")
ax.set_aspect("equal", adjustable="box")
fig.colorbar(image, ax=ax, label="U (source units)")
plt.show()
"""),
        md(r"""
## 2. What is—and is not—validated about the DSMC labels

SHA-256, finite values, row counts, and grid topology establish **data
integrity**. They do not independently establish DSMC convergence. The public
repository does not yet document the following items at a level that this
course can independently reproduce:

1. cell size relative to local mean free path;
2. time step relative to local collision time;
3. particles per cell;
4. sampling duration, independent seeds, and confidence intervals;
5. statistical uncertainty near separation/reattachment; and
6. wall reflection model and accommodation coefficients.

Students must report these as provenance gaps—not silently invent values. Also,
the two studies vary Kn at fixed geometry and $h/H$ at fixed Kn=0.01. Joint
$(Kn,h/H)$ generalization has not been demonstrated.
"""),
        code(r"""
provenance_gaps = pd.Series(manifest["dsmc_provenance_status"], name="status")
display(provenance_gaps.to_frame())
print("scope:", manifest["study_scope"])
"""),
        md(r"""
## 3. Define an independent coordinate surrogate

For a geometry parameter $h/H$ and query coordinate $\mathbf y=(x,y)$, a
standard DeepONet has the form

$$
\widehat{G}(h/H)(\mathbf{y})=
\sum_{k=1}^{r} b_k(h/H)\,t_k(\mathbf{y})+b_0.
$$

Here we use a small coordinate MLP as a transparent CPU baseline. Its features
contain only the height ratio, normalized coordinates, and a wall-relative
coordinate computed from the known step geometry. The required assignment
later replaces this baseline with a strict branch/trunk dot product.

The paper defines the recirculation zone from the reference streamwise
velocity, $U<0$, and balances two separately normalized regional errors:

$$
\mathcal L_{\rm zonal}=\alpha\mathcal L_{U<0}+
(1-\alpha)\mathcal L_{U\ge 0}.
$$

For the CPU baseline, exact zonal weighting is implemented by deterministic
stratified resampling: a fraction $\alpha$ of optimizer samples comes from
$U<0$ and $1-\alpha$ from the main-flow zone. Regional weights are estimated
only from development cases.
"""),
        code(r"""
def fit_coordinate_surrogate(selected_heights, alpha, *, seed=690, sample_size=60_000):
    return fit_step_coordinate_surrogate(
        learning_cases, selected_heights, alpha, bounds_m=bounds_m,
        seed=seed, sample_size=sample_size,
    )


def predict_case(fitted, height, case):
    return predict_step_coordinate_surrogate(fitted, int(height), case)


def evaluate_model(fitted, selected_heights, case_store):
    return pd.DataFrame(evaluate_step_coordinate_surrogate(
        fitted, case_store, selected_heights,
    ))
"""),
        md(r"""
## 4. Freeze the case-wise split and selection rule

The source dataset contains exactly nine heights. We reserve H44 and H67 for
the final test, use H33 and H58 for validation, and fit candidate settings only
on H16, H21, H25, H50, and H75. Every point from a geometry remains in one
split. This is stricter than randomly splitting points from the same cases.

**Predeclared rule:** among $\alpha\in\{0.5,0.6,0.7,0.8\}$, choose the
largest vortex improvement whose validation global relative error is no more
than two percentage points worse than the unweighted fit. This prevents a
zonal win purchased by an unlimited global failure.
"""),
        code(r"""
baseline_fit = fit_coordinate_surrogate(STEP_HEIGHT_DEVELOPMENT_PERCENT, None)
baseline_validation = evaluate_model(
    baseline_fit, STEP_HEIGHT_VALIDATION_PERCENT, learning_cases,
)
baseline_global = 100 * baseline_validation["full_relative_l2"].mean()
selection_rows = []
candidate_models = {}
for alpha in (.5, .6, .7, .8):
    candidate_models[alpha] = fit_coordinate_surrogate(
        STEP_HEIGHT_DEVELOPMENT_PERCENT, alpha,
    )
    metrics = evaluate_model(
        candidate_models[alpha], STEP_HEIGHT_VALIDATION_PERCENT, learning_cases,
    )
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
if selected_alpha != 0.6:
    print("Version-sensitive teaching result: the article used alpha=0.6;",
          "this run selected", selected_alpha)
"""),
        md(r"""
## Stop: held-out geometry gate

At this point the split, architecture, optimizer, sample budget, seed, candidate
weights, and selection rule are frozen. Write your expected global and vortex
errors for H44 and H67. Only then run the next cell. If any choice changes after
viewing these fields, they are no longer held out.
"""),
        code(r"""
final_unweighted = fit_coordinate_surrogate(learning_heights, None)
final_zonal = fit_coordinate_surrogate(learning_heights, selected_alpha)
held_out_cases = load_step_height_archive(REPO_ROOT, split="test")
print("opened archive after freeze: step_height_test_2cases.npz")
unweighted_test = evaluate_model(
    final_unweighted, STEP_HEIGHT_HELD_OUT_PERCENT, held_out_cases,
)
zonal_test = evaluate_model(
    final_zonal, STEP_HEIGHT_HELD_OUT_PERCENT, held_out_cases,
)
comparison = pd.concat([
    unweighted_test.assign(model="unweighted"),
    zonal_test.assign(model=f"zonal alpha={selected_alpha:.1f}"),
], ignore_index=True)
display(comparison[["model", "height_percent", "full_relative_l2", "vortex_relative_l2"]])

fig, axes = plt.subplots(2, 3, figsize=(13.2, 4.7), constrained_layout=True)
for row, height in enumerate(STEP_HEIGHT_HELD_OUT_PERCENT):
    case = held_out_cases[int(height)]
    prediction = predict_case(final_zonal, height, case)
    panels = [case["u"], prediction[:, 0], np.abs(prediction[:, 0] - case["u"])]
    titles = ["DSMC target U", "independent prediction U", "absolute U error"]
    for axis, values, title in zip(axes[row], panels, titles):
        gx, gy, field = on_parent_grid(case, values)
        artist = axis.pcolormesh(gx * 1e9, gy * 1e9, np.ma.masked_invalid(field),
                                 shading="nearest", cmap="coolwarm" if "error" not in title else "magma")
        axis.set_title(f"H{int(height)} — {title}")
        axis.set(xlabel="x (nm)", ylabel="y (nm)")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(artist, ax=axis, shrink=.78)
plt.show()
"""),
        md(r"""
## 5. Reproduce the article cases and compare validation levels

Under the final published paper's numbering, the pinned source checkout retains
DSMC and stored NN fields for Figure 6 at Kn=0.004 and Kn=0.02 and for Figure
15 at H44 and H67. Each reconstruction uses the same coordinates, equal
$x/H$--$y/H$ scaling, a masked solid step, and one color range shared by DSMC
and NN for each velocity component. Error panels are clipped only for display
at their 99th percentile; the CSV retains the maximum error.

Figure 6 also contains Kn=1. The exact DSMC source field is included below, but
the neural panel is not reproducible from the pinned repository because its
stored prediction is absent. The coverage table records that gap explicitly;
no values are inferred from the published raster image.

The first table and images are **retained article evidence**. The second table
and images are the independent H44/H67 result from the frozen classroom model.
Compare topology, the $U=0$ recirculation boundary, reverse-flow IoU, and
reattachment length—not merely color similarity.

Two source inconsistencies remain explicit:

- the article problem statement lists Kn=0.2, while Figure 6, its discussion,
  the source code, and the stored output use Kn=0.02;
- Figure 6 shows Kn=1, but the pinned repository contains no stored Kn=1
  prediction, so this lesson rebuilds only its DSMC contour and does not
  fabricate the neural comparison.
"""),
        code(r"""
article_contours = pd.read_csv(RESULTS / "step_article_contour_metrics.csv")
article_coverage = pd.read_csv(RESULTS / "step_article_case_coverage.csv")
independent_contours = pd.read_csv(RESULTS / "step_independent_contour_metrics.csv")
columns = [
    "case_id", "combined_relative_l2_percent", "vortex_relative_l2_percent",
    "negative_u_iou_percent", "dsmc_reattachment_length_over_L",
]
display(article_contours[columns + ["stored_nn_reattachment_length_over_L"]])
display(article_coverage)
display(independent_contours[columns + ["independent_reattachment_length_over_L"]])

try:
    from IPython.display import Image as _ContourImage
except ModuleNotFoundError:
    _ContourImage = None
if _ContourImage is not None:
    for filename in (
        "step_article_contours/article_figure_06_Kn0p004.png",
        "step_article_contours/article_figure_06_Kn0p02.png",
        "step_article_contours/article_figure_06_Kn1_DSMC_only.png",
        "step_article_contours/article_figure_15_H44.png",
        "step_article_contours/article_figure_15_H67.png",
        "step_independent_contours/held_out_H44_independent.png",
        "step_independent_contours/held_out_H67_independent.png",
    ):
        display(_ContourImage(filename=str(RESULTS / filename)))
"""),
        md(r"""
## 6. Interpret without mixing evidence levels

The real-data classroom experiment should show the intended mechanism: the
selected zonal objective reduces reverse-flow error while accepting a modest
increase in whole-field error. The paper reports the same qualitative tradeoff:
zonal loss changes the reported recirculation-zone error from
14.6135% (MSE) to 11.9413%, while the full-domain value changes from 2.1739%
to 2.2254%.

Those four percentages come from the article table. They are not the notebook
MLP errors and are not the upstream SWAG stored-prediction errors.

### DeepONet implementation exercise

Replace the coordinate MLP with two small Keras networks:

- branch input: one scalar, $h/H$;
- trunk input: two scalars, $(x/H,y/H)$;
- output: dot product of equal-width branch and trunk vectors, with separate
  heads for $U$ and $V$.

Keep complete geometry cases together. Implement regional means before mixing
them with fixed $\alpha$; adaptive $\alpha$ is a separate experiment, not the
fixed-loss paper protocol.

### Required submission

1. the source commit and nine verified hashes;
2. a signed, case-wise split table;
3. the validation-only $\alpha$ sweep;
4. global and reverse-flow metrics for every held-out geometry;
5. one equal-aspect contour locating the largest local error;
6. a branch/trunk diagram and complete case-wise split table; and
7. a DSMC verification checklist that labels unavailable setup values as
   unavailable.
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

This lab uses compact full-field and centerline derivatives of **all 15 real
public DSMC snapshots**
from the Roohi--Mahdavi article *Shock-centered low-rank structure and
shock-aligned surrogate modeling of rarefied micro-nozzle flows* (Physics of
Fluids 38, 082008, 2026; DOI `10.1063/5.0343101`). It asks why a moving shock
is high-rank in laboratory coordinates and low-rank in shock-centered coordinates.

### Learning outcomes

1. audit snapshot provenance and source hashes before learning;
2. detect a centerline compression station and identify noisy outliers;
3. reproduce the 15-snapshot physical versus shock-centered density POD audit;
4. select POD rank by leave-one-case-out development error;
5. open pressures 16, 25, and 30 kPa only after freezing the model;
6. generate fresh 2-D density, velocity, Mach, and pressure predictions; and
7. distinguish the classroom operator from the article's six-output model.
"""),
        md(r"""
## Evidence and claim contract

- `nozzle_fields_15cases.npz` and `nozzle_centerline_15cases.npz` are derived
  from the public Tecplot DSMC files at pinned commit
  `e1b234ba499408d3b6224633972f939f3b2301d6` and remain CC BY 4.0.
- The POD spectrum is directly reproducible from those data.
- The notebook first predicts jump-normalized centerline density, then runs a
  fresh full-field POD trunk plus neural branch for density, $U$, Mach, and
  pressure. This is the FlowMLLab classroom operator, not a stored article
  checkpoint.
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
public_report = {
    "status": report["status"],
    "nozzle_cases": report["nozzle_cases"],
    "held_out_pressures_kpa": report["held_out_pressures_kpa"],
    "physical_density_pod": report["physical_density_pod"],
    "shock_centered_density_pod": report["shock_centered_density_pod"],
    "full_field_validation": {
        "status": report["nozzle_flowmllab_validation"]["status"],
        "development_pressures_kpa": report["nozzle_flowmllab_validation"][
            "development_pressures_kpa"
        ],
        "held_out_pressures_kpa": report["nozzle_flowmllab_validation"][
            "held_out_pressures_kpa"
        ],
        "selected_rank_by_field": report["nozzle_flowmllab_validation"][
            "selected_rank_by_field"
        ],
    },
}
print(json.dumps(public_report, indent=2))
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
## 5. Generate fresh full-field FlowMLLab predictions

The following cell runs the repository's validation program. For each of
density, $U$, Mach number, and pressure, it selects a POD rank by
leave-one-case-out error on the 12 development pressures, fits an eight-unit
tanh neural branch, and only then evaluates 16, 25, and 30 kPa. The POD modes
are the spatial trunk. Nothing is read from a paper raster or a stored neural
prediction.

The program writes a machine-readable selection table, the 12 held-out error
rows, an error heatmap, a 4-by-3 centerline comparison, and a full 2-D contour
comparison for the hard 16 kPa case.
"""),
        code(r"""
command = [
    _flowmllab_sys.executable,
    str(REPO_ROOT / "qa/run_nozzle_field_validation.py"),
    "--root", str(REPO_ROOT),
]
_flowmllab_subprocess.run(command, check=True)
generated_metrics = pd.read_csv(RESULTS / "nozzle_flowmllab_heldout_metrics.csv")
display(generated_metrics.pivot(
    index="field", columns="held_out_pressure_kpa",
    values="full_field_relative_l2_percent",
))

try:
    from IPython.display import Image as _NozzleImage
except ModuleNotFoundError:
    _NozzleImage = None
if _NozzleImage is not None:
    for filename in (
        "nozzle_flowmllab/nozzle_back_pressure_P16_contours.png",
        "nozzle_flowmllab/nozzle_back_pressure_profiles.png",
        "nozzle_flowmllab/nozzle_back_pressure_error_summary.png",
    ):
        display(_NozzleImage(filename=str(RESULTS / filename)))
"""),
        md(r"""
## 6. Retained full-model paper evidence

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
The retained hard-case table shows why global error alone is insufficient. In
the final article's three-seed comparison, the Cartesian Hadamard branch/trunk
model has $34.95\pm20.06\%$ shock-window error. Adding the reduced signed
distance lowers it to $9.12\pm1.01\%$, comparable to the strong Cartesian MLP's
$8.86\pm1.26\%$. The defensible claim is therefore not universal superiority.
The signed-distance representation removes a major translation burden for the
branch/trunk model; the final article explicitly does not claim a new fusion
architecture.

The method is physics-guided through representation, localized weighting, and
shock-envelope features. It does **not** thereby become a PDE-constrained
model: the reported formulation does not explicitly enforce a PDE residual,
Rankine--Hugoniot jump, or global conservation law.

### Required submission

1. the data/provenance audit and held-out pressure list;
2. physical and aligned POD energy tables;
3. leave-one-case-out rank selection;
4. per-pressure held-out profile and shock-location errors;
5. the fresh 2-D FlowMLLab contour and full-field error table;
6. one documented failure or detector ambiguity; and
7. an explicit sentence distinguishing notebook-generated evidence from the
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
