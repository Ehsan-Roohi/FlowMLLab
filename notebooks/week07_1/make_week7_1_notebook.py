"""Build the Week-7.1 student notebook from reviewable source cells."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


BOOTSTRAP = r'''
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
    _flowmllab_os.chdir(_flowmllab_root / "notebooks/week07_1")

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from flowmllab.hypersonic_cylinder import (
    TARGET_NAMES,
    case_interpolation_baseline,
    casewise_split_masks,
    ensemble_predict,
    fit_separable_ridge_ensemble,
    load_cylinder_teaching_data,
    relative_l2,
    validate_hypersonic_cylinder_evidence,
)

REPO_ROOT = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "data/hypersonic_cylinder/manifest.json").is_file()
)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11})
print("FlowMLLab root:", REPO_ROOT)
'''


CELLS = [
    markdown(r'''
# Week 7.1 - Rarefied hypersonic cylinder operator learning

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07_1/W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb)

**Runtime:** CPU, normally under 2 minutes. **Placement:** incremental Week 7.1,
after the continuum/LBM cylinder lab and before Week 8 gas dynamics.

### Central question

When a parameterized DSMC field lives on the same cylinder grid, what evidence
is required before a neural operator is more useful than direct interpolation?

### Learning outcomes

1. distinguish freestream Mach from the predicted local Mach field;
2. formulate the cylinder map as an operator $G:M_\infty\mapsto q(x,y)$;
3. enforce whole-case training, validation, interpolation, and extrapolation splits;
4. explain branch, trunk, layerwise fusion, standardized targets, and weighted loss;
5. compare a strong structured interpolation baseline with a fast separable teaching model; and
6. interpret deep-ensemble spread as a diagnostic, not guaranteed uncertainty calibration.
'''),
    markdown(r'''
## Evidence and reuse contract

The compact fields are an author-released derivative of the DSMC archive used
for the hypersonic-cylinder part of Roohi et al., *Physics of Fluids* **38**,
057108 (2026), [doi:10.1063/5.0334590](https://doi.org/10.1063/5.0334590).
The committed NPZ contains 44,500 deterministic points from 20 cases; the
1.4 GB archive, logs, checkpoints, and 116 historical scripts are not copied.

This lab does **not** report the published full-resolution accuracy. Its default
ridge ensemble is a CPU teaching analog. The optional TensorFlow builder exposes
the reviewed Fusion-DeepONet topology, but full reproduction requires the
original 50,000-point sampling, five trained networks, frozen protocol, and
adequate compute.
'''),
    code(BOOTSTRAP),
    code(r'''
report = validate_hypersonic_cylinder_evidence(REPO_ROOT)
display(pd.DataFrame({
    "item": ["cases", "points", "targets", "paper DOI", "source archive SHA-256"],
    "value": [report["cases"], report["points"], ", ".join(report["targets"]),
              report["paper_doi"], report["source_archive_sha256"]],
}))
data = load_cylinder_teaching_data(REPO_ROOT)
'''),
    markdown(r'''
## 1. Physics before machine learning

The control parameter $M_\infty=U_\infty/a_\infty$ is not the same quantity as
the spatial target $M(x,y)$. Rarefaction is governed by $Kn=\lambda/D$; when the
mean free path is not negligible relative to the cylinder diameter, continuum
closure and no-slip assumptions need qualification. DSMC estimates moments of
a particle distribution, so its fields also carry sampling noise.

The three nondimensional targets are

$$q(x,y;M_\infty)=\left[M(x,y),\;T(x,y)/T_\infty,\;p(x,y)/p_\infty\right].$$

**Prediction prompt:** Where should the largest interpolation error occur: in
the freestream, inside the shock layer, or near the surface? Record your reason
before plotting.
'''),
    code(r'''
case = np.isclose(data.mach_inf, 8.5)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
labels = (r"local $M$", r"$T/T_\infty$", r"$p/p_\infty$")
for j, (axis, label) in enumerate(zip(axes, labels)):
    artist = axis.tricontourf(data.x[case], data.y[case], data.targets[case, j],
                              levels=28, cmap="viridis")
    axis.set(title=rf"$M_\infty=8.5$: {label}", xlabel="x", ylabel="y", aspect="equal")
    fig.colorbar(artist, ax=axis, shrink=0.82)
plt.show()
'''),
    markdown(r'''
## 2. A split is a scientific statement

Randomly splitting points would put the same DSMC flow case in training and
test sets. That tests spatial interpolation inside a known solution, not
generalization to an unseen operating condition. We instead freeze whole Mach
cases:

- **train:** integer $M_\infty=5,\ldots,14$;
- **validation:** 8.25, 8.75, 9.25, 9.75;
- **blind interpolation:** 5.5, 6.5, 7.5, 8.5, 9.5;
- **blind extrapolation:** 15.

Validation may guide design. Blind sets are opened once for reporting.
'''),
    code(r'''
masks = casewise_split_masks(data.mach_inf)
split_table = []
for name, mask in masks.items():
    split_table.append({"split": name, "cases": np.unique(data.mach_inf[mask]).tolist(),
                        "points": int(mask.sum())})
display(pd.DataFrame(split_table))
assert not any(np.any(masks[a] & masks[b]) for i, a in enumerate(masks)
               for b in list(masks)[i + 1:])
'''),
    markdown(r'''
## 3. Mandatory strong baseline

All cases share source-grid row identifiers, so a direct interpolation between
the two bracketing training Mach fields is cheap and physically transparent.
For $M_a<M_q<M_b$,

$$\hat q(M_q)=q(M_a)+\frac{M_q-M_a}{M_b-M_a}\,[q(M_b)-q(M_a)].$$

A neural operator must be compared with this baseline, not only with a trivial
constant or mean field.
'''),
    code(r'''
baseline_rows = []
baseline_predictions = {}
for split_name in ("validation", "interpolation", "extrapolation"):
    mask = masks[split_name]
    prediction = case_interpolation_baseline(data, masks["train"], mask)
    baseline_predictions[split_name] = prediction
    errors = relative_l2(data.targets[mask], prediction)
    baseline_rows.append({"split": split_name, **dict(zip(TARGET_NAMES, errors))})
display(pd.DataFrame(baseline_rows).style.format({name: "{:.3%}" for name in TARGET_NAMES}))
'''),
    markdown(r'''
## 4. Fusion-DeepONet anatomy

The scalar branch sees $M_\infty$; the trunk sees $(x,y)$. Four 256-unit tanh
layers form each path. In the reviewed stored topology, branch features are
injected into the corresponding trunk layer, dropout regularizes both paths,
the final branch/trunk states are combined by an inner product, and a linear
three-output head returns the standardized targets.

Only training cases may fit input and output scalers. The research workflow
uses Adam, early stopping, learning-rate reduction, and five independent fits.
One reviewed script weights standardized pressure error five times more than
the other two components:

$$L=\operatorname{MSE}(M)+\operatorname{MSE}(T/T_\infty)+5\operatorname{MSE}(p/p_\infty).$$

Because historical archive scripts disagree in several settings, this notebook
records the reviewed topology and avoids presenting any one script filename as
the canonical paper release.
'''),
    markdown(r'''
## 5. Fast separable teaching analog

To keep the required path CPU-only, we freeze random tanh branch and trunk
features, multiply them elementwise, and fit ridge coefficients. This preserves
the separable operator idea but **is not a trained DeepONet**. Its purpose is to
make the split, baseline, ensemble, and failure analysis executable.

**Prediction prompt:** Will this low-capacity analog beat field interpolation?
Why might a more complex model still be valuable for unstructured meshes,
multiple geometry parameters, or unavailable bracketing cases?
'''),
    code(r'''
started = time.perf_counter()
ensemble = fit_separable_ridge_ensemble(
    data, masks["train"], members=5, latent_dim=64, alpha=1e-3, seed=760
)
print(f"Five-member CPU fit: {time.perf_counter() - started:.2f} s")

operator_rows = []
predictions = {}
spreads = {}
for split_name, mask in masks.items():
    mean, spread = ensemble_predict(
        ensemble, data.mach_inf[mask], data.x[mask], data.y[mask]
    )
    predictions[split_name], spreads[split_name] = mean, spread
    errors = relative_l2(data.targets[mask], mean)
    coverage = np.mean(np.abs(data.targets[mask] - mean) <= 2 * spread, axis=0)
    operator_rows.append({
        "split": split_name,
        **{f"L2 {name}": value for name, value in zip(TARGET_NAMES, errors)},
        **{f"2sigma {name}": value for name, value in zip(TARGET_NAMES, coverage)},
    })
display(pd.DataFrame(operator_rows))
'''),
    markdown(r'''
## 6. Error localization and uncertainty honesty

Ensemble spread is epistemic disagreement under the selected architecture and
training procedure. It is not automatically a calibrated probability interval.
Report empirical coverage of $\mu\pm2s$ and inspect where the error is large.
Low coverage means the ensemble is overconfident; increasing its member count
does not by itself fix model bias.
'''),
    code(r'''
mask = masks["interpolation"] & np.isclose(data.mach_inf, 8.5)
all_interpolation_indices = np.flatnonzero(masks["interpolation"])
case_positions = np.flatnonzero(np.isclose(data.mach_inf[masks["interpolation"]], 8.5))
truth = data.targets[mask]
baseline = baseline_predictions["interpolation"][case_positions]
operator = predictions["interpolation"][case_positions]

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7), constrained_layout=True)
for j, label in enumerate(labels):
    for row, (title, estimate) in enumerate((("Mach interpolation", baseline),
                                             ("teaching operator", operator))):
        error = np.abs(estimate[:, j] - truth[:, j])
        artist = axes[row, j].tricontourf(data.x[mask], data.y[mask], error,
                                          levels=25, cmap="magma")
        axes[row, j].set(title=f"{title}: |error| {label}", xlabel="x", ylabel="y",
                         aspect="equal")
        fig.colorbar(artist, ax=axes[row, j], shrink=0.8)
plt.show()
'''),
    markdown(r'''
## 7. Optional full topology - not the default runtime

If TensorFlow is available through `pip install -e ".[ml]"`, the following
constructs the reviewed architecture. Keep it disabled unless you have defined
a full training budget and frozen the validation protocol.
'''),
    code(r'''
RUN_FULL_NEURAL = False
if RUN_FULL_NEURAL:
    from flowmllab.hypersonic_cylinder import build_fusion_deeponet
    model = build_fusion_deeponet(latent_dim=256, hidden_layers=4, dropout_rate=0.2)
    model.summary()
else:
    print("Topology build skipped; the CPU evidence path above is complete.")
'''),
    markdown(r'''
## 8. Conclusions and required submission

1. Give the three interpolation and three extrapolation relative-$L_2$ errors
   for both the strong baseline and the teaching operator.
2. Identify one region where each method fails and connect it to cylinder-flow physics.
3. Report two-sigma coverage per target and state whether the ensemble is calibrated.
4. Explain why a random point split would inflate the scientific claim.
5. Propose one change that could justify a full Fusion-DeepONet experiment.

### Claim boundary

You may claim that the compact author-released DSMC derivative and frozen split
were executed. You may not claim reproduction of the paper's model accuracy,
speedup, full-resolution fields, or uncertainty calibration from this notebook.
'''),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = HERE / "W7_1_Hypersonic_Rarefied_Cylinder_DeepONet.ipynb"
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
