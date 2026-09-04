#!/usr/bin/env python3
"""Build the complete Week-10 article-reproduction teaching notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "W10_DSMC_Data_Driven_Surrogates_Student.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


cells = [
    markdown("""# Week 10 - DSMC data-driven surrogates: reproduce, test, and critique

MIE690A article-aligned validation v4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week10/W10_DSMC_Data_Driven_Surrogates_Student.ipynb)

This lab rebuilds numerical results from Roohi and Shoja-Sani, *Data-driven surrogate modeling of DSMC solutions using deep neural networks*, Aerospace Science and Technology 168 (2026) 110785, DOI `10.1016/j.ast.2025.110785`.

**Learning outcomes**

1. explain the DSMC move-collide-sample cycle and its resolution checks;
2. audit complete raw numerical tables and their checksums;
3. reproduce rarefied-cavity predictions at unseen Knudsen numbers;
4. build a branch-trunk operator for monatomic and diatomic shocks;
5. distinguish interpolation from extrapolation and representation error from field error; and
6. report unavailable targets instead of replacing them with a plotted image.

Runtime: CPU, normally under two minutes after installation."""),
    markdown("""## Scientific contract

The raw cavity and shock tables are committed under `data/aescte_dsmc/raw/`. The notebook never digitizes a paper raster. Training cases and assessment cases remain case-wise separated.

The attached cavity fields contain lid speeds **10 and 30 m/s**. Although the paper also discusses 100 m/s, that raw target was not supplied. The diatomic archive ends at Mach 1.9; therefore the published Mach-2 plot cannot be independently scored here. These boundaries are part of the result."""),
    code("""# FLOWMLLAB_COLAB_BOOTSTRAP_V1
# Colab/local bootstrap
from pathlib import Path as _Path
import os as _os, subprocess as _subprocess, sys as _sys

if "google.colab" in _sys.modules:
    _root = _Path("/content/FlowMLLab")
    if not (_root / ".git").is_dir():
        _subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(_root)], check=True)
    _subprocess.run([_sys.executable, "-m", "pip", "install", "-q", "-e", str(_root)], check=True)
    _os.chdir(_root / "notebooks/week10")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

REPO_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
RESULTS = REPO_ROOT / "results/aescte_dsmc"
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))
print("FlowMLLab:", REPO_ROOT)
print("Python:", _sys.version.split()[0])"""),
    markdown("""## 1. What DSMC computes

For each time step, a DSMC solver:

1. moves representative particles ballistically;
2. applies wall/inlet/outlet interactions;
3. selects collision pairs locally (Bird NTC here);
4. redistributes translational and internal energy using a molecular model; and
5. samples macroscopic moments only after the transient.

A defensible dataset requires cell size relative to mean free path, time step relative to collision time, particles per cell, sampling duration, and repeat-seed uncertainty. A visually smooth contour is not itself CFD validation."""),
    code("""from flowmllab.aescte_dsmc import (
    CAVITY_LID_SPEEDS_MS, CAVITY_TEST_KNUDSEN,
    cavity_case, fit_pod_polynomial_operator,
    load_cavity_archive, load_shock_archive,
    logarithmic_kn_prediction, maxwell_speed_pdf,
    normalized_rmse, predict_pod_polynomial_operator,
    relative_l2, sha256,
)

manifest = json.loads((RESULTS / "data_manifest.json").read_text())
display(pd.DataFrame([manifest["raw_data_contract"]]).T.rename(columns={0: "value"}))
print("tracked raw files:", len(manifest["source_file_sha256"]))
print("manifest SHA-256:", sha256(RESULTS / "data_manifest.json"))"""),
    markdown(r"""## 2. Rarefied cavity: the family-of-specialists construction

Specialist solutions are available at

\[Kn=10^{-3},10^{-2},10^{-1},1,10.\]

For a held-out value bracketed by \(Kn_l\) and \(Kn_u\), the article uses

\[w=\frac{\log_{10}Kn_* - \log_{10}Kn_l}{\log_{10}Kn_u-\log_{10}Kn_l},\qquad
\widehat q_*=(1-w)q_l+wq_u.\]

Here the fixed-Kn spatial specialists are represented by their complete converged DSMC fields. This isolates and exactly tests the parametric synthesis step."""),
    code("""cavity = load_cavity_archive(RESULTS / "cavity_fields_14cases.npz")
print("temperature tensor:", cavity["temperature_k"].shape)
print("lid speeds:", np.unique(cavity["lid_speed_ms"]))
print("Kn values:", np.unique(cavity["knudsen"]))

cavity_rows = []
cavity_predictions = {}
for lid in CAVITY_LID_SPEEDS_MS:
    for kn in CAVITY_TEST_KNUDSEN:
        pred = logarithmic_kn_prediction(cavity, lid_speed_ms=lid, test_knudsen=kn)
        cavity_predictions[(int(lid), float(kn))] = pred
        target = cavity_case(cavity, lid, kn)
        for field, scale in (("u_ms", lid), ("v_ms", lid), ("temperature_k", 50.0)):
            cavity_rows.append({
                "U_lid_m/s": int(lid), "held_out_Kn": kn, "field": field,
                "NRMSE_percent": normalized_rmse(cavity[field][target], pred[field], scale),
            })
cavity_metrics = pd.DataFrame(cavity_rows)
display(cavity_metrics)
assert cavity_metrics.NRMSE_percent.max() < 2.0"""),
    code("""display(Image(filename=str(RESULTS / "cavity_kn005_reproduction.png")))
display(Image(filename=str(RESULTS / "cavity_validation_profiles.png")))"""),
    markdown(r"""### Interpret the metric

Velocity RMSE is normalized by lid speed and temperature RMSE by the 50 K wall-temperature difference. Component-wise relative \(L_2\) can look large for \(V\) because its reference norm is small. Always state the denominator.

The retained primary-variable gate is below 2% for every supplied held-out cavity case. Heat flux and stress are also reported in the machine-readable CSV, but are harder because DSMC moment noise grows with moment order."""),
    markdown(r"""## 3. Diatomic nitrogen shock: delayed internal relaxation

The six source profiles correspond to Mach 1.4--1.9. The original filenames (`M14.5` ... `M19.5`) are preserved; the trailing `.5` is a DSMC internal-setting suffix, not Mach 14.5.

The teaching operator uses POD modes as a trunk \(t_k(x)\) and a polynomial Mach branch \(b_k(M)\):

\[\widehat q(M,x)=\bar q(x)+\sum_{k=1}^{r} b_k(M)t_k(x).\]

Mach 1.7 is removed for interpolation. Mach 1.4 is then removed and predicted from Mach 1.5--1.9 to demonstrate one-sided extrapolation."""),
    code("""diatomic = load_shock_archive(RESULTS / "diatomic_shock_6cases.npz")
shock_fields = ["rotational_temperature", "translational_temperature", "normalized_velocity"]

def predict_shock(test_mach, training_mach, rank, degree):
    train = np.flatnonzero(np.isin(np.round(diatomic["mach"], 8), np.round(training_mach, 8)))
    out = {}
    for field in shock_fields:
        model = fit_pod_polynomial_operator(diatomic["mach"], diatomic[field], train, rank=rank, degree=degree)
        out[field] = predict_pod_polynomial_operator(model, [test_mach])[0]
    return out

pred_17 = predict_shock(1.7, [1.4, 1.5, 1.6, 1.8, 1.9], rank=4, degree=3)
pred_14 = predict_shock(1.4, [1.5, 1.6, 1.7, 1.8, 1.9], rank=4, degree=2)

rows = []
for mach, pred, kind in ((1.7, pred_17, "interpolation"), (1.4, pred_14, "extrapolation")):
    target = int(np.flatnonzero(np.isclose(diatomic["mach"], mach))[0])
    for field in shock_fields:
        rows.append({"task": kind, "Mach": mach, "field": field,
                     "relative_L2_percent": relative_l2(diatomic[field][target], pred[field])})
shock_metrics = pd.DataFrame(rows)
display(shock_metrics)
assert shock_metrics.relative_L2_percent.max() < 1.1"""),
    code("""display(Image(filename=str(RESULTS / "diatomic_shock_reproduction.png")))"""),
    markdown(r"""The translational-temperature overshoot is the key nonequilibrium feature: translational energy rises rapidly at the shock and then relaxes into the rotational mode. Matching only upstream and downstream plateaus would miss this physics.

The result is strongest for interpolation. Mach-1.4 extrapolation remains below 1.1% profile-relative \(L_2\), but the Mach-2 article case is marked unavailable because the target table is absent."""),
    markdown("""## 4. Monatomic shock and Maxwell equilibrium checks

The second shock archive covers Mach 1.4--2.0. We remove Mach 1.7 and predict density, velocity, and temperature. Separately, the Maxwell speed PDF must integrate to one before it is used as an equilibrium target."""),
    code("""monatomic = load_shock_archive(RESULTS / "monatomic_shock_7cases.npz")
mono_fields = ["density", "velocity", "temperature"]
train = np.flatnonzero(~np.isclose(monatomic["mach"], 1.7))
target = int(np.flatnonzero(np.isclose(monatomic["mach"], 1.7))[0])
mono_pred = {}
for field in mono_fields:
    model = fit_pod_polynomial_operator(monatomic["mach"], monatomic[field], train, rank=5, degree=3)
    mono_pred[field] = predict_pod_polynomial_operator(model, [1.7])[0]

display(pd.DataFrame({field: [relative_l2(monatomic[field][target], mono_pred[field])] for field in mono_fields}, index=["relative L2 (%)"]))

speed = np.linspace(0, 3000, 10000)
argon_mass = 39.948e-3 / 6.02214076e23
pdf_325 = maxwell_speed_pdf(speed, 325.0, argon_mass)
integral = np.trapezoid(pdf_325, speed)
print("Maxwell PDF integral at unseen T=325 K:", integral)
assert abs(integral - 1.0) < 1e-10"""),
    code("""display(Image(filename=str(RESULTS / "monatomic_relaxation_reproduction.png")))"""),
    markdown(r"""## 5. One-command reproduction and retained evidence

The following program rebuilds every metric and figure shown in this notebook. It stops if any primary cavity NRMSE exceeds 2% or any retained shock-profile relative \(L_2\) exceeds 1.5%."""),
    code("""command = [_sys.executable, str(REPO_ROOT / "qa/run_week10_aescte_validation.py")]
_subprocess.run(command, check=True)
summary = json.loads((RESULTS / "validation_summary.json").read_text())
display(pd.Series(summary).to_frame("value"))
display(pd.read_csv(RESULTS / "week10_validation_metrics.csv"))"""),
    markdown("""## Submission

1. Explain why DSMC sampling uncertainty should be checked before ML error.
2. Derive the log-Kn weight for 0.05 and 0.5.
3. Report all primary cavity errors with their normalization scales.
4. Explain the translational-temperature overshoot.
5. Compare interpolation at Mach 1.7 with extrapolation at Mach 1.4.
6. Identify the two article claims that cannot be independently scored from the supplied data.
7. Propose one additional DSMC run that most improves the evidence, including its seed and resolution study."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
