"""Build the Week-5 student notebook from small, reviewable cells."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat as nbf
except ModuleNotFoundError:  # keep release generation dependency-free
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
            return {
                "cells": cells, "metadata": metadata,
                "nbformat": 4, "nbformat_minor": 5,
            }

    class _NotebookFormat:
        v4 = _V4()

        @staticmethod
        def write(notebook, target):
            Path(target).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")

    nbf = _NotebookFormat()


HERE = Path(__file__).resolve().parent
TARGET = HERE / "W5_Lattice_Boltzmann_Cylinder_Student.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(r"""
# Week 5 — From a steady wake to vortex shedding with D2Q9 LBM

<!-- MIE690A article-aligned validation v3 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week05/W5_Lattice_Boltzmann_Cylinder_Student.ipynb)

**Runtime:** CPU. **Quick track:** about 8–15 min. **Qualification track:** substantially longer.<br>
This is an **educational** cylinder-wake laboratory. It is deliberately separate from the cavity article and does not claim a new bifurcation result.

### Learning outcomes

By the end, you should be able to:

1. derive D2Q9 BGK/TRT collision and connect Reynolds number, viscosity, and relaxation time;
2. distinguish attached, steady-recirculating, and periodically shedding wakes;
3. extract (C_D), (C_L), recirculation length, and Strouhal number from an LBM run;
4. separate a fast demonstration from a numerically qualified result;
5. split time-resolved data by **complete Reynolds case**, never random snapshots; and
6. diagnose why a global POD surrogate diffuses advecting vortices; and
7. validate a four-frame multi-scale CNN with downstream enstrophy and spectra.
"""),
    md(r"""
## Concept map and scope

\[
\text{D2Q9 populations}
\rightarrow (\rho,u,v,p)
\rightarrow \{C_D,C_L,St,L_r\}
\rightarrow \text{Reynolds sweep}
\rightarrow \text{POD failure baseline}
\rightarrow \text{four-frame CNN}
\rightarrow \text{validation then blind-Re forecast}.
\]

For an effectively unconfined two-dimensional cylinder, the classical sequence is approximately

| Reynolds range | Expected physics | What this notebook tests |
|---:|---|---|
| (Re<6.3) | no closed recirculation bubble | centreline (u) does not contain a closed negative interval |
| (6.3\lesssim Re<47) | steady symmetric recirculating wake | perturbation decays; (C_{L,\mathrm{rms}}) remains small |
| (Re\gtrsim47) | two-dimensional Hopf instability and periodic shedding | sustained (C_L), alternating vorticity, spectral peak |
| (Re\gtrsim188) | three-dimensional Mode A becomes relevant | **outside this D2Q9 notebook** |

The precise onset is sensitive to blockage, domain length, boundary treatment, cylinder resolution, and integration time. Therefore a coarse classroom run may illustrate the sequence, but it must not be used to re-measure (Re_c\approx47).
"""),
    md(r"""
## Reproducibility contract

- `PROFILE="quick"` is for learning and debugging; its flow values are **qualitative**.
- `PROFILE="qualification"` uses a larger domain, more cylinder nodes, and a longer observation window. It is still an educational NumPy TRT code, not a substitute for a research DNS.
- The inlet perturbation is small, deterministic, and transient. It seeds the Hopf mode; it does not force shedding continuously.
- The transverse boundary is periodic. Thus the numerical problem is formally a weakly interacting cylinder array; keep (D/L_y) small and report it.
- Complete Reynolds cases remain on one side of each split. Random temporal-frame splitting is prohibited.
- The archived POD result at (Re=100) is now a **validation/failure case** because its field was inspected to design the CNN.
- CNN architecture, loss, normalization, and stopping are selected only with (Re=100); the new untouched blind case is (Re=105).
"""),
    code(r"""
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
    _flowmllab_os.chdir(_flowmllab_root / "notebooks/week05")

from pathlib import Path
import json, platform, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import display, Video
from scipy.signal import hilbert
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from flowmllab.cylinder_lbm import (
    CS2, LATTICE_VELOCITIES, LATTICE_WEIGHTS, simulate_cylinder,
)
from flowmllab import cylinder_ml

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10})
OUTPUT = Path("week05_outputs")
OUTPUT.mkdir(exist_ok=True)
print("Python:", platform.python_version())
"""),
    md(r"""
## 1. D2Q9 collision in lattice units

The populations (f_i(\mathbf{x},t)) travel along nine lattice velocities. Collision relaxes them toward the second-order isothermal equilibrium:

\[
f_i^\mathrm{eq}=w_i\rho\left[1+3\mathbf{c}_i\!\cdot\!\mathbf{u}
+\frac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u})^2-\frac{3}{2}|\mathbf{u}|^2\right],
\]

\[
f_i^*(\mathbf{x},t)=f_i-\frac{1}{\tau}(f_i-f_i^\mathrm{eq}),\qquad
f_i(\mathbf{x}+\mathbf{c}_i,t+1)=f_i^*(\mathbf{x},t).
\]

For D2Q9, (c_s^2=1/3), so

\[
\nu=c_s^2(\tau-1/2),\qquad Re=\frac{UD}{\nu},\qquad
\tau=\frac12+\frac{3UD}{Re}.
\]

Macroscopic variables follow from density and momentum moments of the populations, and the weakly compressible gauge pressure is `p = cs² (rho-rho0)`. Low lattice Mach number and viscous relaxation time safely above 0.5 are numerical requirements, not optional reporting details.

The displayed equation is the one-relaxation-time **BGK** form. The runs below use **TRT** by default: symmetric non-equilibrium populations relax with the viscous time, while antisymmetric populations use the conventional magic parameter 3/16. TRT preserves the same Navier–Stokes viscosity but is more robust for bounce-back at the small viscosities in this lesson. Set `collision_model="bgk"` only as a controlled comparison and keep its relaxation time at least 0.53.
"""),
    code(r"""
lattice = pd.DataFrame({
    "i": np.arange(9), "cx": LATTICE_VELOCITIES[:, 0],
    "cy": LATTICE_VELOCITIES[:, 1], "weight": LATTICE_WEIGHTS,
})
display(lattice)
assert np.isclose(LATTICE_WEIGHTS.sum(), 1.0)
assert np.allclose((LATTICE_WEIGHTS[:, None] * LATTICE_VELOCITIES).sum(axis=0), 0.0)
"""),
    md(r"""
### Boundary and force models

- **Cylinder:** Bouzidi interpolated bounce-back locates the analytical circular wall along every fluid–solid link; halfway bounce-back remains an optional comparison. Momentum exchange gives lift and drag.
- **Inlet:** uniform low-Mach Zou–He velocity.
- **Outlet:** first-order convective population boundary.
- **Transverse far field:** periodic.

These choices make the algorithm compact enough to inspect. They also create discretization, blockage, and outlet errors. The notebook exposes those limitations instead of hiding them.
"""),
    code(r"""
PROFILE = "quick"       # change to "qualification" only after completing the quick track
RUN_SWEEP = True

profiles = {
    "quick": dict(nx=240, ny=96, diameter=12, inflow_velocity=0.05,
                  steps=4500, history_stride=5, snapshot_start=2500,
                  snapshot_stride=125, perturbation=1e-3, seed=690,
                  collision_model="trt", cylinder_boundary="bouzidi"),
    "qualification": dict(nx=480, ny=240, diameter=24, inflow_velocity=0.05,
                           steps=16000, history_stride=5, snapshot_start=8000,
                           snapshot_stride=200, perturbation=1e-3, seed=690,
                           collision_model="trt", cylinder_boundary="bouzidi"),
}
cfg = profiles[PROFILE]
SWEEP_RE = [5, 20, 40, 60, 80, 100, 120, 180]

def numerical_card(reynolds, settings):
    tau = 0.5 + 3 * settings["inflow_velocity"] * settings["diameter"] / reynolds
    return dict(Re=reynolds, tau=tau,
                Mach=settings["inflow_velocity"] / np.sqrt(CS2),
                blockage=settings["diameter"] / settings["ny"],
                observed_time_D=settings["steps"] * settings["inflow_velocity"] / settings["diameter"])

display(pd.DataFrame([numerical_card(re, cfg) for re in SWEEP_RE]))
assert max(numerical_card(re, cfg)["Mach"] for re in SWEEP_RE) < 0.1
assert min(numerical_card(re, cfg)["tau"] for re in SWEEP_RE) > 0.505
"""),
    md(r"""
## 2. First run: (Re=100)

Predict before running:

1. Will the final vorticity remain reflection-symmetric?
2. What sign pattern should appear in (C_L(t))?
3. How many nondimensional shedding periods fit inside the retained window if (St\approx0.165)?

The solver returns direct `u`, `v`, `p`, vorticity, force histories, density history, and sparse time snapshots. No field is synthesized by ML at this stage.
"""),
    code(r"""
t0 = time.perf_counter()
case100 = simulate_cylinder(100, **cfg)
print(f"Re=100 elapsed: {time.perf_counter()-t0:.1f} s")
print(json.dumps({k: case100["metadata"][k] for k in
                  ("relaxation_time", "lattice_mach", "blockage_ratio")}, indent=2))
print("St estimate:", case100["strouhal"])
print("recirculation length/D:", case100["recirculation_length_over_diameter"])
"""),
    code(r"""
def plot_case(case, title):
    meta = case["metadata"]
    D = meta["config"]["diameter"]
    cx, cy = meta["cylinder_center"]
    xD = (case["x"] - cx) / D
    yD = (case["y"] - cy) / D
    extent = [xD.min(), xD.max(), yD.min(), yD.max()]
    tstar = case["time"] * meta["config"]["inflow_velocity"] / D

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.8), constrained_layout=True)
    levels = np.linspace(-0.22, 0.22, 45)
    im = ax[0, 0].contourf(xD, yD, case["vorticity"], levels=levels,
                           cmap="RdBu_r", extend="both")
    ax[0, 0].add_patch(Circle((0, 0), 0.5, facecolor="#F7F7F7",
                              edgecolor="black", linewidth=1.2, zorder=5))
    fig.colorbar(im, ax=ax[0, 0], label=r"$\omega_z$ (lattice units)")
    ax[0, 0].set(title="Vorticity", xlabel=r"$(x-x_c)/D$", ylabel=r"$(y-y_c)/D$")
    ax[0, 0].set_aspect("equal", adjustable="box")

    pmax = np.nanpercentile(np.abs(case["p"]), 99)
    im = ax[0, 1].imshow(case["p"], origin="lower", extent=extent,
                         cmap="coolwarm", vmin=-pmax, vmax=pmax, aspect="auto")
    fig.colorbar(im, ax=ax[0, 1], label=r"$p-p_0$")
    ax[0, 1].set(title="Gauge pressure", xlabel=r"$(x-x_c)/D$", ylabel=r"$(y-y_c)/D$")
    ax[0, 1].set_aspect("equal", adjustable="box")

    ax[1, 0].plot(tstar, case["lift_coefficient"], lw=1.3, label=r"$C_L$")
    ax[1, 0].axhline(0, color="0.35", lw=0.8)
    ax[1, 0].set(xlabel=r"$tU/D$", ylabel=r"$C_L$", title="Lift history")

    ax[1, 1].plot(tstar, case["drag_coefficient"], color="#C44E52", lw=1.3,
                  label=r"$C_D$")
    ax[1, 1].set(xlabel=r"$tU/D$", ylabel=r"$C_D$", title="Drag history")
    fig.suptitle(title, fontsize=15)
    return fig

fig = plot_case(case100, f"D2Q9 cylinder, Re=100 ({PROFILE} profile)")
fig.savefig(OUTPUT / "re100_fields_and_forces.png", dpi=180)
plt.show()
"""),
    md(r"""
## 3. Reynolds sweep: classify physics before fitting ML

Each case is evaluated using three kinds of evidence:

1. **field:** closed reverse-flow region and alternating vorticity;
2. **signal:** post-transient (C_{L,\mathrm{rms}}) and a sustained oscillation;
3. **frequency:** a resolvable dominant (St=fD/U).

An FFT peak by itself is not evidence of vortex shedding: a decaying transient can also have a peak. Near onset, fit the envelope growth/decay rate and lengthen the run.
"""),
    code(r"""
cases = {100: case100}
if RUN_SWEEP:
    for re in SWEEP_RE:
        if re in cases:
            continue
        t0 = time.perf_counter()
        cases[re] = simulate_cylinder(re, **cfg)
        print(f"Re={re:3d}: {time.perf_counter()-t0:6.1f} s, St={cases[re]['strouhal']:.4g}")

def post_transient_rms(values, fraction=0.5):
    values = np.asarray(values)
    tail = values[int(fraction * len(values)):]
    return float(np.sqrt(np.mean((tail - tail.mean())**2)))

rows = []
for re in sorted(cases):
    case = cases[re]
    rho = case["mean_density_ratio"]
    rows.append({
        "Re": re,
        "tau": case["metadata"]["relaxation_time"],
        "blockage": case["metadata"]["blockage_ratio"],
        "density_drift": float(np.max(np.abs(rho - 1))),
        "Cd_mean_tail": float(np.mean(case["drag_coefficient"][len(case["time"])//2:])),
        "Cl_rms_tail": post_transient_rms(case["lift_coefficient"]),
        "St": case["strouhal"],
        "Lr_over_D": case["recirculation_length_over_diameter"],
    })
sweep = pd.DataFrame(rows).sort_values("Re")
display(sweep)
sweep.to_csv(OUTPUT / "cylinder_sweep_metrics.csv", index=False)
"""),
    code(r"""
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for re in sorted(cases):
    case = cases[re]
    D = case["metadata"]["config"]["diameter"]
    U = case["metadata"]["config"]["inflow_velocity"]
    tstar = case["time"] * U / D
    ax[0, 0].plot(tstar, case["lift_coefficient"], lw=1.0, label=f"Re={re}")
ax[0, 0].set(xlabel=r"$tU/D$", ylabel=r"$C_L$", title="Lift: decay or sustained oscillation?")
ax[0, 0].legend(ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

ax[0, 1].semilogy(sweep.Re, np.maximum(sweep.Cl_rms_tail, 1e-10), "o-", color="#4C72B0")
ax[0, 1].axvline(47, color="0.35", ls="--", lw=1, label="classical onset ≈47")
ax[0, 1].set(xlabel="Re", ylabel=r"tail $C_{L,\mathrm{rms}}$", title="Wake unsteadiness")
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

valid_st = np.isfinite(sweep.St) & (sweep.Re >= 60)
ax[1, 0].plot(sweep.loc[valid_st, "Re"], sweep.loc[valid_st, "St"], "o-", label="LBM")
ax[1, 0].axhspan(0.15, 0.19, color="#55A868", alpha=0.18, label="broad 2-D reference band")
ax[1, 0].set(xlabel="Re", ylabel="St", title="Dominant shedding frequency")
ax[1, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

ax[1, 1].plot(sweep.Re, sweep.Cd_mean_tail, "o-", color="#C44E52", label=r"mean $C_D$")
ax[1, 1].set(xlabel="Re", ylabel=r"$\overline C_D$", title="Mean drag diagnostic")
plt.show()
"""),
    md(r"""
### Validation ladder

Use a ladder rather than a single reassuring plot.

| Gate | Classroom acceptance | Why it matters |
|---|---:|---|
| lattice Mach | (Ma<0.10) | limits weak-compressibility error |
| viscous relaxation | tau > 0.505, preferably farther from 0.5 | stability and viscosity resolution |
| blockage | (D/L_y\le0.15) | limits periodic-image influence |
| density drift | (\max|\bar\rho/\rho_0-1|<0.03) | mass/boundary diagnostic |
| subcritical wake | (Re=40): no sustained tail oscillation | avoids false shedding |
| supercritical wake | (Re=100): sustained (C_L) and alternating vorticity | correct qualitative regime |
| literature sanity check | (Re=100): (St\approx0.164{-}0.166), (overline C_D\approx1.32{-}1.34) in well-resolved 2-D references | quantitative target, **qualification profile only** |

The last row is not a gate for the quick profile. If it fails, report the deviation and test grid, blockage, outlet distance, and observation time; do not tune a correction factor.
"""),
    code(r"""
validation = []
for row in rows:
    validation.append({
        "Re": row["Re"],
        "Mach_pass": cfg["inflow_velocity"] / np.sqrt(CS2) < 0.10,
        "tau_pass": row["tau"] > 0.505,
        "blockage_pass": row["blockage"] <= 0.15,
        "density_pass": row["density_drift"] < 0.03,
    })
validation = pd.DataFrame(validation)
display(validation)
assert validation[["Mach_pass", "tau_pass", "blockage_pass", "density_pass"]].all().all()

if PROFILE == "qualification":
    row100 = sweep[sweep.Re == 100].iloc[0]
    print("Re=100 deviations from reference centres:",
          {"St": float(row100.St - 0.165), "Cd": float(row100.Cd_mean_tail - 1.33)})
else:
    print("Quick profile: quantitative literature agreement is reported, not enforced.")
"""),
    md(r"""
## 4. Optional numerical-sensitivity checkpoint

Change one numerical choice at a time. A convincing result uses at least three cylinder resolutions and checks domain/blockage separately. The cell below performs only a modest classroom comparison; the qualification report should add (D=24,36,48) with matched (Ma), domain measured in (D), and observation time measured in (D/U).
"""),
    code(r"""
RUN_REFINEMENT = False
if RUN_REFINEMENT:
    checks = []
    for D, ny, nx in [(10, 80, 220), (14, 112, 308), (18, 144, 396)]:
        check = simulate_cylinder(
            100, nx=nx, ny=ny, diameter=D, inflow_velocity=0.05,
            steps=int(4500 * D / 12), history_stride=5,
            perturbation=1e-3, seed=690, collision_model="trt",
        )
        tail = slice(len(check["time"]) // 2, None)
        checks.append({"D": D, "ny": ny, "nx": nx,
                       "St": check["strouhal"],
                       "Cd_mean": check["drag_coefficient"][tail].mean()})
    refinement = pd.DataFrame(checks)
    display(refinement)
"""),
    md(r"""
## 5. POD baseline: a useful failure, not the final model

Lee & You (JFM, 2019) motivate the educational question: can a data-driven model reconstruct unsteady cylinder-wake fields across Reynolds number? We borrow that **problem framing**, not their architecture or a claim of reproducing their results.

The original small exercise used phase-conditioned POD coefficients:

\[
(Re,\sin\phi,\cos\phi)\longmapsto (a_1,\ldots,a_r)
\longmapsto (u,v,p).
\]

We compare:

- a non-neural ridge/harmonic POD baseline; and
- a fixed two-layer `tanh` MLP branch with the same training snapshots and POD trunk.

It captured the near wake but visibly diffused translating downstream vortices. Its retained (Re=100) result is therefore a **failure baseline**. Because that result was inspected and used to redesign the model, (Re=100) is no longer called blind. Random snapshot splitting would still leak nearly identical neighbouring phases and remains prohibited.
"""),
    code(r"""
ML_RE = [60, 80, 100, 120]
BLIND_RE = [100]
missing = [re for re in ML_RE if re not in cases or "snapshots" not in cases[re]]
if missing:
    raise RuntimeError(f"Run the sweep with snapshots before ML; missing {missing}")

def snapshot_phase(case):
    signal = np.asarray(case["lift_coefficient"])
    analytic = hilbert(signal - signal.mean())
    phase_history = np.unwrap(np.angle(analytic))
    phase = np.interp(case["snapshot_time"], case["time"], phase_history)
    return np.mod(phase, 2*np.pi)

fields = {name: [] for name in ("u", "v", "p")}
re_labels, phase_labels = [], []
for re in ML_RE:
    case = cases[re]
    n = len(case["snapshot_time"])
    for name in fields:
        scale = cfg["inflow_velocity"] if name in ("u", "v") else cfg["inflow_velocity"]**2
        fields[name].append(case["snapshots"][name] / scale)
    re_labels.extend([re] * n)
    phase_labels.extend(snapshot_phase(case))

fields = {name: np.concatenate(parts) for name, parts in fields.items()}
re_labels = np.asarray(re_labels, dtype=float)
phase_labels = np.asarray(phase_labels, dtype=float)
split = cylinder_ml.casewise_reynolds_split(re_labels, test_reynolds=BLIND_RE)
print("development Re:", split.train_reynolds)
print("blind Re:", split.test_reynolds)
assert not np.intersect1d(split.train_reynolds, split.test_reynolds).size
"""),
    md(r"""
### Development-only POD and non-neural baseline

The POD basis is fit only on development Reynolds cases. Rank is fixed at 8 before the blind test; it is not selected by looking at (Re=100). The harmonic ridge baseline supplies a serious non-neural comparison rather than comparing the MLP only with a constant mean field.
"""),
    code(r"""
train_fields = {name: values[split.train_indices] for name, values in fields.items()}
test_fields = {name: values[split.test_indices] for name, values in fields.items()}

RANK = min(8, len(split.train_indices) - 1)
baseline = cylinder_ml.fit_pod_regressor(
    train_fields, re_labels[split.train_indices], phase_labels[split.train_indices],
    field_names=("u", "v", "p"), rank=RANK,
    reynolds_degree=2, phase_harmonics=2, ridge=1e-8,
)
baseline_prediction = baseline.predict_fields(
    re_labels[split.test_indices], phase_labels[split.test_indices]
)
baseline_metrics = cylinder_ml.reconstruction_diagnostics(test_fields, baseline_prediction)
print("POD energy at fixed rank:", baseline.pod.cumulative_energy[RANK-1])
display(pd.DataFrame([{"model": "harmonic-ridge POD", **baseline_metrics}]))
"""),
    md(r"""
### Fixed neural branch

Only the POD coefficients are learned. The spatial trunk is exactly the same development-only basis used by the baseline. Inputs and coefficient targets are standardized using development data only. The architecture and seed are frozen before opening the blind case.
"""),
    code(r"""
train_vectors, _ = cylinder_ml.pack_fields(train_fields, ("u", "v", "p"))
train_coeff = cylinder_ml.project_pod(baseline.pod, train_vectors)

def branch_features(reynolds, phase):
    reynolds = np.asarray(reynolds)
    phase = np.asarray(phase)
    return np.column_stack((reynolds, np.sin(phase), np.cos(phase)))

x_scaler = StandardScaler().fit(branch_features(
    re_labels[split.train_indices], phase_labels[split.train_indices]))
y_scaler = StandardScaler().fit(train_coeff)
mlp = MLPRegressor(
    hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
    alpha=1e-5, max_iter=3000, random_state=690,
)
mlp.fit(
    x_scaler.transform(branch_features(re_labels[split.train_indices], phase_labels[split.train_indices])),
    y_scaler.transform(train_coeff),
)
blind_coeff = y_scaler.inverse_transform(mlp.predict(
    x_scaler.transform(branch_features(re_labels[split.test_indices], phase_labels[split.test_indices]))
))
blind_vectors = cylinder_ml.reconstruct_pod(baseline.pod, blind_coeff)
mlp_prediction = cylinder_ml.unpack_fields(blind_vectors, baseline.layout)
"""),
    md(r"""
# Diagnostic gate: explain the POD failure

Before running the next cell, record:

- development Reynolds cases and the untouched Reynolds case;
- POD rank, MLP layers, regularization, optimizer, maximum iterations, and seed;
- the non-neural baseline specification;
- the primary metric: combined relative (L_2) over all blind snapshots; and
- the physical checks: divergence, solid speed, pressure gauge, and qualitative wake structure.

This archived exercise deliberately shows why low global field error and high cumulative POD energy do not certify vorticity fidelity. Do not tune this model and continue calling (Re=100) blind.
"""),
    code(r"""
mlp_metrics = cylinder_ml.reconstruction_diagnostics(test_fields, mlp_prediction)
comparison = pd.DataFrame([
    {"model": "harmonic-ridge POD", **baseline_metrics},
    {"model": "two-layer MLP POD", **mlp_metrics},
])
display(comparison)

mask = cases[BLIND_RE[0]]["solid"]
physics = []
for name, prediction in [("harmonic-ridge POD", baseline_prediction),
                         ("two-layer MLP POD", mlp_prediction)]:
    diag = cylinder_ml.flow_diagnostics(
        {"u": prediction["u"], "v": prediction["v"]},
        dx=1.0, dy=1.0, solid_mask=mask,
    )
    physics.append({"model": name, "divergence_rms": diag["divergence_rms"],
                    "solid_speed_rms": diag["solid_speed_rms"],
                    "mean_pressure_gauge": float(prediction["p"].mean())})
display(pd.DataFrame(physics))
comparison.to_csv(OUTPUT / "blind_re100_ml_metrics.csv", index=False)
"""),
    code(r"""
# Show one blind phase without placing legends over the fields.
j = len(split.test_indices) // 2
truth = test_fields["v"][j]
pred_b = baseline_prediction["v"][j]
pred_n = mlp_prediction["v"][j]
limit = np.percentile(np.abs(truth), 99)
err_limit = max(np.percentile(np.abs(pred_n-truth), 99), 1e-8)

fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for axis, field, title in zip(ax.flat[:3], [truth, pred_b, pred_n],
                              ["LBM reference", "harmonic-ridge POD", "two-layer MLP POD"]):
    im = axis.imshow(field, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_title(title)
    axis.set(xlabel="x lattice node", ylabel="y lattice node")
    axis.set_aspect("equal", adjustable="box")
    axis.add_patch(Circle(cases[BLIND_RE[0]]["metadata"]["cylinder_center"],
                          cfg["diameter"] / 2, facecolor="#F7F7F7",
                          edgecolor="black", linewidth=1.2, zorder=5))
fig.colorbar(im, ax=ax.flat[:3].tolist(), label=r"$v/U$")
imerr = ax[1, 1].imshow(pred_n-truth, origin="lower", cmap="coolwarm",
                        vmin=-err_limit, vmax=err_limit, aspect="auto")
ax[1, 1].set(title="MLP error", xlabel="x lattice node", ylabel="y lattice node")
fig.colorbar(imerr, ax=ax[1, 1], label=r"error in $v/U$")
fig.suptitle("Complete-Re blind field: Re=100")
fig.savefig(OUTPUT / "blind_re100_field_comparison.png", dpi=180)
plt.show()
"""),
    md(r"""
### Archived high-resolution POD failure animation

The release also contains an executed 1920x1080 vortex-shedding comparison.
It uses a longer saturated periodic window than the quick live cells above.
The complete `Re=100` trajectory was withheld from that fit; the video model was trained on
`Re=60,80,90,110,120,140`.  Its left, middle, and right panels show the blind
LBM vorticity, neural POD prediction, and signed error.  The lift-history
marker identifies the displayed LBM snapshot.

[Open or download the retained MP4](../../results/cylinder_ml/blind_re100_lbm_vs_neural.mp4).

The visual downstream diffusion is retained rather than hidden. This is a
phase-conditioned interpolation: phase is extracted from the LBM lift signal.
It is not an autonomous time rollout, and the quick LBM target is not described
as grid-converged DNS.
"""),
    code(r"""
video_path = Path("../../results/cylinder_ml/blind_re100_lbm_vs_neural.mp4")
if video_path.is_file():
    display(Video(str(video_path), embed=False, html_attributes="controls loop"))
else:
    print("Retained video not found. Run qa/run_cylinder_blind_video.py from the repository root.")
"""),
    md(r"""
## 6. Four-frame multi-scale CNN: the corrected model

Following the predictive structure of Lee & You (JFM, 2019), the corrected
model receives four consecutive fields and predicts the next field:

\[
\{u,v,p\}_{n-3:n},\ Re,\ \chi_f
\longmapsto \{u,v,p\}_{n+1}.
\]

It uses fine, half-resolution, and quarter-resolution convolution branches and
learns an increment from the latest frame. The last convolution is initialized
to zero, so training starts from the matched persistence baseline. The fluid
mask enforces exact no-slip velocity inside the analytical cylinder.

The dense snapshot interval is 25 lattice steps,

\[
\Delta t^*=25\,U_\infty/D=0.1042,
\]

rather than the old (0.521). Development cases are
`60,80,90,110,120,140`; the complete `Re=100` trajectory selects stopping;
the complete `Re=105` trajectory is opened only after every validation gate
passes.

The composite objective is

\[
\mathcal L=\mathcal L_{field}
+0.20\mathcal L_{gradient}
+0.20\mathcal L_{\omega}
+0.05\mathcal L_{\nabla\cdot u}.
\]

This is a teacher-forced one-step forecast from four true previous LBM frames.
It is not an autonomous rollout.
"""),
    code(r"""
cnn_metrics_path = Path("../../results/cylinder_cnn/multiscale_cnn_metrics.json")
cnn_metrics = json.loads(cnn_metrics_path.read_text())
assert cnn_metrics["validation_pass"]

rows = []
for split_name in ("validation", "blind"):
    split_metrics = cnn_metrics[split_name]
    for model_name, values in split_metrics["models"].items():
        rows.append({
            "split": split_name,
            "Re": split_metrics["reynolds"],
            "model": "multi-scale CNN" if model_name == "prediction" else model_name,
            "vorticity relative L2": values["vorticity_relative_l2"],
            "mean station profile L2": values["mean_station_profile_relative_l2"],
            "mean enstrophy error": values["mean_station_enstrophy_relative_error"],
            "mean normalized PSD L2": values["mean_station_psd_relative_l2"],
        })
display(pd.DataFrame(rows))
display(cnn_metrics["validation_gates"])
"""),
    code(r"""
from IPython.display import Image
display(Image(filename="../../results/cylinder_cnn/re105_blind_downstream.png"))
display(Video("../../results/cylinder_cnn/re105_lbm_vs_multiscale_cnn.mp4",
              embed=False, html_attributes="controls loop"))
"""),
    md(r"""
### Reproduce the frozen experiment

Install the optional ML dependencies and run the development/validation stage.
The blind flag refuses to open (Re=105) unless every (Re=100) gate passes.

```bash
python -m pip install -e '.[ml]'
python qa/run_cylinder_multiscale_cnn.py --workers 4
python qa/run_cylinder_multiscale_cnn.py --reuse-weights --run-blind
```

Acceptance is spatial, not merely global: compare vorticity profiles,
enstrophy amplitude, and transverse PSD at
((x-x_c)/D=2,4,6,8), plus exact no-slip and the matched persistence baseline.
"""),
    md(r"""
## Interpretation: what this exercise does and does not show

- The POD result shows how a global low-rank representation can smear translated structures despite attractive aggregate errors.
- The CNN result shows one-step field forecasting across a complete unseen Reynolds case using four previous fields.
- It does **not** establish a general neural operator for arbitrary inlet functions or geometries.
- It does **not** identify the Hopf point: training is restricted to already unsteady cases and phase is supplied.
- A low field error does not guarantee correct force, frequency, or long-time phase. Those require direct tests.
- The CNN is useful only if it beats persistence and retains downstream amplitude, profile, spectrum, and physical diagnostics.

For a research extension, use longer phase-aligned trajectories, train/validation/test Reynolds bands, multiple seeds, force consistency, 20–50-cycle phase drift, and independent high-fidelity DNS. None of those claims belongs in this classroom release unless executed.
"""),
    md(r"""
## Required deliverables

1. One D2Q9 derivation page with your computed viscosity, relaxation time, Mach number, and blockage for every case.
2. A four-panel (Re=100) field/force figure with readable axes and unobstructed legends.
3. The complete sweep table containing density drift, (overline C_D), (C_{L,\mathrm{rms}}), (St), and (L_r/D).
4. Evidence for attached/no-bubble, steady-recirculating, and periodic-shedding regimes; explain any mismatch.
5. One resolution or blockage comparison at (Re=100).
6. The archived POD failure, including why (Re=100) is now validation rather than blind.
7. The frozen CNN specification, persistence comparison, validation gates, and untouched (Re=105) result.
8. Enstrophy/profile/PSD evidence at (2D,4D,6D,8D) and a paragraph distinguishing one-step forecasting from rollout.
9. A short paragraph distinguishing an educational demonstration from quantitative DNS validation.

### Exercises

1. Set the initial perturbation to zero. How long does round-off take to seed shedding at (Re=100)?
2. At (Re=40,46,48,60), fit the logarithm of the (C_L) envelope. Which cases grow and which decay?
3. Double the transverse size at fixed (D). Quantify changes in (St), drag, and recirculation length.
4. Replace random snapshot splitting with complete-Re splitting and compare the two reported errors. Explain the leakage.
5. Remove phase from both models. What failure is visible even if the mean field remains plausible?
6. Hold out an edge case rather than an interpolated case. Why is extrapolation harder?
"""),
    md(r"""
## References

- S. Succi (2001), *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*, Oxford University Press.
- Q. Zou & X. He (1997), “On pressure and velocity boundary conditions for the lattice Boltzmann BGK model,” *Physics of Fluids* 9, 1591–1598.
- C. H. K. Williamson (1996), “Vortex dynamics in the cylinder wake,” *Annual Review of Fluid Mechanics* 28, 477–539. [doi:10.1146/annurev.fl.28.010196.002401](https://doi.org/10.1146/annurev.fl.28.010196.002401)
- C. P. Jackson (1987), “A finite-element study of the onset of vortex shedding in flow past variously shaped bodies,” *Journal of Fluid Mechanics* 182, 23–45.
- D. Barkley & R. D. Henderson (1996), “Three-dimensional Floquet stability analysis of the wake of a circular cylinder,” *Journal of Fluid Mechanics* 322, 215–241.
- S. Lee & D. You (2019), “Data-driven prediction of unsteady flow fields over a circular cylinder using deep learning,” *Journal of Fluid Mechanics* 879, 217–254. [doi:10.1017/jfm.2019.700](https://doi.org/10.1017/jfm.2019.700)

The numerical reference values in the validation ladder are targets for well-resolved two-dimensional calculations, not guarantees for the quick classroom profile.
"""),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "course": {
            "title": "MIE 690A AI in Fluid Mechanics",
            "lab": "Week 5 LBM cylinder wake",
            "protocol_version": "2026.09",
        },
    },
)
nbf.write(notebook, TARGET)
print(TARGET)
