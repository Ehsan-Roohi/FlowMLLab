"""Build the Week-7 student notebook from small, reviewable cells."""

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
TARGET = HERE / "W7_Lattice_Boltzmann_Cylinder_Student.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(r"""
# Week 7 — From a steady wake to vortex shedding with D2Q9 LBM

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week07/W7_Lattice_Boltzmann_Cylinder_Student.ipynb)

**Runtime:** CPU. **Retained-evidence track:** under 2 min, including an optional checkpoint continuation. **Full regeneration:** substantially longer.<br>
This is an **educational** cylinder-wake laboratory. It is deliberately separate from the cavity article and does not claim a new bifurcation result.

### Learning outcomes

By the end, you should be able to:

1. derive D2Q9 BGK/TRT collision and connect Reynolds number, viscosity, and relaxation time;
2. distinguish attached, steady-recirculating, and periodically shedding wakes;
3. extract $C_D$, $C_L$, recirculation length, and Strouhal number from an LBM run;
4. separate code verification, statistical convergence, grid convergence,
   domain sensitivity, and external validation;
5. diagnose a failed three-grid asymptotic/GCI check and specify the next
   refinement without deleting a result or relaxing a gate;
6. split time-resolved data by **complete Reynolds case**, never random snapshots;
7. diagnose why a global POD surrogate diffuses advecting vortices;
8. validate a four-frame multi-scale CNN with downstream enstrophy and spectra,
   while separating low-error one-step prediction from recursive rollout; and
9. construct and audit a phase-stable learned decoder for autonomous long-horizon prediction.
"""),
    md(r"""
## Concept map and scope

\[
\text{D2Q9 populations}
\rightarrow (\rho,u,v,p)
\rightarrow \{C_D,C_L,St,L_r\}
\rightarrow \text{Reynolds sweep}
\rightarrow \text{three-grid verification}
\rightarrow \text{POD failure baseline}
\rightarrow \text{four-frame CNN audit}
\rightarrow \text{phase-stable decoder}
\rightarrow \text{validation then fresh test}.
\]

For an effectively unconfined two-dimensional cylinder, the classical sequence is approximately

| Reynolds range | Expected physics | What this notebook tests |
|---:|---|---|
| $Re<6.3$ | no closed recirculation bubble | centreline $u$ does not contain a closed negative interval |
| $6.3\lesssim Re<47$ | steady symmetric recirculating wake | perturbation decays; $C_{L,\mathrm{rms}}$ remains small |
| $Re\gtrsim47$ | two-dimensional Hopf instability and periodic shedding | sustained $C_L$, alternating vorticity, spectral peak |
| $Re\gtrsim188$ | three-dimensional Mode A becomes relevant | **outside this D2Q9 notebook** |

The precise onset is sensitive to blockage, domain length, boundary treatment, cylinder resolution, and integration time. Therefore a coarse classroom run may illustrate the sequence, but it must not be used to re-measure $Re_c\approx47$.
"""),
    md(r"""
## Reproducibility contract

- `PROFILE="quick"` is for learning and debugging; its flow values are **qualitative**.
- `PROFILE="validation"` uses a larger domain, more cylinder nodes, and a longer observation window. It is still an educational NumPy TRT code, not a substitute for a research DNS.
- The localized wake perturbation is small, deterministic, and transient. It seeds the Hopf mode; it is not an inlet disturbance and does not force shedding continuously.
- The transverse boundary is periodic. Thus the numerical problem is formally a cylinder array of pitch $L_y$; keep $D/L_y$ small and report it.
- Complete Reynolds cases remain on one side of each split. Random temporal-frame splitting is prohibited.
- The archived POD result at $Re=100$ is now a **validation/failure case** because its field was inspected to design the CNN.
- CNN architecture, loss, normalization, and stopping were selected only with $Re=100$; $Re=105$ is the retained withheld interpolation test for that experiment.
- For the long-horizon decoder, `Re=90,110,120,140` are development, `Re=100` is validation, `Re=95` is a fresh test opened once after harmonic-order selection, and `Re=105` is retained historical evidence.
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
    _flowmllab_os.chdir(_flowmllab_root / "notebooks/week07")

from pathlib import Path
import json, platform, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from IPython.display import HTML, Image, Video, display

from flowmllab.cylinder_lbm import (
    CS2, LATTICE_VELOCITIES, LATTICE_WEIGHTS, simulate_cylinder,
    grid_convergence_diagnostics, strouhal_diagnostics,
)

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10})
OUTPUT = Path("week07_outputs")
OUTPUT.mkdir(exist_ok=True)
REPO_ROOT = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "results/cylinder_lbm").is_dir()
)
print("Python:", platform.python_version())
"""),
    md(r"""
## 1. D2Q9 collision in lattice units

The populations $f_i(\mathbf{x},t)$ travel along nine lattice velocities. Collision relaxes them toward the second-order isothermal equilibrium:

\[
f_i^\mathrm{eq}=w_i\rho\left[1+3\mathbf{c}_i\!\cdot\!\mathbf{u}
+\frac{9}{2}(\mathbf{c}_i\!\cdot\!\mathbf{u})^2-\frac{3}{2}|\mathbf{u}|^2\right],
\]

\[
f_i^*(\mathbf{x},t)=f_i-\frac{1}{\tau}(f_i-f_i^\mathrm{eq}),\qquad
f_i(\mathbf{x}+\mathbf{c}_i,t+1)=f_i^*(\mathbf{x},t).
\]

For D2Q9, $c_s^2=1/3$, so

\[
\nu=c_s^2(\tau-1/2),\qquad Re=\frac{UD}{\nu},\qquad
\tau=\frac12+\frac{3UD}{Re}.
\]

Macroscopic variables follow from density and momentum moments of the populations, and the weakly compressible gauge pressure is `p = cs² (rho-rho0)`. Low lattice Mach number and viscous relaxation time safely above 0.5 are numerical requirements, not optional reporting details.

The displayed equation is the one-relaxation-time **BGK** form. The runs below use **TRT** by default: symmetric non-equilibrium populations relax with the viscous time, while antisymmetric populations use the conventional magic parameter 3/16. TRT preserves the same Navier–Stokes viscosity but is more robust for bounce-back at the small viscosities in this lesson. Set `collision_model="bgk"` only as a controlled comparison and keep its relaxation time at least 0.53.
"""),
    md(r"""
### LBM algorithm: one time step in seven operations

| Order | Operation | What happens |
|---:|---|---|
| 1 | moments | recover $\rho=\sum_i f_i$ and $\rho\mathbf{u}=\sum_i f_i\mathbf{c}_i$ |
| 2 | equilibrium | compute $f_i^{eq}(\rho,\mathbf{u})$ |
| 3 | collision | relax non-equilibrium populations with BGK or TRT |
| 4 | streaming | move $f_i^*$ to the neighbour in direction $\mathbf{c}_i$ |
| 5 | cylinder wall | apply Bouzidi reflection on analytical-circle links and accumulate momentum exchange |
| 6 | outer boundaries | impose the Zou–He inlet and update the convective outlet |
| 7 | diagnostics | record mass, $C_D$, $C_L$, fields, and restart state |

Initialize with equilibrium populations and a half-cosine acceleration, then repeat these seven operations. LBM advances the nine populations—not $u$, $v$, and $p$ directly. Therefore a valid restart stores all nine $f_i$ populations and the outlet memory.
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
- **Startup/inlet:** mass-preserving half-cosine far-field acceleration followed by uniform low-Mach Zou–He velocity.
- **Outlet:** first-order convective population boundary.
- **Transverse far field:** periodic.

These choices make the algorithm compact enough to inspect. They also create discretization, blockage, and outlet errors. A longer startup ramp reduces the initial pulse but does not absorb the transverse standing wave supported by periodic boundaries: ramp lengths 0, 300, and 1500 steps retain about 0.015 RMS high-frequency lift jitter with an 83-step period, consistent with $L_y/(2c_s)$. An explicitly labelled 83-step moving average may be used for force presentation because the shedding period is about 1300 steps, but raw force remains the audit signal. A sponge or changed far-field boundary—not a longer ramp—is the numerical remedy to test.
"""),
    code(r"""
PROFILE = "retained_quick"
RUN_LIVE_CONTINUATION = True
EVIDENCE = REPO_ROOT / "results/cylinder_lbm"
SWEEP_RE = [5, 20, 40, 100, 180]

def load_retained_case(reynolds):
    with np.load(EVIDENCE / f"re{reynolds}_teaching_case.npz", allow_pickle=False) as archive:
        case = {key: archive[key] for key in archive.files}
    case["solid"] = case["solid"].astype(bool)
    case["metadata"] = json.loads(str(case["metadata"]))
    if "strouhal_diagnostics" in case:
        case["strouhal_diagnostics"] = json.loads(str(case["strouhal_diagnostics"]))
    return case

cases = {re: load_retained_case(re) for re in SWEEP_RE}
sweep = pd.read_csv(EVIDENCE / "regime_metrics.csv")
cfg = cases[100]["metadata"]["config"]

display(sweep[["Re", "tau", "Mach", "blockage", "observed_time_D_over_U"]])
assert float(sweep["Mach"].max()) < 0.1
assert float(sweep["blockage"].max()) <= 0.15
"""),
    md(r"""
## 2. First run: $Re=100$

Predict before running:

1. Will the final vorticity remain reflection-symmetric?
2. What sign pattern should appear in $C_L(t)$?
3. How many nondimensional shedding periods fit inside the retained window if $St\approx0.165$?

The solver returns direct `u`, `v`, `p`, vorticity, force histories, density history, and sparse time snapshots. No field is synthesized by ML at this stage.
"""),
    code(r"""
case100 = cases[100]
row100 = sweep.loc[sweep.Re == 100].iloc[0]
print(json.dumps({k: case100["metadata"][k] for k in
                  ("relaxation_time", "lattice_mach", "blockage_ratio")}, indent=2))
print("quality-gated St:", row100.St, "| reason:", row100.St_reason,
      "| retained cycles:", row100.St_cycles)
print("time-mean recirculation length/D:", row100.Lr_over_D)

if RUN_LIVE_CONTINUATION:
    required = {"restart_populations", "restart_outlet_previous", "restart_completed_steps"}
    if required.issubset(case100):
        restart = {
            "populations": case100["restart_populations"],
            "outlet_previous": case100["restart_outlet_previous"],
            "completed_steps": int(case100["restart_completed_steps"]),
        }
        live = simulate_cylinder(
            100, nx=cfg["nx"], ny=cfg["ny"], diameter=cfg["diameter"],
            center=tuple(case100["metadata"]["cylinder_center"]),
            inflow_velocity=cfg["inflow_velocity"], steps=1200, history_stride=4,
            statistics_start=0, perturbation=0.0, collision_model="trt",
            cylinder_boundary="bouzidi", restart_state=restart,
        )
        for name in ("rho", "u", "v", "p", "vorticity"):
            case100[name] = live[name]
        print("Continued the saturated checkpoint by 1200 lattice steps.")
    else:
        print("Checkpoint not present; displaying the retained saturated field.")
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
2. **signal:** post-transient $C_{L,\mathrm{rms}}$ and a sustained oscillation;
3. **frequency:** a resolvable dominant $St=fD/U$.

An FFT peak by itself is not evidence of vortex shedding: a decaying transient can also have a peak. Near onset, fit the envelope growth/decay rate and lengthen the run.
"""),
    code(r"""
def post_transient_rms(values, fraction=0.5):
    values = np.asarray(values)
    tail = values[int(fraction * len(values)):]
    return float(np.sqrt(np.mean((tail - tail.mean())**2)))

display(sweep[[
    "Re", "tau", "blockage", "observed_time_D_over_U", "density_drift",
    "Cd_mean", "Cl_rms", "St", "St_valid", "St_reason", "St_cycles",
    "lift_relative_rms_change", "Lr_over_D", "regime_pass",
]])
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

ax[0, 1].semilogy(sweep.Re, np.maximum(sweep.Cl_rms, 1e-10), "o-", color="#4C72B0")
ax[0, 1].axvline(47, color="0.35", ls="--", lw=1, label="classical onset ≈47")
ax[0, 1].set(xlabel="Re", ylabel=r"tail $C_{L,\mathrm{rms}}$", title="Wake unsteadiness")
ax[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

valid_st = sweep.St_valid.astype(bool)
ax[1, 0].plot(sweep.loc[valid_st, "Re"], sweep.loc[valid_st, "St"], "o-", label="LBM")
ax[1, 0].axhspan(0.15, 0.19, color="#55A868", alpha=0.18, label="broad 2-D reference band")
ax[1, 0].set(xlabel="Re", ylabel="St", title="Dominant shedding frequency")
ax[1, 0].legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

ax[1, 1].plot(sweep.Re, sweep.Cd_mean, "o-", color="#C44E52", label=r"mean $C_D$")
ax[1, 1].set(xlabel="Re", ylabel=r"$\overline C_D$", title="Mean drag diagnostic")
plt.show()
"""),
    md(r"""
### Validation ladder

Use a ladder rather than a single reassuring plot.

| Gate | Classroom acceptance | Why it matters |
|---|---:|---|
| lattice Mach | $Ma<0.10$ | limits weak-compressibility error |
| viscous relaxation | $\tau>0.505$, preferably farther from 0.5 | stability and viscosity resolution |
| blockage | $D/L_y\le0.15$ | limits periodic-image influence |
| density drift | $\max|\bar\rho/\rho_0-1|<0.03$ | mass/boundary diagnostic |
| subcritical wake | $Re=40$: no sustained tail oscillation | avoids false shedding |
| supercritical wake | $Re=100$: sustained $C_L$ and alternating vorticity | correct qualitative regime |
| frequency quality | $0.05<St<0.5$, at least 8 cycles, stationary amplitude | rejects acoustic/startup peaks |
| literature sanity check | $Re=100$: retained comparison bands $0.158\le St\le0.171$ and $1.27\le\overline C_D\le1.38$; central well-resolved values are often near $St=0.164{-}0.166$ and $\overline C_D=1.32{-}1.34$ | quantitative target, **validation profile only** |

The last row is not a gate for the quick profile. If it fails, report the deviation and test grid, blockage, outlet distance, and observation time; do not tune a correction factor.
"""),
    code(r"""
validation = sweep.assign(
    Mach_pass=sweep.Mach < 0.10,
    tau_pass=sweep.tau > 0.505,
    blockage_pass=sweep.blockage <= 0.15,
    density_pass=sweep.density_drift < 0.03,
)
display(validation)
assert validation[["Mach_pass", "tau_pass", "blockage_pass", "density_pass"]].all().all()

print("Retained quick profile: quantitative literature agreement is reported, not enforced.")
"""),
    md(r"""
### CFD-to-ML firewall: qualify the labels first

The CFD workflow is deliberately executable without importing, training, or evaluating a learned model:

1. Freeze the physical case, geometry, boundaries, solver options, observables, and pass/fail gates.
2. Check implementation invariants, mass, no-slip, force signs, restart consistency, and post-transient stationarity.
3. Change grid spacing, domain/blockage, Mach/time step, and sampling window one factor at a time.
4. Compare the qualified CFD outputs with independent CFD/DNS and retain provenance, discrepancies, and hashes.
5. Freeze a versioned CFD dataset; only then define complete-Re development, validation, and test splits.
6. Measure ML-to-CFD error separately. A CFD change creates a new dataset version and requires retraining; CFD parameters are never tuned against neural predictions.

This release has not passed its formal spatial-grid gate. Therefore the existing $D/\Delta x=12$ neural labels remain classroom evidence rather than a grid- and domain-qualified reference solution.
"""),
    md(r"""
## 4. Executed grid study: retain failure, then refine

Grid refinement changes only $\Delta x/D$. The frozen sequence is

\[
D/\Delta x = 12,18,27,\qquad r=1.5.
\]

while keeping $Re=100$, $Ma=0.0866$, the $20D\times8D$ domain, boundary models, perturbation, $100D/U$ observation time, and $45D/U$ transient removal fixed. Changing the grid and domain together would make the source of any improvement unknowable.

Under this constant-Mach acoustic scaling, $\tau=1/2+3U(D/\Delta x)/Re$ changes algebraically with resolution to preserve $Re$. It is reported for every grid and is not tuned to improve an output.

Constant-$Ma$ refinement also retains an $O(Ma^2)\approx0.75\%$ weak-compressibility floor. That is comparable with the 1–2% grid differences and roughly 1% half-window drag variation, so a non-positive observed order is plausible. A formal asymptotic continuation should use diffusive scaling $U\propto\Delta x$ (or lower Mach) and average at least 30 shedding cycles per grid.

All three medium-to-fine changes satisfy the declared practical tolerances,
but the changes did not decrease consistently, so the sequence did not yield a
positive asymptotic order. That is a **failed formal grid gate**, not a reason
to delete a point or relax the criterion. The next declared continuation is
$D/\Delta x=40$ with the same physics, but it has not been completed in this
retained evidence.

For a scalar output $\phi$ ordered coarse, medium, fine,

\[
p=\frac{\log| (\phi_1-\phi_2)/(\phi_2-\phi_3)|}{\log r},\qquad
\phi_{h\to0}=\phi_3+\frac{\phi_3-\phi_2}{r^p-1},
\]

\[
GCI_{fine}=1.25\frac{|\phi_3-\phi_2|/|\phi_3|}{r^p-1}\times100\%.
\]

The code reports no order or GCI for a non-monotone sequence. Before examining the results, the declared fine-pair/GCI limits are 3%/5% for $\overline C_D$, 2%/3% for $St$, and 5%/8% for $L_r/D$. Each run must also pass mass, tail-drag, cycle-count, and lift-stationarity gates.

**Prediction checkpoint—write before running the next cell:** Which quantity will converge slowest? Do you expect the finest drag and Strouhal values to move toward or away from the published bands? State PASS or FAIL for each quantity before opening the retained table.
"""),
    code(r"""
GRID_EVIDENCE = REPO_ROOT / "results/cylinder_grid_convergence"
grid_metrics = pd.read_csv(GRID_EVIDENCE / "grid_metrics.csv")
grid_gci = pd.read_csv(GRID_EVIDENCE / "grid_convergence.csv")
grid_summary = json.loads((GRID_EVIDENCE / "grid_summary.json").read_text())

recomputed = []
for quantity in ("Cd_mean", "St", "Lr_over_D"):
    diagnostic = grid_convergence_diagnostics(
        grid_metrics["nodes_per_diameter"], grid_metrics[quantity]
    )
    recomputed.append({"quantity": quantity, **diagnostic})
recomputed = pd.DataFrame(recomputed)
for quantity in recomputed.quantity:
    stored = grid_gci.loc[grid_gci.quantity == quantity].iloc[0]
    current = recomputed.loc[recomputed.quantity == quantity].iloc[0]
    assert np.isclose(current.observed_order, stored.observed_order, equal_nan=True)
    assert np.isclose(current.fine_grid_gci_percent, stored.fine_grid_gci_percent,
                      equal_nan=True)

display(grid_metrics[[
    "nodes_per_diameter", "tau", "Mach", "blockage", "density_drift",
    "Cd_mean", "Cd_tail_half_change_percent", "St", "St_cycles",
    "lift_relative_rms_change", "Lr_over_D", "statistical_convergence_pass",
]])
display(grid_gci[[
    "quantity", "observed_order", "richardson_extrapolated",
    "fine_pair_relative_change_percent", "fine_grid_gci_percent",
    "fine_pair_gate_percent", "gci_gate_percent", "pass",
]])
assert grid_metrics.nodes_per_diameter.astype(int).tolist() == [12, 18, 27]
print("grid-independence gate:", grid_summary["status"].upper())
display(Image(filename=str(GRID_EVIDENCE / "cylinder_grid_independence.png")))
"""),
    md(r"""
### Interpret the result correctly

- **Statistical convergence:** did each trajectory reach a stationary sampled wake?
- **Grid convergence:** do the declared outputs stop changing as $\Delta x/D\to0$?
- **Domain sensitivity:** do transverse pitch and outlet distance still bias those outputs?
- **External validation:** do the grid/domain-qualified outputs agree with independent CFD/DNS?
- **ML error:** after all of the above, how closely does the surrogate reproduce the CFD labels?

A small ML-versus-LBM error cannot repair a biased LBM solution. The archived neural models in this release were trained on the low-cost $D/\Delta x=12$ dataset. The retained three-grid study does not establish formal asymptotic independence, so those ML claims remain explicitly educational.
"""),
    code(r"""
RUN_FULL_GRID_REGENERATION = False
if RUN_FULL_GRID_REGENERATION:
    import subprocess, sys
    subprocess.run([
        sys.executable,
        str(REPO_ROOT / "qa/run_cylinder_grid_independence.py"),
        "--root", str(REPO_ROOT), "--regenerate", "--workers", "3",
    ], check=True)
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

It captured the near wake but visibly diffused translating downstream vortices. Its retained $Re=100$ result is therefore a **failure baseline**. Because that result was inspected and used to redesign the model, $Re=100$ is no longer called blind. Random snapshot splitting would still leak nearly identical neighbouring phases and remains prohibited.
"""),
    code(r"""
pod_report = json.loads(
    (REPO_ROOT / "results/cylinder_ml/blind_re100_metrics.json").read_text()
)
print("development Re:", pod_report["split"]["training_reynolds"])
print("historical withheld Re:", pod_report["split"]["blind_reynolds"])
print("claim scope:", pod_report["claim_scope"])
"""),
    md(r"""
### Development-only POD and non-neural baseline

The POD basis is fit only on development Reynolds cases. Rank is fixed at 16 before the withheld test; it is not selected by looking at $Re=100$. The harmonic ridge baseline supplies a serious non-neural comparison rather than comparing the MLP only with a constant mean field.
"""),
    code(r"""
baseline_metrics = pod_report["harmonic_pod_baseline_metrics"]
display(pd.DataFrame([{"model": "harmonic-ridge POD", **baseline_metrics}]))
"""),
    md(r"""
### Fixed neural branch

Only the POD coefficients are learned. The spatial trunk is exactly the same development-only basis used by the baseline. Inputs and coefficient targets are standardized using development data only. The architecture and seed were frozen before the historically withheld case was opened.
"""),
    code(r"""
mlp_metrics = pod_report["neural_blind_metrics"]
display(pd.DataFrame([{"model": "two-layer MLP POD", **mlp_metrics}]))
"""),
    md(r"""
# Diagnostic gate: explain the POD failure

Before running the next cell, record:

- development Reynolds cases and the historically withheld Reynolds case;
- POD rank, MLP layers, regularization, optimizer, maximum iterations, and seed;
- the non-neural baseline specification;
- the primary metric: combined relative $L_2$ over all withheld snapshots; and
- the physical checks: divergence, solid speed, pressure gauge, and qualitative wake structure.

This archived exercise deliberately shows why low global field error and high cumulative POD energy do not certify vorticity fidelity. Do not tune this model and continue calling $Re=100$ blind.
"""),
    code(r"""
comparison = pd.DataFrame([
    {"model": "harmonic-ridge POD", **baseline_metrics},
    {"model": "two-layer MLP POD", **mlp_metrics},
])
display(comparison)
comparison.to_csv(OUTPUT / "blind_re100_ml_metrics.csv", index=False)
"""),
    code(r"""
display(Image(filename=str(
    REPO_ROOT / "results/cylinder_ml/blind_re100_lbm_vs_neural_poster.png"
)))
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
video_path = REPO_ROOT / "results/cylinder_ml/blind_re100_lbm_vs_neural.mp4"
if video_path.is_file():
    display(Video(str(video_path), embed=True, html_attributes="controls loop muted"))
else:
    raw = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cylinder_ml/blind_re100_lbm_vs_neural.mp4"
    display(HTML(f'<video src="{raw}" controls loop muted style="width:100%"></video>'))
"""),
    md(r"""
## 6. Four-frame multi-scale CNN: low-error one-step forecast and rollout audit

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
the complete `Re=105` trajectory was originally opened only after every
validation gate passed. It is now retained historical evidence, not a fresh
blind case for future redesign.

The composite objective is

\[
\mathcal L=\mathcal L_{field}
+0.20\mathcal L_{gradient}
+0.20\mathcal L_{\omega}
+0.05\mathcal L_{\nabla\cdot u}.
\]

For a fair one-step comparison, the same four frames also feed persistence,
linear, quadratic, and cubic temporal extrapolation. The CNN must beat the
best declared polynomial baseline. A separate 50-step recursive audit then
feeds the CNN output back and is reported even when it fails.
"""),
    code(r"""
cnn_metrics_path = REPO_ROOT / "results/cylinder_cnn/multiscale_cnn_metrics.json"
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
            "mean spectral incoherence": values["mean_station_complex_spectral_incoherence"],
        })
display(pd.DataFrame(rows))
display(cnn_metrics["validation_gates"])
print("best validation baseline:", cnn_metrics["validation"]["best_non_neural_baseline"])
print("best retained-case baseline:", cnn_metrics["blind"]["best_non_neural_baseline"])
"""),
    code(r"""
from IPython.display import Image
display(Image(filename=str(REPO_ROOT / "results/cylinder_cnn/re105_blind_downstream.png")))
display(Image(filename=str(REPO_ROOT / "results/cylinder_cnn/re105_retained_rollout.png")))
preview_url = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cylinder_cnn/re105_lbm_vs_multiscale_cnn.webp"
display(Image(url=preview_url))
video_url = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cylinder_cnn/re105_lbm_vs_multiscale_cnn.mp4"
display(HTML(f'<video src="{video_url}" controls loop muted style="width:100%"></video>'))
"""),
    md(r"""
### Reproduce the frozen experiment

Install the optional ML dependencies and run the development/validation stage.
The retained protocol refused to open $Re=105$ unless every $Re=100$ gate passed.

```bash
python -m pip install -e '.[ml]'
python qa/run_cylinder_multiscale_cnn.py --workers 4
python qa/run_cylinder_multiscale_cnn.py --reuse-weights --run-blind
```

Acceptance is spatial, not merely global: compare vorticity profiles,
enstrophy deviation, and complex space-time spectral coherence at
$(x-x_c)/D=2,4,6,8$, plus exact no-slip and all four non-neural baselines.
The autonomous-rollout gate is separate from the one-step gate.
"""),
    md(r"""
## 7. Phase-stable correction: 277 autonomous future fields

The recursive CNN failure identifies a specific mechanism: every predicted
field becomes part of the next input, so small transport, amplitude, and phase
errors accumulate. For a saturated low-Reynolds cylinder wake, the dominant
dynamics are approximately periodic. We therefore introduce a separate,
transparent **phase-stable learned Fourier decoder**:

\[
\widehat{\boldsymbol q}(x,y,\phi;Re)
=\boldsymbol a_0(x,y;Re)
+\sum_{k=1}^{K}\left[
\boldsymbol a_k^{s}(x,y;Re)\sin(k\phi)
+\boldsymbol a_k^{c}(x,y;Re)\cos(k\phi)\right],
\]

where \(\boldsymbol q=(u,v,p)\). Spatial coefficients are least-squares fits
to complete development trajectories and are interpolated between bracketing
development Reynolds cases. At a target Reynolds number,

\[
\phi_n=\phi_0+2\pi St(Re)\,n\Delta t^*.
\]

Only four true initial fields are used to align \(\phi_0\). After alignment,
all 277 future fields use **zero future CFD inputs**. Because the phase is
advanced explicitly instead of recursively inferred from predicted fields,
the representation cannot numerically diffuse merely through repeated
one-step feedback.

This decoder is a learned reduced-order surrogate, not a neural network. The
distinction is intentional: the CNN remains the low-error one-step model, while
the Fourier decoder is the passed long-horizon correction for the saturated,
nearly periodic regime.
"""),
    code(r"""
phase_dir = REPO_ROOT / "results/cylinder_phase"
phase_metrics = json.loads((phase_dir / "phase_stable_metrics.json").read_text())
protocol = phase_metrics["protocol"]

assert protocol["development_reynolds"] == [90, 110, 120, 140]
assert protocol["validation_reynolds"] == 100
assert protocol["fresh_test_reynolds"] == 95
assert protocol["retained_test_reynolds"] == 105
assert protocol["initial_true_frames"] == 4
assert protocol["future_cfd_inputs"] == 0
assert phase_metrics["selected_harmonics"] == 6
assert phase_metrics["all_gates_pass"]

phase_rows = []
for split_name, label in (("validation", "Validation"),
                          ("fresh_test", "Fresh test"),
                          ("retained_test", "Retained test")):
    values = phase_metrics[split_name]
    phase_rows.append({
        "split": label,
        "Re": {"validation": 100, "fresh_test": 95, "retained_test": 105}[split_name],
        "future frames": values["future_frames"],
        "global vorticity error (%)": 100 * values["vorticity_global_relative_l2"],
        "worst-frame error (%)": 100 * values["vorticity_max_frame_relative_l2"],
        "last-frame error (%)": 100 * values["vorticity_last_frame_relative_l2"],
        "CFD St": values["cfd_strouhal"],
        "decoder St": values["predicted_strouhal"],
        "St error (%)": 100 * values["strouhal_relative_error"],
        "pass": values["passes"],
    })
phase_table = pd.DataFrame(phase_rows)
display(phase_table.round(3))
"""),
    code(r"""
display(Image(filename=str(phase_dir / "phase_stable_validation.png")))
display(Image(filename=str(phase_dir / "re095_phase_stable_poster.png")))

preview_url = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.webp"
display(Image(url=preview_url))
video_url = "https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/results/cylinder_phase/re095_phase_stable_lbm_vs_decoder.mp4"
display(HTML(f'<video src="{video_url}" controls loop muted style="width:100%"></video>'))
"""),
    md(r"""
### Frozen long-horizon gates and reproduction

Harmonic order \(K\in\{2,3,4,5,6\}\) is selected only by global vorticity
error on complete-case `Re=100` validation; \(K=6\) is then frozen. The fresh
`Re=95` test is opened once. Every reported split must satisfy:

- global vorticity relative \(L_2<15\%\);
- worst-frame vorticity relative \(L_2<15\%\); and
- relative Strouhal error \(<2\%\).

The checksummed inputs are published in the `cylinder-cfd-v1` GitHub release.
After downloading them into `data/cylinder_cfd/`, reproduce the full evidence:

```bash
python qa/validate_cylinder_cfd_dataset.py
python qa/run_cylinder_phase_stable.py
```

The release workflow runs unit tests, downloads the versioned CFD inputs,
regenerates the metrics/spectra/video, and commits evidence only after all
gates pass.
"""),
    md(r"""
## Interpretation: what this exercise does and does not show

- The POD result shows how a global low-rank representation can smear translated structures despite attractive aggregate errors.
- The CNN result shows low-error one-step field forecasting across a complete withheld Reynolds interpolation case using four previous fields.
- It does **not** establish a general neural operator for arbitrary inlet functions or geometries.
- It does **not** identify the Hopf point: training is restricted to already unsteady cases and phase is supplied.
- A low one-step field error does not guarantee correct force, frequency, or long-time phase; the retained recursive audit demonstrates that distinction.
- The CNN one-step claim is useful only if it beats the best polynomial baseline and retains downstream amplitude, profile, complex spectrum, and physical diagnostics.
- The phase-stable decoder passes a 277-frame autonomous test for saturated periodic wakes within the bracketed Reynolds interval; it does **not** establish arbitrary-transient, geometry, or extrapolation generality.

For a research extension, use longer phase-aligned trajectories, train/validation/test Reynolds bands, multiple seeds, force consistency, 20–50-cycle phase drift, and independent high-fidelity DNS. None of those claims belongs in this classroom release unless executed.
"""),
    md(r"""
## Required deliverables

1. One D2Q9 derivation page with your computed viscosity, relaxation time, Mach number, and blockage for every case.
2. A four-panel $Re=100$ field/force figure with readable axes and unobstructed legends.
3. The complete sweep table containing density drift, $\overline C_D$, $C_{L,\mathrm{rms}}$, $St$, and $L_r/D$.
4. Evidence for attached/no-bubble, steady-recirculating, and periodic-shedding regimes; explain any mismatch.
5. The complete $D/\Delta x=12,18,27$ grid table, statistical gates,
   fine-pair changes, the retained asymptotic/GCI failure, and a justified next
   refinement without changing the thresholds.
6. The archived POD failure, including why $Re=100$ is now validation rather than blind.
7. The frozen CNN specification, polynomial-baseline comparison, validation gates, and retained $Re=105$ result.
8. Enstrophy/profile/complex-spectral evidence at $2D,4D,6D,8D$ and the retained 50-step rollout audit.
9. The phase-decoder split, selected harmonic order, 277-frame errors, and LBM-versus-decoder Strouhal comparison.
10. A short paragraph distinguishing grid independence, domain independence,
    external CFD validation, and ML-to-CFD agreement.

### Exercises

1. Set the initial perturbation to zero. How long does round-off take to seed shedding at $Re=100$?
2. At $Re=40,46,48,60$, fit the logarithm of the $C_L$ envelope. Which cases grow and which decay?
3. Double the transverse size at fixed $D$. Quantify changes in $St$, drag, and recirculation length.
4. Replace random snapshot splitting with complete-Re splitting and compare the two reported errors. Explain the leakage.
5. Remove phase from both models. What failure is visible even if the mean field remains plausible?
6. Hold out an edge case rather than an interpolated case. Why is extrapolation harder?
7. Perturb the decoder Strouhal number by 1%. Predict and measure the accumulated phase error after 20 shedding cycles.
"""),
    md(r"""
## References

- S. Succi (2001), *The Lattice Boltzmann Equation for Fluid Dynamics and Beyond*, Oxford University Press.
- Q. Zou & X. He (1997), “On pressure and velocity boundary conditions for the lattice Boltzmann BGK model,” *Physics of Fluids* 9, 1591–1598.
- C. H. K. Williamson (1996), “Vortex dynamics in the cylinder wake,” *Annual Review of Fluid Mechanics* 28, 477–539. [doi:10.1146/annurev.fl.28.010196.002401](https://doi.org/10.1146/annurev.fl.28.010196.002401)
- C. P. Jackson (1987), “A finite-element study of the onset of vortex shedding in flow past variously shaped bodies,” *Journal of Fluid Mechanics* 182, 23–45.
- I. B. Celik et al. (2008), “Procedure for estimation and reporting of uncertainty due to discretization in CFD applications,” *Journal of Fluids Engineering* 130, 078001. [doi:10.1115/1.2960953](https://doi.org/10.1115/1.2960953)
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
            "lab": "Week 7 LBM cylinder wake",
            "protocol_version": "2026.09",
        },
    },
)
nbf.write(notebook, TARGET)
print(TARGET)
