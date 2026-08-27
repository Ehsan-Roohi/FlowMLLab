#!/usr/bin/env python3
"""Build the additive Week-4.1 classical-ROM cavity notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "week04" / "W4_1_Classical_ROM_Cavity.ipynb"


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
        r"""# Week 4.1 — Classical POD–Galerkin and POD–DEIM for the lid-driven cavity

<!-- MIE690A article-aligned validation v3 -->
<!-- MIE690A additive course extension v1; prior lectures and notebooks remain unchanged -->

This optional lab fills the conceptual gap between the validated cavity CFD solver and the POD–DeepONet laboratory. It uses **the same nondimensional two-dimensional square lid-driven cavity**, the same streamfunction–vorticity formulation, the same wall treatment, and the same development/validation/blind Reynolds-number discipline as the rest of FlowMLLab. No Burgers, heat-equation, or pipe-flow substitute is introduced.

By the end of the lab, you should be able to:

1. distinguish POD reconstruction from a dynamical reduced-order model (ROM);
2. derive a centered intrusive POD–Galerkin model for cavity vorticity;
3. explain why a small modal state can still have full-order online cost;
4. use the discrete empirical interpolation method (DEIM) to hyper-reduce nonlinear convection;
5. select rank without opening the blind Reynolds cases;
6. audit field error, vorticity error, walls, divergence, vortex position, timing, and break-even count; and
7. state when POD–DeepONet is useful and when a classical ROM is the more transparent baseline.

The implementation is original MIT-licensed FlowMLLab code. It was motivated by the classical FOM → snapshots → POD → projection teaching sequence; no source code was copied from the unlicensed `mini-rom` repository.
"""
    ),
    markdown(
        r"""## 1. What is new—and what is not?

Project Track 3 already asks whether a small number of POD modes can reconstruct cavity fields and whether a neural branch can predict modal coefficients from Reynolds number. That is a **representation and regression** question. Here the modal coefficients evolve because the Navier–Stokes vorticity equation is projected onto the POD space. The governing equation, rather than a coefficient regressor, supplies the time derivative.

The full-order model (FOM) is



\[
\frac{\partial \omega}{\partial t}
+u\frac{\partial \omega}{\partial x}
+v\frac{\partial \omega}{\partial y}
=\frac{1}{Re}\nabla^2\omega,
\qquad
\nabla^2\psi=-\omega,
\qquad
u=\frac{\partial\psi}{\partial y},\quad
v=-\frac{\partial\psi}{\partial x}.
\]

The top wall has (u=1,v=0); the other walls are no-slip. The ROM state contains only interior vorticity. Each reconstructed state is mapped through the same discrete Poisson equation, so streamfunction and velocity remain kinematically coupled. This is why the reported divergence remains near round-off and why the prescribed velocity walls are exact. Pressure is not a dynamical state in this formulation and is deliberately not added as an unsupported ROM output.
"""
    ),
    markdown(
        r"""## 2. Centered POD and Galerkin projection

Let (q(t;Re)) denote the vector of interior vorticity values. Development snapshots are centered,

\[
q(t;Re)\approx \overline q+\Phi a(t;Re),
\qquad \Phi^{T}\Phi=I,
\]

where the columns of (Phi) are retained left singular vectors. Substitution into the semi-discrete FOM (dot q=F(q;Re)) and Galerkin projection give

\[
\dot a=\Phi^T F(\overline q+\Phi a;Re).
\]

This is a dynamical ROM, but it is not automatically a fast ROM. A direct evaluation reconstructs all interior values, solves the full Poisson problem, forms velocity and vorticity gradients over the complete grid, and only then projects the result. The modal vector may contain 16 numbers while the nonlinear work still scales with the 961 interior grid values. That distinction is the central scientific lesson of this notebook.

Centering is essential. Without it, the first mode can mostly encode the large mean cavity circulation, making cumulative energy look more favorable while using modes inefficiently for Reynolds- and time-dependent changes. Boundary lifting is implicit here because all trajectories share the same lid and wall conditions: the mean carries the affine part and centered modes carry homogeneous variations.
"""
    ),
    code(
        """from pathlib import Path
import json, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(
    candidate for candidate in [Path.cwd(), *Path.cwd().parents]
    if (candidate / "results" / "cavity_rom" / "validation_summary.json").exists()
)
RESULTS = ROOT / "results" / "cavity_rom"
sys.path.insert(0, str(ROOT))
from flowmllab import cavity_rom

print("course root:", ROOT)
print("cavity ROM module:", cavity_rom.CAVITY_ROM_VERSION)
print("frozen evidence:", RESULTS)
"""
    ),
    markdown(
        r"""## 3. Validate the added FOM path before reducing it

The snapshot collector is not accepted merely because it uses familiar formulas. At (Re=100) and (400), it was run for exactly the time steps recorded in the fixed 65×65 FlowMLLab archive. Velocity and vorticity must match the already accepted Week-4 implementation to numerical round-off. Those two reproduced fields are then checked against the independent Ghia centerline data already used elsewhere in the course.

A second study uses (Re=275) at (t=5). Spatial errors on 25×25, 33×33, and 49×49 grids are measured against a 65×65 reference interpolated to each comparison grid. Temporal errors for (Delta t=0.004,0.002,0.001) are measured against (Delta t=0.0005) at fixed grid and final time. We require monotonic reduction; apparent agreement from one convenient grid is not a convergence study.
"""
    ),
    code(
        """fom_validation = pd.read_csv(RESULTS / "fom_validation.csv")
convergence = pd.read_csv(RESULTS / "convergence.csv")
display(fom_validation)
display(convergence)

assert fom_validation["archive_relative_L2_u_v_omega"].max() < 5e-13
for study in ("grid", "time_step"):
    errors = convergence.loc[convergence["study"] == study, "relative_L2_uv"].to_numpy()
    assert np.all(np.diff(errors) < 0), f"{study} refinement is not monotone"
"""
    ),
    markdown(
        r"""## 4. Hyper-reduction with DEIM

Write the full-order right-hand side as convection plus diffusion,

\[
F(q;Re)=g(q)+\frac{1}{Re}L(q).
\]

For this discretization, the wall treatment makes diffusion affine in the interior state. Its reduced offset and matrix are therefore assembled exactly once at unit viscosity and scaled by (1/Re) online. The nonlinear convection snapshots form a separate centered basis (U_g). QDEIM chooses interpolation indices (P), and the projected nonlinearity is approximated by

\[
\Phi^Tg(q)\approx
\Phi^T\overline g+
\Phi^TU_g(P^TU_g)^{-1}P^T[g(q)-\overline g].
\]

The implementation does not reconstruct the entire nonlinear field online. At the selected points, affine maps provide (u,v,\partial_x\omega,partial_y\omega), after which only the sampled products (-u\partial_x\omega-v\partial_y\omega) are evaluated. The frozen model archive contains numeric arrays only and is loaded with `allow_pickle=False`.
"""
    ),
    markdown(
        r"""## 5. Leakage-free model selection

Development Reynolds numbers are (100,150,200,225,250,350,400). Complete trajectories at (Re=300) are used for selection. The untouched blind cases are (175,275,375), matching the physical split used elsewhere in FlowMLLab. Candidate POD ranks are (4,8,12,16), with the DEIM dimension equal to rank. The predeclared rule selects the smallest candidate for which **both** standard POD–Galerkin and POD–DEIM have less than 1% maximum-in-time relative velocity error on (Re=300).

This table also demonstrates why cumulative POD energy is not an accuracy certificate. Rank 12 retains more than 99.9% energy yet fails the dynamical gate, especially after nonlinear hyper-reduction. The decision uses a-posteriori trajectory error, not a visually attractive singular-value threshold.
"""
    ),
    code(
        """selection = pd.read_csv(RESULTS / "selection.csv")
display(selection.assign(
    POD_Galerkin_max_percent=100*selection["POD_Galerkin_max_time_relative_L2_uv"],
    POD_DEIM_max_percent=100*selection["POD_DEIM_max_time_relative_L2_uv"],
))

accepted = selection[selection["passes_one_percent_gate"]]
assert len(accepted) == 1
assert int(accepted.iloc[0]["rank"]) == 16
print("Frozen choice: POD rank 16, DEIM dimension 16")
"""
    ),
    markdown(
        r"""# Stop: blind-test gate

Before displaying the next table, record the development cases, selection case, candidate set, 1% rule, and frozen rank. Do not change rank, DEIM dimension, time window, or normalization after inspecting the blind results. Any redesign requires a new untouched physical test family.
"""
    ),
    code(
        """blind = pd.read_csv(RESULTS / "blind_metrics.csv")
display(blind.assign(
    final_Euv_percent=100*blind["final_relative_L2_uv"],
    max_time_Euv_percent=100*blind["max_time_relative_L2_uv"],
))

assert sorted(blind["Re"].unique().tolist()) == [175, 275, 375]
assert blind["max_time_relative_L2_uv"].max() < 0.01
assert blind["final_relative_L2_omega"].max() < 0.01
assert blind["wall_rms_error"].max() == 0.0
assert blind["divergence_l2"].max() < 1e-12
"""
    ),
    markdown(
        r"""## 6. Run a fresh blind query from the portable model

The following cell does not read a stored prediction. It loads the frozen numeric model, independently reruns the (Re=275) FOM, integrates the POD–Galerkin baseline and POD–DEIM model to (t=5), and recomputes velocity error. This is a small but meaningful restart-and-run check of the public API. Because the FOM and ROM use the same forward-Euler time step, the comparison isolates state reduction and nonlinear approximation rather than mixing temporal schemes.
"""
    ),
    code(
        """model = cavity_rom.load_deim_model(RESULTS / "cavity_rom_model.npz")
fom = cavity_rom.simulate_fom(275, n=33, dt=0.002, steps=2500, snapshot_stride=25)
galerkin = cavity_rom.simulate_pod_galerkin(model.pod, 275, 0.002, 2500, 25)
deim = cavity_rom.simulate_pod_deim(model, 275, 0.002, 2500, 25)

pg_error = cavity_rom.velocity_error_trajectory(fom["states"], galerkin["states"], 33)
deim_error = cavity_rom.velocity_error_trajectory(fom["states"], deim["states"], 33)
print(f"fresh POD-Galerkin max-time E_uv: {100*pg_error.max():.4f}%")
print(f"fresh POD-DEIM max-time E_uv:      {100*deim_error.max():.4f}%")
assert pg_error.max() < 0.01 and deim_error.max() < 0.01

truth = cavity_rom.state_to_fields(fom["states"][-1], 33)
prediction = cavity_rom.state_to_fields(deim["states"][-1], 33)
x = np.linspace(0, 1, 33)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
speed = np.hypot(truth["u"], truth["v"])
error = np.hypot(prediction["u"]-truth["u"], prediction["v"]-truth["v"])
axes[0].contourf(x, x, speed, 18, cmap="viridis")
axes[0].streamplot(x, x, truth["u"], truth["v"], color="white", density=.75, linewidth=.4)
axes[1].contourf(x, x, np.hypot(prediction["u"], prediction["v"]), 18, cmap="viridis")
axes[2].contourf(x, x, error, 18, cmap="magma")
for ax, title in zip(axes, ["FOM speed", "POD-DEIM speed", "vector error"]):
    ax.set(title=title, xlabel="x/L", ylabel="y/L", aspect="equal")
plt.tight_layout()
"""
    ),
    markdown(
        r"""## 7. Cost diagnosis and break-even point

Timing covers a complete (t=0) to (5) query plus one final full-field reconstruction. Seven measurements follow one warm-up and the median is reported with Python, NumPy, SciPy, and platform metadata. Offline time includes the seven development FOM trajectories, the centered POD, nonlinear snapshot evaluation, and nonlinear SVD; selection sweeps and blind FOM runs are not hidden inside the online number.

Standard POD–Galerkin is expected to be no faster—and can be slower—because it still performs the full Poisson and nonlinear work every step, then pays an additional projection cost. POD–DEIM removes that bottleneck. The break-even count is

\[
N_{\mathrm{BE}}=
\frac{T_{\mathrm{offline}}}{T_{\mathrm{FOM/query}}-T_{\mathrm{DEIM/query}}}.
\]

A speedup is therefore meaningful for repeated parametric queries, not for a single calculation whose training data were created solely for that query.
"""
    ),
    code(
        """timing = json.loads((RESULTS / "timing.json").read_text())
display(pd.DataFrame([timing]).T.rename(columns={0: "recorded value"}))
assert timing["POD_Galerkin_speedup"] < 1.0
assert timing["POD_DEIM_speedup"] > 2.0
"""
    ),
    markdown(
        r"""![Frozen cavity-ROM validation summary](../../results/cavity_rom/cavity_rom_validation.png)

## 8. Interpretation, limitations, and comparison with POD–DeepONet

The blind velocity error remains below 0.64%, final vorticity error below 0.56%, wall error exactly zero, and divergence near $10^{-16}$. Those statements apply only to the fixed square geometry, fixed lid boundary condition, $100\le Re\le400$, 33×33 ROM grid, forward-Euler interval $0\le t\le5$, and the recorded development split. They are not evidence for new geometries, turbulent Reynolds numbers, arbitrary lid functions, or indefinite-time stability.

POD–Galerkin is intrusive: it needs access to the discrete governing operator, but it uses relatively little data and offers direct dynamical interpretation. POD–DEIM adds offline work and an approximation of nonlinear convection in exchange for real online acceleration. POD–DeepONet is non-intrusive after data generation and can make extremely fast parameter-to-field queries, but its branch must learn parameter dependence and its reliability is tied to the training distribution. No method should be declared universally superior.

This first extension deliberately stops before closure stabilization, pressure ROMs, geometry variation, or turbulent cavity flow. A low-rank Galerkin model can become unstable outside the validated interval; adding an eddy-viscosity correction merely to suppress failure would require its own selection set and physical audit. Likewise, increasing DEIM dimension after inspecting blind cases would violate the frozen protocol.

### Required student evidence

Submit the FOM reproduction table, refinement table, rank-selection table, every blind case, one fresh-query field/error figure, timing with break-even count, and a paragraph explaining why rank 12 fails despite 99.9% cumulative energy. State one condition under which you would prefer POD–Galerkin, POD–DEIM, or POD–DeepONet. A high-quality report distinguishes representation error, projection/closure error, DEIM nonlinear error, discretization error, and parameter-generalization error rather than collapsing them into one number.

### Further reading

- Ghia, Ghia, and Shin (1982), the independent cavity benchmark used throughout FlowMLLab: https://doi.org/10.1016/0021-9991(82)90058-4
- Berkooz, Holmes, and Lumley (1993), a classical review of POD in turbulent flows: https://doi.org/10.1146/annurev.fl.25.010193.002543
- Chaturantabut and Sorensen (2010), the DEIM method and nonlinear-cost motivation used here: https://doi.org/10.1137/090766498
"""
    ),
    markdown(
        r"""## 9. Optional full regeneration

The public evidence can be regenerated by the dedicated validation program. It reproduces accepted 65×65 fields, performs grid and time-step refinement, carries out selection without blind leakage, opens all blind cases once, measures timing, serializes the fitted model without pickle, and rewrites the machine-readable tables and figure. Runtime depends on the linear-algebra backend. Keep this switch `False` during ordinary reading.
"""
    ),
    code(
        """REGENERATE_ALL = False
if REGENERATE_ALL:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "qa" / "run_cavity_rom_validation.py")],
        cwd=ROOT,
        check=False,
    )
    assert completed.returncode == 0
    print("Validation package regenerated.")
else:
    print("Using the frozen, versioned validation package.")
"""
    ),
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

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"w41-{index:03d}"

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(OUTPUT)
