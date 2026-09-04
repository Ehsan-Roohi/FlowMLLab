"""Build the two Week-8 gas-dynamics student notebooks from reviewable cells."""

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
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }

        @staticmethod
        def new_notebook(cells, metadata):
            return {
                "cells": cells,
                "metadata": metadata,
                "nbformat": 4,
                "nbformat_minor": 5,
            }

    class _NotebookFormat:
        v4 = _V4()

        @staticmethod
        def write(notebook, target):
            Path(target).write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
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
    _flowmllab_os.chdir(_flowmllab_root / "notebooks/week08")

from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except ModuleNotFoundError:
    display = print

REPO_ROOT = next(
    candidate for candidate in (Path.cwd(), *Path.cwd().parents)
    if (candidate / "results/gas_dynamics_week8").is_dir()
)
if str(REPO_ROOT) not in _flowmllab_sys.path:
    _flowmllab_sys.path.insert(0, str(REPO_ROOT))
RESULTS = REPO_ROOT / "results/gas_dynamics_week8"
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9})
print("Python:", platform.python_version())
print("FlowMLLab root:", REPO_ROOT)
"""


def lab1_cells():
    return [
        md(r"""
# Week 8 Lab 1 - Exact gas dynamics before machine learning

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab1_Exact_Gas_Dynamics_Student.ipynb)

**Runtime:** CPU, normally under 3 minutes. **Required background:** Mach number,
perfect-gas relations, and basic Python arrays.

### Central question

Before training a neural surrogate, can you identify the exact relation, its
physical domain, every solution branch, and the numerical operation that the
network is supposed to replace?

### Learning outcomes

By the end of this lab, you should be able to:

1. compute Rayleigh and Fanno reference ratios and explain choking from either side of $M=1$;
2. recover weak and strong oblique-shock roots without mixing the branches;
3. locate a normal shock inside a converging-diverging nozzle with a bounded root solve;
4. solve the shock-tube pressure compatibility equation and verify its residual;
5. classify a task as closed-form evaluation, bracketed inversion, ODE integration, or CFD; and
6. define what must remain exact when a learned approximation is introduced.
"""),
        md(r"""
## Scientific-use contract

- The gas is calorically perfect unless a cell explicitly changes $\gamma$.
- All inverse problems are evaluated only on a declared physical branch and domain.
- Exact relations and bracketed roots are the references; a neural prediction is never its own validation target.
- A converged nonlinear solver is not automatically a physically admissible solution.
- The nine chapter notebooks in `Introduction-to-Compressible-Flows` remain the detailed classical source. This lab is a compact bridge into FlowMLLab.
- The retained formulas are synchronized to `GasDynamicsSciML` commit
  `374431a1033138f56e2752bf8bbf9b75a454d80c`.
- The teaching interpretation follows the author-supplied revised manuscript
  *Physics-Guided Neural Surrogates for Canonical Compressible Thermal-Fluid
  Relations* (`AITF-D-26-00044R1`). A revision identifier is not treated as
  evidence of editorial acceptance.
"""),
        code(BOOTSTRAP),
        code(r"""
from flowmllab.gas_dynamics import (
    GAMMA, area_mach, fanno_inverse_friction_length, fanno_ratios,
    mach_from_area, nozzle_back_pressure, nozzle_shock_area,
    oblique_beta, oblique_detachment, oblique_theta,
    rayleigh_inverse_t0, rayleigh_ratios,
    shock_tube_pressure_ratio, shock_tube_residual_general,
)

print("gamma =", GAMMA)
"""),
        md(r"""
## 1. Branches are part of the problem definition

Rayleigh flow models heat transfer in a constant-area frictionless duct. Fanno
flow models adiabatic flow with wall friction. In both models, the sonic state
is the reference state and the same normalized target can correspond to a
subsonic and a supersonic Mach number.

Predict first:

1. Does heating drive both branches toward or away from $M=1$?
2. Can a scalar target such as $T_0/T_0^*$ uniquely determine Mach number?
3. Why must `branch` be an input to an inverse solver or learned model?
"""),
        code(r"""
m_sub = np.linspace(0.15, 0.995, 260)
m_sup = np.linspace(1.005, 3.0, 300)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
for mach, color, label in ((m_sub, "#2F75B5", "subsonic"),
                           (m_sup, "#C44536", "supersonic")):
    axes[0].plot(mach, rayleigh_ratios(mach)[:, 4], color=color, lw=2.3, label=label)
    axes[1].plot(mach, fanno_ratios(mach)[:, 4], color=color, lw=2.3, label=label)
for axis in axes:
    axis.axvline(1.0, color="#17365D", ls="--")
    axis.grid(alpha=0.3)
    axis.set_xlabel("Mach number")
axes[0].set(title="Rayleigh", ylabel=r"$T_0/T_0^*$")
axes[1].set(title="Fanno", ylabel=r"$4fL^*/D$", ylim=(0, 4))
axes[0].legend(frameon=False)
plt.show()

target_t0 = 0.82
rayleigh_roots = {
    branch: rayleigh_inverse_t0(target_t0, branch)
    for branch in ("subsonic", "supersonic")
}
target_friction = 0.18
fanno_roots = {
    branch: fanno_inverse_friction_length(target_friction, branch)
    for branch in ("subsonic", "supersonic")
}
display(pd.DataFrame([rayleigh_roots, fanno_roots], index=["Rayleigh", "Fanno"]))

assert rayleigh_roots["subsonic"] < 1.0 < rayleigh_roots["supersonic"]
assert fanno_roots["subsonic"] < 1.0 < fanno_roots["supersonic"]
"""),
        md(r"""
### Checkpoint 1

Write one sentence explaining why a single-output regression
`target ratio -> Mach` is ill-posed if branch identity is hidden. Then change
each target above and verify by forward substitution that both returned roots
recover it.
"""),
        code(r"""
rayleigh_closure = {
    branch: float(rayleigh_ratios(mach)[4] - target_t0)
    for branch, mach in rayleigh_roots.items()
}
fanno_closure = {
    branch: float(fanno_ratios(mach)[4] - target_friction)
    for branch, mach in fanno_roots.items()
}
print("Rayleigh inverse closure:", rayleigh_closure)
print("Fanno inverse closure:", fanno_closure)
assert max(abs(value) for value in rayleigh_closure.values()) < 1e-9
assert max(abs(value) for value in fanno_closure.values()) < 1e-9
"""),
        md(r"""
## 2. Oblique shocks: two attached roots and a detachment limit

For upstream Mach number $M_1$, shock angle $\beta$, and flow deflection
$\theta$, the theta-beta-M relation is

$$
\tan\theta = 2\cot\beta\,
\frac{M_1^2\sin^2\beta-1}{M_1^2(\gamma+\cos 2\beta)+2}.
$$

Below the maximum turning angle there are weak and strong roots. At the maximum
they merge; above it, an attached oblique shock is impossible. This is a
topological feature of the solution manifold, not a training-data nuisance.
"""),
        code(r"""
mach_1 = 2.0
beta_peak, theta_max = oblique_detachment(mach_1)
theta_requested = np.radians(12.0)
beta_weak = oblique_beta(mach_1, theta_requested, "weak")
beta_strong = oblique_beta(mach_1, theta_requested, "strong")

beta_grid = np.linspace(np.arcsin(1 / mach_1) + 1e-4, np.pi / 2 - 1e-4, 500)
theta_grid = oblique_theta(mach_1, beta_grid)
plt.figure(figsize=(7.4, 4.4))
plt.plot(np.degrees(beta_grid), np.degrees(theta_grid), color="#2A9D8F", lw=2.4)
plt.scatter(np.degrees([beta_weak, beta_strong]), [12, 12],
            color=["#2F75B5", "#C44536"], s=65, label="weak / strong roots")
plt.scatter(np.degrees([beta_peak]), np.degrees([theta_max]),
            color="#E9A23B", s=65, label="detachment")
plt.xlabel(r"shock angle $\beta$ (deg)")
plt.ylabel(r"turn angle $\theta$ (deg)")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.show()

print(f"weak beta = {np.degrees(beta_weak):.3f} deg")
print(f"strong beta = {np.degrees(beta_strong):.3f} deg")
print(f"theta_max = {np.degrees(theta_max):.3f} deg")
assert beta_weak < beta_peak < beta_strong
"""),
        md(r"""
## 3. Nozzle shock location: bounded inverse design

For a fixed exit-to-throat area ratio, an internal normal shock maps its area
location $A_s/A_t$ to a back-pressure ratio. The inverse must remain in
$1 < A_s/A_t < A_e/A_t$. A generic unconstrained regressor can violate that
geometry even when its mean error looks small.
"""),
        code(r"""
exit_area_ratio = 2.5
shock_area_grid = np.linspace(1.0001, exit_area_ratio - 0.0001, 240)
back_pressure_grid = np.array([
    nozzle_back_pressure(exit_area_ratio, area) for area in shock_area_grid
])

target_index = 135
target_back_pressure = float(back_pressure_grid[target_index])
recovered_area = nozzle_shock_area(exit_area_ratio, target_back_pressure)

plt.figure(figsize=(7.4, 4.4))
plt.plot(shock_area_grid, back_pressure_grid, color="#E9A23B", lw=2.4)
plt.scatter([recovered_area], [target_back_pressure], color="#C44536", s=65)
plt.xlabel(r"shock area $A_s/A_t$")
plt.ylabel(r"back pressure $P_b/P_{01}$")
plt.grid(alpha=0.3)
plt.show()

print("target back pressure:", target_back_pressure)
print("recovered shock area:", recovered_area)
assert 1.0 < recovered_area < exit_area_ratio
assert abs(nozzle_back_pressure(exit_area_ratio, recovered_area) - target_back_pressure) < 1e-9
"""),
        md(r"""
## 4. Shock tube: solve the compatibility equation, then inspect the residual

The star pressure behind the incident shock and ahead of the contact surface is
implicit. A bracketed scalar root is the transparent reference. The generalized
solver may also vary driver temperature, both heat-capacity ratios, and the gas
constant ratio; this is the five-input problem used later to study dimensional scaling.
"""),
        code(r"""
driver_ratios = np.geomspace(1.05, 50.0, 100)
star_pressures = np.array([shock_tube_pressure_ratio(value) for value in driver_ratios])
residuals = shock_tube_residual_general(
    star_pressures, driver_ratios, np.ones_like(driver_ratios)
)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
axes[0].loglog(driver_ratios, star_pressures, color="#2F75B5", lw=2.4)
axes[0].set(xlabel=r"$P_4/P_1$", ylabel=r"$P_2/P_1$", title="implicit pressure map")
axes[1].semilogx(driver_ratios, np.abs(residuals), color="#C44536", lw=2.2)
axes[1].set(xlabel=r"$P_4/P_1$", ylabel="absolute compatibility residual", title="closure check")
for axis in axes:
    axis.grid(alpha=0.3)
plt.show()

print("maximum compatibility residual:", np.max(np.abs(residuals)))
assert np.max(np.abs(residuals)) < 1e-9
"""),
        md(r"""
## 5. Continue into the nine classical chapter notebooks

These are the canonical, CI-tested educational notebooks in
[`Introduction-to-Compressible-Flows`](https://github.com/Ehsan-Roohi/Introduction-to-Compressible-Flows)
at commit `fc14721ae80f48da63e55955c1caf096d8448f7b`:

| Topic | Numerical idea | Open in Colab |
|---|---|---|
| Rayleigh flow | sonic reference, heat-addition branches, bisection | [Chapter 4](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter04/04_rayleigh_flow_ai_solver.ipynb) |
| Emanuel oblique shock | explicit weak/strong construction | [6.1](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter06/06_01_emanuel_oblique_shock.ipynb) |
| Shock polar | velocity-space geometry | [6.2](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter06/06_02_shock_polar.ipynb) |
| Shock collision and slip line | pressure compatibility | [6.3](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter06/06_03_oblique_shock_collision_slip_line.ipynb) |
| Shock tube | incident shock, expansion, contact matching | [7.1](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter07/07_01_shock_tube_pressure_solver.ipynb) |
| Interacting shocks | corrected nonlinear pressure-ratio equation | [7.2](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter07/07_02_interacting_shock_pressure_ratios.ipynb) |
| C-D nozzle | normal-shock location | [Chapter 8](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter08/08_normal_shock_location_cd_nozzle.ipynb) |
| Conical flow | Taylor-Maccoll RK4 integration | [10.1](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter10/10_01_conical_flow_from_shock_angle.ipynb) |
| Cone sweep | ODE event detection and surface state | [10.2](https://colab.research.google.com/github/Ehsan-Roohi/Introduction-to-Compressible-Flows/blob/main/notebooks/chapter10/10_02_taylor_maccoll_cone_sweep.ipynb) |

### Solver classification

For each row, record whether the reference is an algebraic evaluation, a
branch-wise scalar root, an ODE initial-value/event problem, or a multidimensional
CFD calculation. That classification determines the proper baseline and validation.
"""),
        md(r"""
## 6. Bridge to multidimensional compressible CFD

The separate
[`SU2-Diamond-Airfoil-Verification`](https://github.com/Ehsan-Roohi/SU2-Diamond-Airfoil-Verification)
repository advances from canonical relations to a Mach-3 diamond airfoil at
$\alpha=0,4,8$ degrees with Euler, laminar Navier-Stokes, and SST RANS configurations.

At the frozen Week-8 source commit, only the sharp-wall `euler_alpha0` case is a
**qualified teaching reference**. The other eight distributed cases remain
unverified. Therefore they appear here as a future verification project, not as
accepted CFD labels or ML training data.

A student must check residual reduction, force-window stability, physicality
warnings, symmetry, shock angle, and wall resolution before promoting any SU2
output into the FlowMLLab evidence chain.
"""),
        md(r"""
## Exit ticket

Submit a one-page evidence card containing:

1. one inverse problem and its declared domain;
2. every valid branch or bound;
3. the exact/numerical reference operation;
4. one closure residual evaluated by forward substitution;
5. the strongest non-neural baseline you would test before an MLP; and
6. one result that would make you reject the learned model despite a small global error.

Proceed to Lab 2 only after your reference solver and branch contract are explicit.
"""),
    ]


def lab2_cells():
    return [
        md(r"""
# Week 8 Lab 2 - When should a gas-dynamics surrogate be neural?

<!-- MIE690A article-aligned validation v4 -->

<!-- FLOWMLLAB_COLAB_LAUNCH_V1 -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week08/W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb)

**Runtime:** CPU, normally under 4 minutes. The small branch experiment trains
deterministically with scikit-learn; all five research-scale benchmark results
are read from immutable CSV evidence.

### Central question

Does a neural network add value relative to an exact relation, a bracketed root,
and interpolation - and is that conclusion still true near domain edges or when
the number of physical inputs grows?

### Learning outcomes

By the end, you should be able to:

1. expose the failure caused by hiding a multi-valued solution branch;
2. compare an MLP with matched interpolation using both error and coverage;
3. distinguish an ordinary blind split from an omitted-edge generalization test;
4. explain the matched-budget dimensional crossover in the generalized shock tube;
5. inspect physical bounds and equation residuals alongside relative norms; and
6. write a limited claim that says when interpolation remains preferable.
"""),
        md(r"""
## Evidence contract

The authoritative research code is
[`GasDynamicsSciML`](https://github.com/Ehsan-Roohi/GasDynamicsSciML) at commit
`374431a1033138f56e2752bf8bbf9b75a454d80c`. FlowMLLab stores byte-identical
copies of the small CSV/JSON evidence files so this lab can run offline. A
separate FlowMLLab audit adds a local RBF baseline trained on the same 4096
random states as the MLP; it is not presented as upstream article evidence.

Do not claim that neural models replace exact relations or high-fidelity CFD.
On covered one-dimensional tables, interpolation is often faster and more
accurate. The defensible neural advantage emerges when branch constraints,
coverage, derivatives, repeated root solves, or higher-dimensional storage
matter under a declared operating domain.

The accompanying revised manuscript is used for motivation, terminology,
training protocol, diagnostics, and limitations. Numerical claims in this
notebook still come from the frozen public repository and its machine-readable
evidence—not from values copied out of a plotted curve.
"""),
        code(BOOTSTRAP),
        code(r"""
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flowmllab.gas_dynamics import (
    load_week8_evidence, rayleigh_ratios, validate_week8_evidence,
)

report = validate_week8_evidence(REPO_ROOT)
evidence = load_week8_evidence(REPO_ROOT)
print(json.dumps(report, indent=2, sort_keys=True))
assert report["status"] == "pass"
"""),
        md(r"""
## 1. A deliberately ill-posed inverse

We first construct a small teaching experiment. Subsonic and supersonic
Rayleigh states can share the same $T_0/T_0^*$ value. A naive network sees only
that scalar target; two branch experts see the same target **and** a declared
branch through separate models.

Predict which model will average incompatible roots. This cell is pedagogical;
the five headline results later in the notebook come from the frozen research pipeline.
"""),
        code(r"""
m_sub = np.linspace(0.15, 0.95, 180)
m_sup = np.linspace(1.05, 3.00, 180)
x_sub = rayleigh_ratios(m_sub)[:, 4]
x_sup = rayleigh_ratios(m_sup)[:, 4]
x = np.r_[x_sub, x_sup]
y = np.r_[m_sub, m_sup]
branch = np.r_[np.zeros_like(m_sub, dtype=int), np.ones_like(m_sup, dtype=int)]

index = np.arange(len(x))
test = index % 5 == 0
train = ~test

def small_mlp(seed):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(24, 24), solver="lbfgs", max_iter=3000,
            alpha=1e-8, random_state=seed,
        ),
    )

naive = small_mlp(8)
naive.fit(x[train, None], y[train])
prediction_naive = naive.predict(x[test, None])

prediction_experts = np.empty(test.sum())
for branch_id in (0, 1):
    selected_train = train & (branch == branch_id)
    selected_test = branch[test] == branch_id
    expert = small_mlp(8 + branch_id)
    expert.fit(x[selected_train, None], y[selected_train])
    prediction_experts[selected_test] = expert.predict(x[test][selected_test, None])

def relative_l2(prediction, reference):
    return float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))

branch_demo = pd.DataFrame({
    "model": ["naive single-valued MLP", "declared two-expert MLP"],
    "relative_L2": [relative_l2(prediction_naive, y[test]),
                    relative_l2(prediction_experts, y[test])],
    "max_absolute_error": [np.max(np.abs(prediction_naive - y[test])),
                           np.max(np.abs(prediction_experts - y[test]))],
})
display(branch_demo)
assert branch_demo.loc[1, "relative_L2"] < 0.1 * branch_demo.loc[0, "relative_L2"]
"""),
        code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True, sharey=True,
                         constrained_layout=True)
for axis, prediction, title in zip(
    axes,
    (prediction_naive, prediction_experts),
    ("hidden branch: incompatible roots averaged", "declared branches: two experts"),
):
    axis.scatter(y[test], prediction, c=branch[test], cmap="coolwarm", s=24, alpha=0.8)
    limits = (0.1, 3.1)
    axis.plot(limits, limits, "k--", lw=1)
    axis.set(xlim=limits, ylim=limits, xlabel="exact Mach", ylabel="predicted Mach", title=title)
    axis.grid(alpha=0.3)
plt.show()
"""),
        md(r"""
### Checkpoint 1

The two-expert model is better because the **problem statement became
single-valued**, not because its optimizer is intrinsically superior. Name one
other Week-8 relation with the same ambiguity and specify the missing branch label.
"""),
        md(r"""
## 2. What “physics-guided” means in this article

This is **not** a strict PDE-residual PINN. The manuscript inserts physical
structure before and after an ordinary supervised MLP:

| Device | Gas-dynamics example | Failure it blocks |
|---|---|---|
| branch decomposition | Rayleigh/Fanno inverse experts | averaging incompatible Mach roots |
| sonic coordinate + positive target | $(M-1)^2\exp[g(M)]$ for Fanno | negative friction length and broken sonic limit |
| physics-informed features | $(M,\log M,M^2)$ for Rayleigh | poor conditioning over a wide Mach range |
| exact output envelope | oblique-shock endpoint bounds | detached or misordered attached roots |
| hybrid analytical reconstruction | learn only the shock-tube implicit pressure step | smoothing wave discontinuities that exact relations can reconstruct |

The reported common contract uses Tanh activations, L-BFGS, $10^{-8}$ weight
penalty, $10^{-10}$ tolerance, at most 700 iterations and 50,000 function
evaluations. Hyperparameters are frozen before blind evaluation. Seeds 11, 29,
and 47 measure training variability; three seeds are not calibrated uncertainty.

**Checkpoint:** For one row, name the property guaranteed by construction and
one conservation, closure, or accuracy property that is still *not* guaranteed.
"""),
        md(r"""
## 3. Five retained blind tests

The research pipeline trains separate constrained models for Rayleigh, Fanno,
oblique-shock, nozzle-shock, and shock-tube inverse tasks. Architecture and
preprocessing are selected without using the blind cases. The following values
are retained evidence, not results generated by the small demonstration above.
"""),
        code(r"""
primary = evidence["primary"].copy()
primary["relative_L2_percent"] = 100.0 * primary["rel_l2"]
display(primary[["problem", "relative_L2_percent", "rel_linf", "mae"]])

plt.figure(figsize=(8.0, 4.6))
plt.barh(primary["problem"], primary["relative_L2_percent"], color="#2F75B5")
plt.axvline(0.35, color="#C44536", ls="--", label="declared 0.35% gate")
plt.xlabel("blind relative L2 error (%)")
plt.gca().invert_yaxis()
plt.grid(axis="x", alpha=0.3)
plt.legend(frameon=False)
plt.show()

assert primary["rel_l2"].max() < 3.5e-3
"""),
        md(r"""
## 4. The baseline can win - and coverage still matters

PCHIP-style interpolation is the strongest simple baseline for a covered 1-D
table. Compare both accuracy and coverage. A row with low error on only 65% of
the test points is not directly equivalent to a model that returns an admissible
answer for all points.
"""),
        code(r"""
baselines = evidence["baselines"].copy()
baselines["relative_L2_percent"] = 100.0 * baselines["rel_l2"]
display(baselines[["problem", "model", "coverage", "relative_L2_percent"]])

error_table = baselines.pivot(index="problem", columns="model", values="relative_L2_percent")
coverage_table = baselines.pivot(index="problem", columns="model", values="coverage")
axes = error_table.plot.barh(figsize=(9.0, 5.2), logx=True,
                             color=["#E9A23B", "#2A9D8F"])
axes.set_xlabel("relative L2 error (%) - log scale")
axes.grid(axis="x", alpha=0.3)
axes.legend(frameon=False)
plt.show()

print("Interpolation coverage by problem:")
display(coverage_table[["classical_interpolation"]])
"""),
        md(r"""
### Decision rule

- Prefer the exact formula when it is already cheap and available.
- Prefer interpolation for a covered low-dimensional table when it meets both
  accuracy and coverage requirements.
- Consider a bounded neural model when the query is implicit, branch identity
  is explicit, repeated root solves dominate, regular-grid storage grows
  combinatorially, or differentiable compact evaluation is needed.

Never choose the MLP merely because it is the most sophisticated method in the notebook.
"""),
        md(r"""
## 5. Ordinary blind accuracy is not edge generalization

The omitted-edge audit trains on interior parameter boxes and tests only the
excluded low/high bands within the declared domain. This is harder than a
random blind split and is still **not** unrestricted extrapolation.
"""),
        code(r"""
edges = evidence["range_generalization"].copy()
edges["edge_relative_L2_percent"] = 100.0 * edges["rel_l2"]
comparison = primary[["problem", "relative_L2_percent"]].merge(
    edges[["problem", "edge_relative_L2_percent", "valid_rate"]], on="problem"
)
display(comparison)

comparison.set_index("problem")[["relative_L2_percent", "edge_relative_L2_percent"]].plot.bar(
    figsize=(9.5, 4.8), color=["#2F75B5", "#C44536"]
)
plt.ylabel("relative L2 error (%)")
plt.xticks(rotation=18, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend(["random blind set", "omitted edge bands"], frameon=False)
plt.show()

assert (comparison["edge_relative_L2_percent"] > comparison["relative_L2_percent"]).all()
"""),
        md(r"""
## 6. Dimensional scaling under a matched offline budget

The generalized shock tube increases from two inputs
$(P_4/P_1,T_4/T_1)$ to five by adding $\gamma_1$, $\gamma_4$, and $R_4/R_1$.
Both interpolation and MLP receive roughly 4096 reference states. The regular
grid becomes sparse per axis as dimension grows; the MLP retains one compact
parameterization.
"""),
        code(r"""
dimensions = evidence["high_dimensional"].copy()
scattered = evidence["scattered_baseline"].copy()
dimensions["interpolation_percent"] = 100.0 * dimensions["interpolation_rel_l2"]
dimensions["mlp_percent"] = 100.0 * dimensions["mlp_rel_l2"]
scattered["rbf_percent"] = 100.0 * scattered["relative_l2"]
display(dimensions[["dimension", "grid_nodes_per_axis", "grid_training_states",
                    "mlp_training_states", "interpolation_percent", "mlp_percent"]])
display(scattered[["dimension", "training_states", "neighbors", "rbf_percent"]])

plt.figure(figsize=(7.8, 4.7))
plt.semilogy(dimensions["dimension"], dimensions["interpolation_percent"], "o-",
             color="#E9A23B", lw=2.3, label="regular-grid interpolation")
plt.semilogy(dimensions["dimension"], dimensions["mlp_percent"], "o-",
             color="#2A9D8F", lw=2.3, label="bounded MLP")
plt.semilogy(scattered["dimension"], scattered["rbf_percent"], "s--",
             color="#2F75B5", lw=2.0, label="scattered local RBF")
plt.xticks([2, 3, 4, 5])
plt.xlabel("number of physical inputs")
plt.ylabel("relative L2 error (%) - log scale")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.show()

five_d = dimensions.loc[dimensions["dimension"] == 5].iloc[0]
five_d_rbf = scattered.loc[scattered["dimension"] == 5].iloc[0]
assert five_d["mlp_rel_l2"] < five_d["interpolation_rel_l2"]
assert five_d["mlp_rel_l2"] < five_d_rbf["relative_l2"] < five_d["interpolation_rel_l2"]
"""),
        md(r"""
### Why the scattered baseline changes the conclusion

The regular grid has only five nodes per coordinate in five dimensions, so it
is an intentionally difficult storage geometry. The local thin-plate RBF uses
the same 4096 Latin-hypercube states as the MLP and reaches about 0.95% relative
$L_2$, compared with 4.88% for the regular grid and 0.177% for the bounded MLP.
The neural result therefore survives a fairer data-spread comparison, but the
margin is much smaller than the regular-grid-only plot suggests.
"""),
        md(r"""
## 7. Application-scale evidence and physical diagnostics

The retained application audit evaluates 100,000 generalized shock-tube states
in one single-thread process. It compares one MLP with one bracketed Brent solve
per state after warm-up. Separately, the nozzle audit checks the analytical
network Jacobian against centered finite differences.
"""),
        code(r"""
application = evidence["application"]
application_table = pd.DataFrame({
    "quantity": [
        "shock-tube states", "MLP relative L2 (%)", "speedup vs one Brent root/state",
        "nozzle inverse-design cases", "maximum Newton iterations",
        "network-vs-finite-difference gradient relative difference",
    ],
    "value": [
        application["shock_tube_queries"],
        100.0 * application["shock_tube_rel_l2"],
        application["shock_tube_speedup"],
        application["nozzle_cases"],
        application["nozzle_max_iterations"],
        application["nozzle_max_gradient_relative_difference"],
    ],
})
display(application_table)
assert application["shock_tube_queries"] == 100000
assert application["shock_tube_speedup"] > 10

physical = evidence["physical"].copy()
display(physical.pivot(index="problem", columns="diagnostic", values="value"))

validity = physical[physical["diagnostic"].str.endswith(("rate", "ordering"))]
assert (validity["value"] == 1.0).all()
"""),
        md(r"""
### Checkpoint 2: a hard constraint is not a complete validation

The transforms make selected outputs admissible, but they do not enforce every
conservation identity. Pair one validity rate above with its independent
residual or thermodynamic diagnostic. Explain why both must be reported.
"""),
        md(r"""
## 8. Write the claim at the correct level

Complete this table in your own words:

| Question | Evidence to cite | Forbidden shortcut |
|---|---|---|
| Is the inverse single-valued? | declared branch/bounds and closure | hide the branch and average roots |
| Is ML more accurate? | matched exact/interpolation baseline | compare against a weaker baseline |
| Does it generalize? | complete blind cases plus edge audit | call in-domain edge tests extrapolation |
| Is it physically valid? | bounds, entropy/closure residuals | report only global relative L2 |
| Is it faster? | same process, workload, warm-up, statistic | compare unmatched implementations |
| Does it replace CFD? | it does not; these are canonical relations | promote 1-D evidence to multidimensional CFD |

### Defensible conclusion

Within the declared perfect-gas domains, branch-aware bounded MLPs reproduce all
five retained inverse benchmarks with blind relative-L2 error below 0.35% and
full physical coverage. Interpolation remains preferable on many covered 1-D
tables. Under a matched roughly 4096-state budget, the neural advantage becomes
clear in the five-input shock-tube audit, while omitted-edge errors remain much
larger than ordinary blind errors. No result here validates reacting flow,
variable heat capacity, multidimensional shocks, or unrestricted extrapolation.
"""),
        md(r"""
## Optional full reproduction

The full research evidence is intentionally not retrained inside this short lab.
To reproduce it, clone the pinned `GasDynamicsSciML` source and follow its
README. The main command trains ensembles, interpolation baselines, ablations,
near-limit audits, and timing benchmarks:

```bash
git clone https://github.com/Ehsan-Roohi/GasDynamicsSciML.git
cd GasDynamicsSciML
git checkout 374431a1033138f56e2752bf8bbf9b75a454d80c
python -m pip install -e .
python -m unittest discover -s tests -v
gasdynbench --output results/revision
```

Record the source commit, Python version, thread limits, seeds, and newly
generated CSV hashes before comparing with the retained evidence.

## Exit ticket

Choose one of the five inverse tasks and submit:

1. input, output, branch/bound, and declared domain;
2. the exact or bracketed reference;
3. the matched interpolation baseline;
4. blind relative-L2 and one physical diagnostic;
5. edge-holdout behavior; and
6. a two-sentence decision: use exact, interpolation, or MLP, and why.
"""),
    ]


def notebook(cells, title: str):
    generated = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "flowmllab": {"week": 8, "title": title, "profile": "student"},
        },
    )
    for index, cell in enumerate(generated["cells"]):
        cell["id"] = f"w8-{index:03d}"
    return generated


def main() -> None:
    outputs = [
        (
            HERE / "W8_Lab1_Exact_Gas_Dynamics_Student.ipynb",
            notebook(lab1_cells(), "Exact gas dynamics before machine learning"),
        ),
        (
            HERE / "W8_Lab2_Gas_Dynamics_SciML_Evidence_Student.ipynb",
            notebook(lab2_cells(), "When should a gas-dynamics surrogate be neural?"),
        ),
    ]
    for target, generated in outputs:
        nbf.write(generated, target)
        print("Wrote", target)


if __name__ == "__main__":
    main()
