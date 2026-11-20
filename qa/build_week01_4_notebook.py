"""Author the standalone Week 1.4 notebook; execute separately before delivery."""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'flowmllab/high_order.py'
source = MODULE.read_text(encoding='utf-8')
tree = ast.parse(source)
functions = {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
cavity_source = (ROOT / 'flowmllab/high_order_cavity.py').read_text(encoding='utf-8')
functions.update({n.name:ast.get_source_segment(cavity_source,n) for n in ast.parse(cavity_source).body
                  if isinstance(n,ast.FunctionDef)})
from week01_4_cavity_cells import append_cavity_cells
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def algorithms(*names):
    code('\n\n\n'.join(functions[name] for name in names))


md(r"""
# Week 1.4 - Spectral and discontinuous Galerkin methods on CPU

**A numerical-methods extension after Week 1.2: SIMPLE, PISO and PIMPLE.**

[Open in Google Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb) · [Lecture PDF](../../lectures/week01_4_spectral_dg.pdf) · [Setup and learning route](README.md) · [Previous pressure-velocity lab](../week01_2/W1_2_Cavity_Pressure_Velocity.ipynb)

**Question:** how does the spatial approximation change accuracy, conservation and cost?
We implement the algorithms below, not merely call an external CFD package.
NumPy, SciPy, Matplotlib and pandas are sufficient; no GPU, MPI, training data or cluster is needed.

**Route:** sections 1-4 (Chebyshev/Poisson), 5-9 (DG/transport), 10 (actual cavity CFD), 11-12 (exercises and evidence).
Plan two teaching sessions plus independent exercises; this is a teaching allocation, not a measured study time.
Run All recomputes every displayed experiment. The output timings describe your computer.

By the end you should be able to:
1. Build a differentiation matrix and impose boundary values.
2. Explain global polynomial approximation versus cell-local polynomials.
3. Derive a conservative DG numerical flux and identify the finite-volume limit.
4. Separate spatial, temporal, quadrature and algebraic error.
5. Choose a method using field/profile errors and computational work.
6. Solve a steady lid-driven cavity on CPU and compare computed velocities with Ghia.

Sections 1-9 verify spatial/time operators on manufactured pressure and passive-scalar transport.
Section 10 then solves the actual incompressible cavity at Re=100 and 400 using Chebyshev collocation.
Keep these three physical problems and their validation criteria distinct.
""")

md(r"""
## 1. Setup and controlled experiments

Use Python 3.10-3.13. If imports are missing, run `%pip install numpy scipy matplotlib pandas`
in a separate cell, restart the kernel, and run from the top. Colab normally provides these packages.
The notebook contains its own solver definitions and needs no repository clone or downloaded fields.

All lengths below are nondimensional. The pressure problem has a unit square and prescribed
zero boundary pressure. The transport problem has unit period, speed $a=1$ and final time $t=0.35$.
There is no Reynolds number in these scalar verification problems.

Change controls in the named experiment cell, then rerun the cells below it.
Keep the exact solution, evaluation locations and error norm fixed when comparing methods.
""")
code(r'''
from pathlib import Path
from time import perf_counter
import os, sys, json, platform
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.polynomial.legendre import leggauss, legvander
from scipy.interpolate import BarycentricInterpolator, RegularGridInterpolator
from scipy.linalg import solve_sylvester, solve
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
from IPython.display import display

OUTPUT_DIR = Path(os.environ.get("FLOWMLLAB_W14_OUTPUT", "week01_4_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10,
                     "figure.dpi": 120, "savefig.dpi": 180, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "stix"})
COLORS = ["#0755a1", "#11855a", "#c73735", "#6a3b9a"]
FIGURES = []
def save_figure(fig, name):
    fig.savefig(OUTPUT_DIR / (name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / (name + ".pdf"), bbox_inches="tight", facecolor="white")
    FIGURES.append(name)
    plt.show()
    plt.close(fig)

RUN_STARTED = perf_counter()
print("CPU notebook | Python", platform.python_version(), "| NumPy", np.__version__, "| SciPy", scipy.__version__)
''')

md(r"""
## 2. From finite differences to Chebyshev collocation

With $N+1$ nodes on $[0,1]$, let $x_j=(1-\cos(j\pi/N))/2$, $j=0,\ldots,N$.
The nodes cluster near the walls. The polynomial $q_N(x)=\sum_j q_j\ell_j(x)$
passes through every sample; $\ell_j$ is the $j$th Lagrange cardinal polynomial.
Then $q'_N(x_i)=\sum_j D_{ij}q_j$ with $D_{ij}=\ell'_j(x_i)$.

For $i\ne j$, $D_{ij}=w_j/[w_i(x_i-x_j)]$, with $w_j=(-1)^j$ and endpoint
weights halved. Set $D_{ii}=-\sum_{j\ne i}D_{ij}$ so a constant differentiates to zero.
The matrix differentiates with respect to **x on [0,1]**, not the reference coordinate on [-1,1].

Compare two targets: $q_s=\sin(2\pi x)+0.2\cos(6\pi x)$ and $q_c=|x-0.37|$.
The second has a corner. Smoothness is a property of the solution, not the plotting grid.
""")
algorithms('chebyshev_grid', 'interpolate_chebyshev', 'interpolate_field')
code(r'''
# Experiment A controls: N is the global polynomial degree (N+1 nodes).
DEGREES = [8, 12, 16, 24, 32, 48]
N_SHOW = 16
fine_x = np.linspace(0, 1, 2001)
targets = {"Smooth": lambda x: np.sin(2*np.pi*x)+0.2*np.cos(6*np.pi*x),
           "Corner": lambda x: np.abs(x-0.37)}
rows_a = []
for n in DEGREES:
    xc, d = chebyshev_grid(n)
    assert np.max(np.abs(d @ xc**3 - 3*xc**2)) < 1e-9
    for name, fun in targets.items():
        error = interpolate_chebyshev(xc, fun(xc), fine_x) - fun(fine_x)
        rows_a.append({"target": name, "N": n, "nodes": n+1,
                       "max_error_2001_points": np.max(np.abs(error))})
table_a = pd.DataFrame(rows_a)
display(table_a.rename_axis("Table A: interpolation"))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
xc, _ = chebyshev_grid(N_SHOW)
axes[0].plot(fine_x, targets["Corner"](fine_x), "k-", lw=1.8, label="Exact corner")
axes[0].plot(fine_x, interpolate_chebyshev(xc, targets["Corner"](xc), fine_x),
             color=COLORS[0], ls="--", label=f"Chebyshev N={N_SHOW}")
axes[0].plot(xc, targets["Corner"](xc), "o", color=COLORS[2], ms=3, label="Samples")
axes[0].set(xlabel="x", ylabel="q(x)")
for color, (name, group) in zip(COLORS, table_a.groupby("target", sort=False)):
    axes[1].semilogy(group["N"], group["max_error_2001_points"], "o-", color=color, label=name)
axes[1].set(xlabel="Global polynomial degree N", ylabel="Maximum sampled absolute error")
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=2)
save_figure(fig, "interpolation_smoothness")
''')
md(r"""
**Read Figure A.** The smooth target can converge rapidly until floating-point accuracy becomes relevant.
The corner remains difficult even at larger $N$. A denser plotting grid does not add degrees of freedom.
The sampled maximum is a reproducible error diagnostic, not a proof of the continuous supremum.

**Checkpoint A:** Why does making $D\mathbf{1}=0$ matter? Explain why a small interpolation error
does not guarantee an equally small derivative error near a corner. Exercise 1 tests both claims.
""")

md(r"""
## 3. A two-dimensional Poisson problem with a known answer

Choose $p^*(x,y)=\sin(\pi x)\sin(2\pi y)$, so
$$\nabla^2p^*=f,\qquad f=-5\pi^2\sin(\pi x)\sin(2\pi y),\qquad p^*|_{\partial\Omega}=0.$$
Here $p^*$ is a nondimensional manufactured pressure; $f$ is a prescribed source and
$\Omega=[0,1]^2$. No density, lid speed or Reynolds number is inferred from this field.

Let $A=(D^2)_{I,I}$ for interior indices $I=1,\ldots,N-1$.
With array convention $P_{j,i}=p^*(x_i,y_j)$, solve
$$AP+PA^T=F.$$
This Sylvester equation avoids forming a dense $(N-1)^2$ by $(N-1)^2$ matrix.
Zero boundary values make the right-hand-side correction zero; **nonzero Dirichlet data
would require subtracting the known boundary contributions before solving**.

The comparison uses a second-order central finite-difference Laplacian on an equispaced
grid with the same number of interior unknowns. Its Kronecker-sum matrix is sparse.
Both solve the same PDE and boundary conditions; neither uses a pressure-velocity loop here.
""")
algorithms('exact_pressure', 'poisson_square')
code(r'''
# Experiment P controls. Each pair has (N-1)^2 interior unknowns.
POISSON_N = [8, 12, 16, 24, 32]
TIMING_REPEATS = 3
pressure_solutions = {}
rows_p = []
for method in ("chebyshev", "finite_difference"):
    poisson_square(8, method)  # unreported warm-up
    for n in POISSON_N:
        repetitions = [poisson_square(n, method) for _ in range(TIMING_REPEATS)]
        sol = repetitions[-1]
        pressure_solutions[method, n] = sol
        rows_p.append({"method": method, "N": n, "unknowns": sol["unknowns"],
                       "relative_L2": sol["relative_l2"], "max_error": sol["max_error"],
                       "equation_residual_max": sol["residual_max"],
                       "assembly_solve_median_s": np.median([s["seconds"] for s in repetitions]),
                       "assembly_solve_min_s": min(s["seconds"] for s in repetitions),
                       "assembly_solve_max_s": max(s["seconds"] for s in repetitions)})
table_p = pd.DataFrame(rows_p)
display(table_p.rename_axis("Table P: Poisson, common 151x151 evaluation grid"))
''')
md(r"""
**Error contract.** Table P uses $\|p_h-p^*\|_2/\|p^*\|_2$ on the same uniform
$151\times151$ evaluation grid. It evaluates the Chebyshev polynomial barycentrically and
the finite-difference nodal field bilinearly. Thus it includes reconstruction error as well as
nodal error. The discrete equation residual is measured separately at the solver's interior nodes.
Small residual means the discrete equations were solved; it does not make the discretization exact.

Timings include assembly and the linear solve; they exclude plotting and evaluation-grid interpolation.
BLAS threading, process warm-up and very short timings affect the ranking. Compare repeats and
work/accuracy rather than treating a single millisecond measurement as a hardware benchmark.
""")
code(r'''
# Same levels and physical extent; no per-method colour rescaling.
N_FIELD = 12
query = np.linspace(0, 1, 151)
yy, xx = np.meshgrid(query, query, indexing="ij")
reference = exact_pressure(xx, yy)
levels = [-0.8, -0.4, 0.0, 0.4, 0.8]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
filled = axes[0].contourf(xx, yy, reference, levels=np.linspace(-1, 1, 25), cmap="turbo")
fig.colorbar(filled, ax=axes[0], label=r"Manufactured $p^*$", shrink=0.86)
for method, color, style in [("chebyshev", "#b012a5", "solid"), ("finite_difference", "#111111", "dashed")]:
    sol = pressure_solutions.get((method, N_FIELD)) or poisson_square(N_FIELD, method)
    evaluated = interpolate_field(sol["x"], sol["p"], query, method)
    axes[0].contour(xx, yy, evaluated, levels=levels, colors=color, linestyles=style, linewidths=1.1)
    axes[1].plot(query, evaluated[:, 75], color=COLORS[0] if method=="chebyshev" else COLORS[2],
                 ls="-" if method=="chebyshev" else "--", label=method.replace("_", " ").capitalize())
axes[0].set(xlabel="x", ylabel="y", aspect="equal")
handles = [Patch(facecolor=plt.get_cmap("turbo")(0.65), label="Analytic"),
           Line2D([], [], color="#b012a5", label="Chebyshev"),
           Line2D([], [], color="k", ls="--", label="Finite difference")]
axes[0].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
axes[1].plot(query[::6], reference[::6, 75], "ko", ms=3, fillstyle="none", label="Analytic")
axes[1].set(xlabel="y at x = 0.5", ylabel=r"$p^*$")
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
axes[1].grid(alpha=0.18)
save_figure(fig, "pressure_poisson_fields")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
for color, (name, group) in zip(COLORS, table_p.groupby("method", sort=False)):
    label = name.replace("_", " ")
    axes[0].loglog(group.unknowns, group.relative_L2, "o-", color=color, label=label)
    axes[1].loglog(group.assembly_solve_median_s, group.relative_L2, "o-", color=color, label=label)
axes[0].set(xlabel="Interior unknowns", ylabel="Relative L2 field error")
axes[1].set(xlabel="Median assembly + solve time (s)", ylabel="Relative L2 field error")
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
save_figure(fig, "poisson_accuracy_cost")
''')

md(r"""
## 4. Algorithm trace and boundary-condition check

**Chebyshev Poisson algorithm:** choose $N$ → construct $(x,D)$ → form $D^2$ →
restrict to interior rows/columns → assemble $F$ from the physical source →
solve $AP+PA^T=F$ → insert prescribed boundary values → evaluate residual and off-grid error.

For a pressure correction $p'$ in an incompressible solver, a simplified projection has
$\nabla^2p'=(\rho/\Delta t)\nabla\cdot\mathbf{u}^{\star}$ and
$\mathbf{u}^{new}=\mathbf{u}^{\star}-(\Delta t/\rho)\nabla p'$.
Here $\mathbf{u}^{\star}$ is the predicted velocity, $\rho$ density and $\Delta t$ the time step.
The actual boundary conditions and discrete gradient/divergence must be derived together
from the wall and face-flux equations. SIMPLE can instead use a variable momentum-based coefficient.
Replacing a pressure matrix by this standalone **Dirichlet** example is not a complete cavity implementation.

**Checkpoint P:** compare a coarse FD row with a Chebyshev row. Can both have a tiny
equation residual but different field errors? At a fixed error tolerance, which row would you choose?
""")

md(r"""
## 5. Discontinuous Galerkin: local polynomials communicate through fluxes

Solve $q_t+a q_x=0$ on $0\leq x<1$ with periodic boundaries.
The exact solution is $q(x,t)=q_0((x-at)\bmod1)$.
Divide the interval into $K$ cells of width $h=1/K$. On cell $e$, use
$$q_h(x,t)=\sum_{n=0}^{p}c_{e,n}(t)P_n(\xi),\qquad \xi=2(x-x_{e,L})/h-1.$$
$P_n$ is the degree-$n$ Legendre polynomial; $p$ is polynomial degree, not pressure.
There are $K(p+1)$ unknown coefficients. Different cells have distinct endpoint traces.

Multiply by $P_m$, integrate by parts within a cell and replace the interface flux by
$$\widehat f=\frac a2(q^-+q^+)-\frac{\alpha|a|}{2}(q^+-q^-).$$
$q^-$ and $q^+$ are the left and right traces at the same interface.
$\alpha=0$ gives a central flux; $\alpha=1$ gives upwind flux for either sign of $a$.

Legendre orthogonality makes the cell mass matrix diagonal: $M_{mm}=h/(2m+1)$.
The coefficient update is
$$\dot c_{e,m}=\frac{2m+1}{h}\left[a\sum_n S_{mn}c_{e,n}-\widehat f_R+(-1)^m\widehat f_L\right],$$
where $S_{mn}=\int_{-1}^{1}P'_mP_n\,d\xi$ equals 2 when $m>n$ and $m-n$ is odd, and zero otherwise.
The same interface flux leaves one cell and enters its neighbour: this is the conservation mechanism.

**Connection to Week 1.2:** at $p=0$, the volume term vanishes and
$\dot c_{e,0}=-(\widehat f_R-\widehat f_L)/h$. This is the conservative finite-volume
update for a cell average. DG extends that idea with more coefficients inside each cell.
""")
algorithms('initial_wave', 'dg_operators', 'dg_project', 'dg_rhs', 'dg_values', 'dg_diagnostics')
code(r'''
# Schematic only: global nodes versus independently represented DG cells.
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), layout="constrained")
nodes, _ = chebyshev_grid(12)
for yrow, nodeset, label, color in [(1.0, np.linspace(0,1,13), "Uniform nodes", COLORS[0]),
                                   (0.0, nodes, "Chebyshev nodes", COLORS[2])]:
    axes[0].plot(nodeset, np.full_like(nodeset,yrow), "o-", color=color, ms=4)
axes[0].set(xlim=(-0.03,1.03), ylim=(-0.5,1.5), xlabel="x", yticks=[0,1],
            yticklabels=["Chebyshev", "Uniform"])
for e in range(3):
    xi = np.linspace(-1,1,60)
    local_x = (e+(xi+1)/2)/3
    values = (0.38+0.12*e) + (0.12-0.13*e)*xi + 0.12*(xi**2-1/3)
    axes[1].plot(local_x, values, color=COLORS[e], lw=1.8)
    axes[1].plot([local_x[0],local_x[-1]], [values[0],values[-1]], "o", color=COLORS[e], ms=4)
    axes[1].text((e+0.5)/3, 0.92, f"Cell {e+1}", ha="center", fontsize=10)
for xface in [1/3,2/3]:
    axes[1].axvline(xface, color="0.5", ls=":", lw=0.9)
axes[1].annotate("Shared flux", xy=(1/3,0.54), xytext=(0.16,0.77),
                 arrowprops={"arrowstyle":"->", "color":"black"}, fontsize=10)
axes[1].set(xlim=(0,1), ylim=(0.1,1.05), xlabel="x", ylabel="Local polynomial (schematic)")
save_figure(fig, "spatial_representation_schematic")
''')

md(r"""
## 6. Projection, time stepping and measurable invariants

Initialize coefficients by the $L^2$ projection
$c_{e,m}(0)=(2m+1)h^{-1}\int_{I_e}q_0(x)P_m(\xi(x))\,dx$.
Gauss quadrature evaluates smooth integrals; the step target is split at its discontinuities
before integration. Sampling only the cell centre would not initialize higher DG modes correctly.

The spatial operator is $L(c)$. SSPRK3 advances it using
$$c^{(1)}=c^n+\Delta tL(c^n),\qquad
c^{(2)}=\tfrac34c^n+\tfrac14[c^{(1)}+\Delta tL(c^{(1)})],$$
$$c^{n+1}=\tfrac13c^n+\tfrac23[c^{(2)}+\Delta tL(c^{(2)})].$$
We choose $\Delta t\leq C h/[|a|(2p+1)]$, shorten it to land exactly on the final time,
and use $C=0.12$ by default. This is a conservative tested lab setting, **not a universal CFL bound**.

Track mass $\mathcal M=h\sum_e c_{e,0}$ and energy
$E=\frac12\sum_{e,m}h c_{e,m}^2/(2m+1)$.
For this periodic linear problem the semidiscrete identity is
$dE/dt=-\frac12\alpha|a|\sum_{interfaces}(q^+-q^-)^2$.
Central flux has no semidiscrete energy dissipation, but time integration still changes energy.
Upwind dissipation suppresses unresolved modes. Neither flux makes an unlimited high-order
solution monotone at a discontinuity; SSPRK3 alone is not a shock limiter.
""")
algorithms('solve_dg')
code(r'''
# Experiment D controls. Equal 64 coefficients, different h/p allocations.
DG_CASES = [(64, 0), (32, 1), (16, 3), (8, 7)]
TARGET = "localized"  # "smooth", "localized", or "step"
SPEED = 1.0
FINAL_TIME = 0.35
CFL = 0.12
ALPHA = 1.0
dg_solutions = [solve_dg(k, p, SPEED, ALPHA, TARGET, FINAL_TIME, CFL) for k, p in DG_CASES]
def dg_row(sol):
    return {"K":sol["ops"]["elements"], "p":sol["ops"]["degree"],
            "alpha":sol["ops"]["dissipation"], "dofs":sol["dofs"],
            "dt":sol["dt"], "rhs_calls":sol["rhs_calls"], "seconds":sol["seconds"],
            "relative_L2":sol["relative_l2"], "max_error":sol["max_error"], "mass_drift":sol["mass_drift"]}
table_d = pd.DataFrame([dg_row(s) for s in dg_solutions])
display(table_d.rename_axis("Table D: same number of coefficients"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
axes[0].plot(dg_solutions[0]["x"], dg_solutions[0]["exact"], "k-", lw=2, label="Exact transport")
for color, sol in zip(COLORS, dg_solutions):
    label = f"K={sol['ops']['elements']}, p={sol['ops']['degree']}"
    axes[0].plot(sol["x"], sol["predicted"], color=color, ls="--", label=label)
    axes[1].plot(sol["x"], sol["predicted"]-sol["exact"], color=color, label=label)
axes[0].set(xlabel="x", ylabel="q(x,t)")
axes[1].set(xlabel="x", ylabel="Signed error: numerical - exact")
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=9)
save_figure(fig, "dg_transport_equal_dofs")
''')

md(r"""
## 7. A discontinuity exposes the flux tradeoff

Use a square pulse equal to one on $[0.25,0.55)$ and zero elsewhere, transported by the same PDE.
The exact mass is 0.3. The field is not a gas-dynamic shock: there is no nonlinear wave speed,
equation of state or Mach number. It isolates numerical transport near a sharp front.

Both calculations below use the same $K$, $p$, time step and projected initial condition.
Do not clip negative values or overshoots: their amplitude is part of the result.
""")
code(r'''
# Experiment F controls. A change to the target is local to this comparison.
FLUX_K, FLUX_P = 24, 3
flux_solutions = [solve_dg(FLUX_K, FLUX_P, speed=SPEED, dissipation=alpha,
                          kind="step", final_time=FINAL_TIME, cfl=CFL) for alpha in (0.0, 1.0)]
table_f = pd.DataFrame([{**dg_row(s), "minimum":s["predicted"].min(), "maximum":s["predicted"].max()}
                        for s in flux_solutions])
display(table_f.rename_axis("Table F: flux and extrema"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
axes[0].plot(flux_solutions[0]["x"], flux_solutions[0]["exact"], "k-", lw=1.6, label="Exact pulse")
for color, label, sol in zip(COLORS, ["Central", "Upwind"], flux_solutions):
    axes[0].plot(sol["x"], sol["predicted"], color=color, label=label)
    history = sol["history"]
    axes[1].plot(history[:,0], history[:,2]/history[0,2], color=color, label=label)
axes[0].set(xlabel="x", ylabel="q(x,t)")
axes[1].set(xlabel="Time", ylabel="Energy / initial projected energy")
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, ncol=3)
save_figure(fig, "dg_flux_pulse_energy")
''')

md(r"""
## 8. Separate time error from spatial error

Keep the DG space and smooth target fixed. Compare $C$, $C/2$ and $C/4$.
The exact final time is unchanged. An observed factor near 8 between successive
coefficient differences is consistent with third-order time stepping when temporal error dominates.
Total error against the exact PDE can plateau because spatial error remains.
Integer step counts make the actual step ratios only approximately two; Table T reports the actual
values. Interpret the observed order as an approximate diagnostic of this nominal CFL-halving study.
""")
code(r'''
# Experiment T controls. The fixed space is deliberately accurate for the smooth wave.
TIME_K, TIME_P = 8, 5
time_solutions = [solve_dg(TIME_K, TIME_P, kind="smooth", final_time=0.35, cfl=c)
                  for c in (0.12, 0.06, 0.03)]
table_t = pd.DataFrame([{**dg_row(s), "CFL":s["cfl"]} for s in time_solutions])
display(table_t.rename_axis("Table T: time refinement"))
gap_1 = np.linalg.norm(time_solutions[0]["coefficients"]-time_solutions[1]["coefficients"])
gap_2 = np.linalg.norm(time_solutions[1]["coefficients"]-time_solutions[2]["coefficients"])
time_ratio = gap_1/gap_2
print("Successive coefficient-difference ratio:", time_ratio)
print("Observed temporal order from differences:", np.log2(time_ratio))
''')

md(r"""
## 9. h-refinement, p-refinement and an honest accuracy decision

$h$-refinement increases the number of cells $K$; $p$-refinement increases polynomial degree
inside a fixed cell partition. Equal unknowns do not imply equal time steps or work.
Table H changes $K$ at fixed degree; Table D held coefficient count fixed.
These are different experimental questions.

For the smooth periodic target, measure error at 4096 fixed midpoint locations in $[0,1)$.
Relative $L^2$ means the sampled Euclidean error norm divided by the exact sampled norm.
This dense evaluation is distinct from the Gauss quadrature used for coefficient projection.
""")
code(r'''
# Experiment H controls. Use a smaller CFL to keep temporal error below the spatial trend.
h_solutions = [solve_dg(k, 2, kind="smooth", final_time=0.35, cfl=0.06) for k in (8, 16, 32)]
table_h = pd.DataFrame([dg_row(s) for s in h_solutions])
table_h["observed_order"] = [np.nan] + [np.log2(h_solutions[i-1]["relative_l2"]/h_solutions[i]["relative_l2"])
                                           for i in range(1, len(h_solutions))]
display(table_h.rename_axis("Table H: fixed degree p=2"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
axes[0].loglog(table_h["dofs"], table_h["relative_L2"], "o-", color=COLORS[0], label="p=2, C=0.06")
axes[0].set(xlabel="Coefficients", ylabel="Relative L2 transport error")
for color, sol in zip(COLORS, time_solutions):
    axes[1].plot(sol["x"], sol["predicted"]-sol["exact"], color=color, label=f"C={sol['cfl']}")
axes[1].set(xlabel="x", ylabel="Signed smooth-wave error")
for ax in axes:
    ax.grid(alpha=0.18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
save_figure(fig, "dg_refinement_time")
''')

append_cavity_cells(md, code, algorithms, ROOT)

md(r"""
## 11. Student investigations - submit evidence, not screenshots alone

Preserve the default tables. Create a copy of each table for your changed setting.
For every exercise write **prediction → controlled change → measured result → explanation**.
Use line/profile plots and a compact table. Explain failures; do not delete them.

1. **Interpolation and differentiation (Table A).** Add $q(x)=x^5$ and its exact derivative $5x^4$.
   Test N=4, 6, 12; use `d @ q(xc)` at nodes. Then differentiate the corner target away from
   x=0.37. Submit nodal derivative errors and explain polynomial exactness versus nonsmoothness.
2. **Choose an elliptic solver (Table P).** Extend FD to N=48 and 64 and Chebyshev to N=20.
   Find the smallest tested unknown count with relative field error below $10^{-3}$.
   Report residual, error, median/min/max time and memory implications; do not infer a general speedup.
3. **h versus p (Tables D and H).** Repeat the equal-64-coefficient comparison for both `smooth`
   and `localized`. Then repeat at 128 coefficients with K=128,64,32,16 and p=0,1,3,7.
   Compare signed errors, RHS calls and measured time. Explain why the best allocation can change.
4. **Flux and sharp fronts (Table F).** Add alpha=0.5. Keep K=24, p=3, C=0.12, t=0.35 fixed.
   Report minimum, maximum, mass drift and energy change. Show full profiles; do not clip overshoot.
   Explain why the least-dissipative option need not give the best field.
5. **Temporal qualification (Table T).** Repeat at K=16, p=5; compare C=0.12,0.06,0.03.
   Report the successive-difference ratio and total exact-solution errors. Distinguish a spatial plateau
   from failed temporal convergence. Change the final time only in a separate experiment.
6. **Conservation and direction.** Run the smooth target with a=-1. Verify the translation direction.
   Use `dg_diagnostics` to check mass; compare the same exact final-time field. Explain the sign in the flux.
7. **Compute and refine the cavity (Tables C and G).** At Re=100 add N=12 and 20 to N=16,24,32.
   Plot computed centreline velocities over Ghia samples, pressure at y=0.5 and signed vorticity
   at x=0.5. Compare successive fields at the same locations and the same pressure gauge.
   Report the all-sample errors and both continuity diagnostics. Identify which quantities have converged.
8. **Change Reynolds number, then the lid.** Compare standard-lid Re=100 and 400 at matched N.
   Describe changes in recirculation and wall gradients using computed flow fields. Retain the flagged
   Re=400 reference point and both score versions. In a separate run use `CAVITY_LID="smooth"` at Re=100;
   compare N=12,20,28 and its off-grid continuity. Explain why Ghia scoring is disabled for this run.
9. **Algebraic convergence versus CFD accuracy.** At Re=400,N=24 use tolerances 1e-5,1e-7,1e-9.
   Compare Newton iterations, residual history, Ghia errors and off-grid continuity. Then set the iteration
   budget to one to inspect an explicit failed-convergence record. Explain why a steady Newton iteration
   is not a time step and why tighter tolerances cannot repair under-resolution.

**Submission:** the executed notebook, revised Tables A/P/D/F/T/H/C/G as relevant, three decisive figures
(including actual cavity fields and a CFD-versus-Ghia comparison),
and a 400-600 word method recommendation with actual settings and limitations.
Assess numerical correctness, controlled comparisons and physical interpretation, not whether a named method wins.
""")

md(r"""
## 12. Save evidence and further reading

The cell below saves the current numerical tables and a compact configuration record beside the figures.
Use Restart and Run All before submitting. Saved outputs document one execution; your own changed
controls must be rerun. No cached CFD field is substituted for a fresh computation.

1. L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM (2000).
   [Author's short programs and chapters](https://people.maths.ox.ac.uk/trefethen/spectral.html).
2. J. S. Hesthaven and T. Warburton, *Nodal Discontinuous Galerkin Methods*, Springer (2008).
   [Book DOI](https://doi.org/10.1007/978-0-387-72067-8). Our code uses modal Legendre coefficients;
   the weak-form and numerical-flux principles are shared with nodal DG.
3. deal.II, [Step 12: discontinuous Galerkin advection](https://dealii.org/9.7.0/doxygen/deal.II/step_12.html).
   This is an external 2-D adaptive-mesh extension, not a dependency of this notebook.
4. S. Gottlieb and C.-W. Shu, *Total variation diminishing Runge-Kutta schemes*, Math. Comp. 67 (1998), 73-85.
   [DOI](https://doi.org/10.1090/S0025-5718-98-00913-2).

The small algorithms here are independently written for FlowMLLab from the displayed equations.
5. Ghia, Ghia and Shin (1982), [original cavity benchmark](https://doi.org/10.1016/0021-9991(82)90058-4),
   Tables I and II. [Repository transcription and flagged-entry policy](../../references/ghia/README.md).
6. Botella and Peyret (1998), [Benchmark spectral results on the lid-driven cavity flow](https://doi.org/10.1016/S0045-7930(98)00002-4).
   Their high-accuracy singularity treatment is not implemented in this small teaching solver.
7. Botella and Peyret (2001), [Computing singular solutions with Chebyshev collocation](https://doi.org/10.1002/fld.121).
   Background on the mixed pressure/velocity spaces and corner singularities; our code uses a direct steady Newton solve.

Polynomial, Poisson, conservation and DG refinement checks are in `tests/test_high_order.py`.
`tests/test_high_order_cavity.py` independently tests a manufactured full Navier–Stokes solution,
the exact Jacobian, wall and pressure-gauge conditions, Ghia comparisons and failed-convergence reporting.
""")
code(r'''
tables = {"A":table_a, "P":table_p, "D":table_d, "F":table_f, "T":table_t, "H":table_h,
          "C":table_c, "G":table_g}
for label, table in tables.items():
    table.to_csv(OUTPUT_DIR / f"table_{label}.csv", index=False)
evidence = {"unit":"1.4", "problem_roles":["manufactured Dirichlet Poisson", "periodic linear scalar transport", "steady incompressible driven cavity"],
            "python":platform.python_version(), "numpy":np.__version__, "scipy":scipy.__version__,
            "platform":platform.platform(), "processor":platform.processor(),
            "controls":{"degrees":DEGREES,"poisson_n":POISSON_N,"timing_repeats":TIMING_REPEATS,
                        "dg_cases":DG_CASES,"target":TARGET,"speed":SPEED,"final_time":FINAL_TIME,
                        "cfl":CFL,"alpha":ALPHA,"flux_k":FLUX_K,"flux_p":FLUX_P,
                        "cavity_reynolds":CAVITY_REYNOLDS,"cavity_degrees":CAVITY_DEGREES,
                        "cavity_tolerance":CAVITY_TOLERANCE,"cavity_max_iterations":CAVITY_MAX_ITERATIONS,
                        "cavity_lid":CAVITY_LID},
            "ghia_source_sha256":GHIA_SOURCE_SHA256,"ghia_source_doi":GHIA_DOI,
            "time_difference_ratio":float(time_ratio),
            "elapsed_cells_seconds":perf_counter()-RUN_STARTED,
            "figures":FIGURES, "tables":{k:v.astype(object).where(pd.notna(v),None).to_dict(orient="records") for k,v in tables.items()}}
(OUTPUT_DIR / "execution.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
print("Saved", len(tables), "tables and", len(FIGURES), "figures to", OUTPUT_DIR)
print("Elapsed experiment/plotting time (s):", round(evidence["elapsed_cells_seconds"], 2))
''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'flowmllab':{'unit':'1.4', 'standalone':True, 'prerequisite':'Week 1.2',
                'algorithm_source':['flowmllab/high_order.py','flowmllab/high_order_cavity.py']}
})
for i, cell in enumerate(notebook.cells):
    cell.id = f'w14-{i:03d}'
path = ROOT / 'notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(notebook)
nbf.write(notebook, path)
print(path, len(cells), 'cells')
