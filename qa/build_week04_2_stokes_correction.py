#!/usr/bin/env python3
"""Build the executed Week 4.2 notebook, figures, and ten-page lecture PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import nbformat as nbf
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from execute_cavity_lesson import execute
from flowmllab.cavity_diversity import fields, lid_profile


RESULTS = ROOT / "results/stokes_refined"
NOTEBOOK = ROOT / "notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb"
PDF = ROOT / "lectures/week04_2_stokes_to_navier_stokes.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#246B9E")
TEAL = colors.HexColor("#21867A")
GOLD = colors.HexColor("#C28A2C")
RED = colors.HexColor("#A9473B")
LIGHT = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#D7E0E7")
TEXT = colors.HexColor("#202A33")


def make_figures() -> dict[str, Path]:
    comparison = pd.read_csv(RESULTS / "comparison.csv")
    out: dict[str, Path] = {}

    rows = comparison[comparison.test.isin(["constant", "diverse"])].copy()
    rows["pair"] = rows.train.str[0].str.upper() + " -> " + rows.test.str[0].str.upper()
    pivot = rows.pivot(index="pair", columns="model", values="velocity_percent")
    pivot = pivot.reindex(["C -> C", "C -> D", "D -> C", "D -> D"])
    fig, ax = plt.subplots(figsize=(9.4, 4.7), layout="constrained")
    x = np.arange(len(pivot)); width = 0.24
    models = ["Original POD", "Refined / same data", "Refined / 64 train"]
    palette = ["#AEBBC5", "#4F89B8", "#21867A"]
    for j, (model, color) in enumerate(zip(models, palette)):
        bars = ax.bar(x + (j - 1) * width, pivot[model], width, label=model, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("Mean velocity relative L2 error (%)")
    ax.set_title("Same retained test cases: effect of optimization and added training data")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    p = RESULTS / "week04_2_error_matrix.png"; fig.savefig(p, dpi=220); plt.close(fig); out["errors"] = p

    manifest = json.loads((ROOT / "results/cavity_diversity_pilot/manifest.json").read_text())
    cases = manifest["cases"]
    x = np.linspace(0, 1, 401)
    diverse = [c for c in cases if c["family"] == "diverse" and c["split"] == "train"][:6]
    fig, ax = plt.subplots(figsize=(9.4, 4.5), layout="constrained")
    ax.plot(x, np.ones_like(x), color="#17324D", lw=3, label="constant family")
    for i, case in enumerate(diverse):
        ax.plot(x, lid_profile(x, case["coefficients"]), lw=1.8, alpha=0.9,
                label="diverse examples" if i == 0 else None)
    ax.set(xlabel="x/L", ylabel="U_lid(x) / U_ref", ylim=(0.35, 1.05))
    ax.set_title("Boundary-condition families used for training and testing")
    ax.grid(alpha=0.22); ax.legend(frameon=False)
    p = RESULTS / "week04_2_lid_families.png"; fig.savefig(p, dpi=220); plt.close(fig); out["lids"] = p

    history = pd.read_csv(RESULTS / "expanded/history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), layout="constrained", sharey=False)
    for ax, family in zip(axes, ["constant", "diverse"]):
        h = history[(history.family == family) & (history.candidate == "24_96_0.02")]
        ax.semilogy(h.step, h.validation_velocity_rel_l2, color="#246B9E", lw=2)
        ax.axvline(1800, color="#A9473B", ls="--", lw=1.2, label="L-BFGS starts")
        ax.set(title=family.title() + "-lid model", xlabel="optimization step",
               ylabel="validation velocity relative L2")
        ax.grid(alpha=0.2); ax.legend(frameon=False)
    p = RESULTS / "week04_2_validation_history.png"; fig.savefig(p, dpi=220); plt.close(fig); out["history"] = p
    return out


def make_field_figures() -> None:
    """Render retained n=25 fields smoothly without changing the saved samples."""
    data = np.load(RESULTS / "velocity_pressure_fields.npz")
    cases = json.loads((ROOT / "results/cavity_diversity_pilot/manifest.json").read_text())["cases"]
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    sample_grid = np.linspace(0, 1, 25)
    display_grid = np.linspace(0, 1, 201)
    yy, xx = np.meshgrid(display_grid, display_grid, indexing="ij")
    display_points = np.column_stack((yy.ravel(), xx.ravel()))

    def display_field(field: np.ndarray) -> np.ndarray:
        # Linear interpolation is used for pixels only; errors and metrics stay
        # on the original 25-by-25 numerical grid.
        return RegularGridInterpolator((sample_grid, sample_grid), field,
                                       method="linear")(display_points).reshape(xx.shape)

    for family in ("constant", "diverse"):
        case = next(c for c in cases if c["family"] == family and c["split"] == "test")
        reference = {key: data[f"reference_{family}_{key}"][0] for key in ("u", "v", "p")}
        prediction = {key: data[f"{family}_{family}_{key}"][0] for key in ("u", "v", "p")}
        reference["speed"] = np.hypot(reference["u"], reference["v"])
        prediction["speed"] = np.hypot(prediction["u"], prediction["v"])

        fig, axes = plt.subplots(4, 3, figsize=(13, 15.6), layout="constrained")
        fig.suptitle(f"{family.title()} lid | first held-out case | Re = {case['Re']:.1f}\n"
                     "Left: NS reference | center: Stokes + neural correction | right: error",
                     fontsize=15)
        for row, (key, label) in enumerate((
            ("u", "u / U_ref"), ("v", "v / U_ref"),
            ("speed", "Speed / U_ref"), ("p", "Recovered p / (rho U_ref^2)"),
        )):
            ref, pred = reference[key], prediction[key]
            error = np.abs(pred - ref)  # Before display interpolation.
            lower = min(float(ref.min()), float(pred.min()))
            upper = max(float(ref.max()), float(pred.max()))
            for col, (values, title) in enumerate(((ref, label), (pred, "Prediction"),
                                                    (error, "Absolute error"))):
                ax = axes[row, col]
                im = ax.imshow(display_field(values), origin="lower", extent=(0, 1, 0, 1),
                               cmap="magma" if col == 2 else "coolwarm",
                               vmin=0 if col == 2 else lower,
                               vmax=float(error.max()) if col == 2 else upper,
                               interpolation="bilinear", aspect="equal")
                ax.set(title=title, xlabel="x/L", ylabel="y/L")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        fig.supxlabel("Display interpolation only; all model errors use the original 25 x 25 samples.",
                      fontsize=9, color="#425466")
        out = figures_dir / f"Cavity_{family}_velocity_pressure.png"
        fig.savefig(out, dpi=170)
        plt.close(fig)


def make_vortex_figures() -> None:
    """Compare speed, streamlines and grid-derived vorticity on retained cases."""
    data = np.load(RESULTS / "expanded/expanded.npz")
    predictions = np.load(RESULTS / "expanded/predictions.npz")
    cases = json.loads((RESULTS / "expanded/expanded_cases.json").read_text())
    n = data["psi"].shape[-1]
    h = 1 / (n - 1)
    grid = np.linspace(0, 1, n)
    display_grid = np.linspace(0, 1, 201)
    yy, xx = np.meshgrid(display_grid, display_grid, indexing="ij")
    display_points = np.column_stack((yy.ravel(), xx.ravel()))
    inner = grid[1:-1]
    inner_display = np.linspace(inner[0], inner[-1], 201)
    yi, xi = np.meshgrid(inner_display, inner_display, indexing="ij")
    inner_points = np.column_stack((yi.ravel(), xi.ravel()))
    corner = (slice(1, 10), slice(15, 24))  # Lower-right on the retained n=25 grid.

    def display(values: np.ndarray, interior: bool = False) -> np.ndarray:
        coordinates = (inner, inner) if interior else (grid, grid)
        points = inner_points if interior else display_points
        return RegularGridInterpolator(coordinates, values, method="linear")(
            points).reshape(201, 201)

    def vorticity(psi: np.ndarray) -> np.ndarray:
        # Match the interior finite-difference operator used by evaluate().
        return -(psi[1:-1, 2:] + psi[1:-1, :-2] + psi[2:, 1:-1]
                 + psi[:-2, 1:-1] - 4 * psi[1:-1, 1:-1]) / h**2

    def centers(psi: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
        primary = np.unravel_index(np.argmin(psi), psi.shape)
        local = np.unravel_index(np.argmax(psi[corner]), psi[corner].shape)
        secondary = (local[0] + corner[0].start, local[1] + corner[1].start)
        return primary, secondary

    rows = []
    for family in ("constant", "diverse"):
        ids = [i for i, case in enumerate(cases)
               if case["family"] == family and case["split"] == "test"]
        predicted = predictions[f"{family}_{family}_ensemble"]
        assert len(ids) == len(predicted) == 6
        for j, i in enumerate(ids):
            ref, pred = data["psi"][i], predicted[j]
            ref_primary, ref_secondary = centers(ref)
            pred_primary, pred_secondary = centers(pred)
            wr, wp = vorticity(ref), vorticity(pred)
            rows.append(dict(
                family=family, case=cases[i]["id"], Re=cases[i]["Re"],
                primary_ref_y=ref_primary[0], primary_ref_x=ref_primary[1],
                primary_pred_y=pred_primary[0], primary_pred_x=pred_primary[1],
                lower_right_ref_y=ref_secondary[0], lower_right_ref_x=ref_secondary[1],
                lower_right_pred_y=pred_secondary[0], lower_right_pred_x=pred_secondary[1],
                lower_right_ref_psi=ref[ref_secondary],
                lower_right_pred_psi=pred[pred_secondary],
                interior_vorticity_rel_l2=np.linalg.norm(wp - wr) / np.linalg.norm(wr),
            ))

        i = ids[0]
        ref, pred = data["psi"][i], predicted[0]
        lid = data["lid"][i]
        ur, vr = (component[0] for component in fields(ref[None], lid[None]))
        up, vp = (component[0] for component in fields(pred[None], lid[None]))
        speed_ref, speed_pred = np.hypot(ur, vr), np.hypot(up, vp)
        omega_ref, omega_pred = vorticity(ref), vorticity(pred)
        family_rows = [row for row in rows if row["family"] == family]
        primary_matches = sum((r["primary_ref_y"], r["primary_ref_x"]) ==
                              (r["primary_pred_y"], r["primary_pred_x"])
                              for r in family_rows)
        secondary_matches = sum((r["lower_right_ref_y"], r["lower_right_ref_x"]) ==
                                (r["lower_right_pred_y"], r["lower_right_pred_x"])
                                for r in family_rows)
        mean_vorticity_error = np.mean([r["interior_vorticity_rel_l2"]
                                        for r in family_rows])

        fig, axes = plt.subplots(2, 3, figsize=(14.4, 9.4), layout="constrained")
        fig.suptitle(f"{family.title()} lid | first retained test case | Re = {cases[i]['Re']:.1f}\n"
                     "Navier-Stokes reference vs Stokes-corrected prediction",
                     fontsize=16)
        for col, (psi, u, v, speed, label) in enumerate((
            (ref, ur, vr, speed_ref, "Reference"),
            (pred, up, vp, speed_pred, "Prediction"),
        )):
            ax = axes[0, col]
            im = ax.imshow(display(speed), origin="lower", extent=(0, 1, 0, 1),
                           cmap="viridis", vmin=0,
                           vmax=max(float(speed_ref.max()), float(speed_pred.max())),
                           interpolation="bilinear")
            ax.streamplot(grid, grid, u, v, density=1.25, color="white",
                          linewidth=0.65, arrowsize=0.65)
            primary, secondary = centers(psi)
            ax.scatter(primary[1] * h, primary[0] * h, s=90, marker="o",
                       c="#FFB547", edgecolors="black", linewidths=0.8, zorder=5)
            ax.scatter(secondary[1] * h, secondary[0] * h, s=95, marker="^",
                       c="#F66DCC", edgecolors="black", linewidths=0.8, zorder=5)
            ax.set(title=f"{label}: speed + streamlines")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax = axes[0, 2]
        im = ax.imshow(display(np.abs(speed_pred - speed_ref)), origin="lower",
                       extent=(0, 1, 0, 1), cmap="magma", vmin=0,
                       interpolation="bilinear")
        ax.set(title="Absolute speed error")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

        limit = max(float(np.abs(omega_ref).max()), float(np.abs(omega_pred).max()))
        omega_norm = SymLogNorm(linthresh=1, vmin=-limit, vmax=limit)
        for col, (omega, label) in enumerate(((omega_ref, "Reference"),
                                              (omega_pred, "Prediction"))):
            ax = axes[1, col]
            im = ax.imshow(display(omega, interior=True), origin="lower",
                           extent=(h, 1-h, h, 1-h), cmap="RdBu_r", norm=omega_norm,
                           interpolation="bilinear")
            ax.set(title=f"{label}: interior vorticity")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax = axes[1, 2]
        im = ax.imshow(display(np.abs(omega_pred - omega_ref), interior=True),
                       origin="lower", extent=(h, 1-h, h, 1-h), cmap="magma",
                       vmin=0, interpolation="bilinear")
        ax.set(title="Absolute interior vorticity error")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        for ax in axes.flat:
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x/L", ylabel="y/L")
            ax.set_aspect("equal")
        fig.supxlabel(
            f"Orange circle: primary vortex; pink triangle: lower-right recirculation. "
            f"Across 6 retained {family}-lid tests: centers match {primary_matches}/6 and "
            f"{secondary_matches}/6 grid nodes; mean interior vorticity error "
            f"{100 * mean_vorticity_error:.2f}%.", fontsize=10)
        fig.savefig(ROOT / "figures" / f"Cavity_{family}_streamlines_vorticity.png", dpi=180)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(RESULTS / "vortex_comparison.csv", index=False,
                              float_format="%.12g")


def build_notebook(figures: dict[str, Path]) -> None:
    colab = "https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/W4_Lab4_Stokes_to_Navier_Stokes.ipynb"
    cells = [
        nbf.v4.new_markdown_cell(f"""<a href=\"{colab}\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open in Colab\"/></a>

# Week 4 - Lab 4: Stokes-to-Navier-Stokes correction

<!-- MIE690A article-aligned validation v4 -->

This lab uses an actual Stokes solution and Reynolds number to predict the nonlinear correction required by the steady Navier-Stokes solution. It is a data-driven correction experiment, not weight transfer from a Stokes network."""),
        nbf.v4.new_markdown_cell("""## Learning goals

1. Separate a low-fidelity physical solution from a learned correction.
2. Keep Reynolds-number changes consistent with velocity changes.
3. Compare constant and spatially diverse lid boundary conditions.
4. Evaluate all four train/test family combinations and two out-of-distribution groups.
5. Distinguish same-grid surrogate accuracy from mesh-independent CFD validation."""),
        nbf.v4.new_code_cell("""# FLOWMLLAB_COLAB_BOOTSTRAP_V1
from pathlib import Path as _FlowMLLabPath
import os as _flowmllab_os
import subprocess as _flowmllab_subprocess
import sys as _flowmllab_sys

if \"google.colab\" in _flowmllab_sys.modules or _flowmllab_os.environ.get(\"COLAB_RELEASE_TAG\"):
    _flowmllab_root = _FlowMLLabPath(\"/content/FlowMLLab\")
    if not (_flowmllab_root / \".git\").is_dir():
        _flowmllab_subprocess.run(
            [\"git\", \"clone\", \"--depth\", \"1\",
             \"https://github.com/Ehsan-Roohi/FlowMLLab.git\", str(_flowmllab_root)],
            check=True,
        )
    _flowmllab_subprocess.run(
        [_flowmllab_sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", str(_flowmllab_root)],
        check=True,
    )
    _flowmllab_os.chdir(_flowmllab_root)
    if str(_flowmllab_root) not in _flowmllab_sys.path:
        _flowmllab_sys.path.insert(0, str(_flowmllab_root))
    print(\"FlowMLLab ready:\", _flowmllab_root)
else:
    print(\"Using the current local FlowMLLab checkout.\")"""),
        nbf.v4.new_code_cell("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

ROOT = Path.cwd()
if not (ROOT / 'results/stokes_refined').exists():
    raise RuntimeError('Run from the FlowMLLab repository root.')
R = ROOT / 'results/stokes_refined'
print({'repository_root': '.', 'result_directory': 'results/stokes_refined'})"""),
        nbf.v4.new_markdown_cell(r"""## 1. Reynolds number and boundary-condition families

The dimensional reference speed is not independent of Reynolds number:

$$Re=\frac{U_{ref}L}{\nu}, \qquad U_{ref}=\frac{Re\,\nu}{L},$$

with $L=1$ and $\nu=0.0025$ in this experiment. The constant family uses

$$U_{lid}(x)=U_{ref}.$$

The diverse family uses a positive, max-normalized Fourier profile

$$U_{lid}(x)=U_{ref}\,\frac{1+\sum_{k=1}^{3}a_k\sin(k\pi x/L)}{\max_{0\leq\xi\leq L}\left|1+\sum_{k=1}^{3}a_k\sin(k\pi \xi/L)\right|}.$$

The held-out shape group activates five modes with larger total amplitude. The Reynolds OOD group uses $450<Re<600$ instead of the training interval $100<Re<400$."""),
        nbf.v4.new_code_cell("""manifest = json.loads((ROOT/'results/cavity_diversity_pilot/manifest.json').read_text())
cases = manifest['cases']
pd.DataFrame(cases).groupby(['family','split']).size().rename('cases').to_frame()"""),
        nbf.v4.new_code_cell("""display(Image(filename=str(R/'week04_2_lid_families.png')))"""),
        nbf.v4.new_markdown_cell(r"""## 2. Physical baseline and correction target

For an incompressible velocity field $\mathbf{u}$, the steady equations are

$$\nabla\cdot\mathbf{u}=0, \qquad (\mathbf{u}\cdot\nabla)\mathbf{u}=-\nabla p+Re^{-1}\nabla^2\mathbf{u}.$$

The Stokes baseline deletes only the nonlinear convection term. With streamfunction $\psi$, $u=\partial_y\psi$ and $v=-\partial_x\psi$. The network predicts a streamfunction correction,

$$\psi_{NS}=\psi_S+\Delta\psi_\theta(\psi_S,Re),$$

so the velocity correction follows by differentiation. Pressure is recovered afterwards from the momentum gradients; pressure is not a network output."""),
        nbf.v4.new_code_cell("""audit = json.loads((ROOT/'results/stokes_correction/solver_audit.json').read_text())
audit"""),
        nbf.v4.new_markdown_cell("""## 3. Numerical data contract

- The Navier-Stokes labels use a streamfunction-vorticity finite-difference solver on an $n=25$ grid.
- Batched midpoint RK2 advances vorticity; a discrete sine transform solves the Poisson equation.
- Labels are accepted only after the steady PDE residual passes twice.
- The Stokes system uses the same grid, lid data, streamfunction convention, and wall-vorticity closure.
- Train, validation, test, shape-OOD, and Reynolds-OOD cases are separated by complete flow case."""),
        nbf.v4.new_code_cell("""status_same = json.loads((R/'same_data/status.json').read_text())
status_expanded = json.loads((R/'expanded/status.json').read_text())
display(pd.DataFrame([status_same, status_expanded], index=['same_data','expanded']))"""),
        nbf.v4.new_markdown_cell(r"""## 4. Network architecture

The low-fidelity Stokes snapshots are compressed by a training-only POD basis. A second training-only POD basis represents the target correction. The MLP receives

$$\mathbf{z}=[Re/400,\;\log(Re/400),\;\widehat{\mathbf{a}}_S]$$

and predicts scaled correction coefficients. Candidate ranks and widths are $(12,64)$, $(24,64)$, and $(24,96)$. The selected model uses correction rank 24, two 96-neuron tanh hidden layers, and a linear rank-24 output. Selection uses validation velocity error only. Seeds 7, 17, and 27 are averaged after selection."""),
        nbf.v4.new_code_cell("""selection = pd.read_csv(R/'expanded/selection.csv')
selection[['family','rank','width','omega_weight','seed','validation','chosen_step','seconds']].round(5)"""),
        nbf.v4.new_markdown_cell(r"""## 5. Derivative-aware objective

Training minimizes a quadratic Sobolev metric in coefficient space,

$$\mathcal{L}=\|\Delta\psi_\theta-\Delta\psi\|_2^2
+\lambda_u\|\nabla(\Delta\psi_\theta-\Delta\psi)\|_2^2
+\lambda_\omega\|\nabla^2(\Delta\psi_\theta-\Delta\psi)\|_2^2.$$

Adam runs for 1800 epochs with cosine learning-rate decay. L-BFGS starts from the validation-selected Adam checkpoint. Checkpoint choice never uses a test field."""),
        nbf.v4.new_code_cell("""display(Image(filename=str(R/'week04_2_validation_history.png')))"""),
        nbf.v4.new_markdown_cell("""## 6. Optional regeneration

The retained checkpoints and predictions make the default notebook fast and deterministic. Set the switch below only when you intend to repeat the full dataset/training workflow. Numerical values can vary slightly with BLAS and PyTorch versions."""),
        nbf.v4.new_code_cell("""RUN_TRAINING = False
if RUN_TRAINING:
    from flowmllab.stokes_refined import run
    run('same_data')
    run('expanded')
else:
    print('Using retained predictions; training was not rerun.')"""),
        nbf.v4.new_markdown_cell("""## 7. Velocity results on unchanged regression tests

The expanded stage adds training and validation cases but retains the original test cases. Therefore, the comparison isolates the effect of optimization and additional training data on the same inspected regression set; it is not a new blind benchmark."""),
        nbf.v4.new_code_cell("""comparison = pd.read_csv(R/'comparison.csv')
main = comparison[comparison.test.isin(['constant','diverse'])]
main.pivot(index=['train','test'], columns='model', values='velocity_percent').round(3)"""),
        nbf.v4.new_code_cell("""display(Image(filename=str(R/'week04_2_error_matrix.png')))"""),
        nbf.v4.new_markdown_cell("""## 8. Pressure recovery audit

Pressure has zero global mean. The reference pressure and predicted pressure are both recovered from their velocity fields using the same least-squares momentum-gradient method. Core errors remove two boundary layers and align the core gauge separately. This is a consistency comparison, not validation against an independently solved pressure field."""),
        nbf.v4.new_code_cell("""pressure = pd.read_csv(R/'pressure_metrics.csv')
pressure.groupby(['train','test'])[['pressure_full_percent','pressure_core_percent','gradient_residual']].mean().round(3)"""),
        nbf.v4.new_markdown_cell("""The following fields are linearly interpolated for display only. Numerical samples and all errors remain on the original 25 x 25 grid."""),
        nbf.v4.new_code_cell("""display(Image(filename=str(ROOT/'figures/Cavity_constant_velocity_pressure.png')))
display(Image(filename=str(ROOT/'figures/Cavity_diverse_velocity_pressure.png')))"""),
        nbf.v4.new_markdown_cell("""## 9. Streamlines and vortex capture

The speed maps below overlay streamlines from the original `n=25` velocity samples. Orange circles mark the primary streamfunction minimum; pink triangles mark the positive streamfunction maximum in the lower-right corner. Interior vorticity is computed with the same discrete `-Laplacian(psi)` operator for reference and prediction. Colors are interpolated for display only. The secondary corner feature is a grid-resolved recirculation candidate, not an independently validated vortex core."""),
        nbf.v4.new_code_cell("""vortices = pd.read_csv(R/'vortex_comparison.csv')
display(vortices.groupby('family')['interior_vorticity_rel_l2'].agg(['mean','max']).round(4))
display(Image(filename=str(ROOT/'figures/Cavity_constant_streamlines_vorticity.png')))
display(Image(filename=str(ROOT/'figures/Cavity_diverse_streamlines_vorticity.png')))"""),
        nbf.v4.new_markdown_cell("""## 10. Interpretation and claim boundary

The diverse-lid model generalizes much better across boundary families and OOD groups. Constant-only training remains poor on diverse lids because it never learns dependence on lid shape. The best retained mean errors are 0.107% for constant-to-constant and 0.917% for diverse-to-diverse velocity prediction.

These are same-grid surrogate errors against an educational finite-difference solver. The study does not establish mesh-independent CFD accuracy, a new neural architecture, or greater physical fidelity than the numerical labels. The useful result is narrower: an actual Stokes field is a strong low-fidelity coordinate for learning the nonlinear Navier-Stokes correction when the training boundary conditions span the intended use."""),
        nbf.v4.new_markdown_cell("""## 11. Numerical-grid refinement

The [51 × 51 validation companion](W4_Lab4_Grid51_Validation.ipynb) recomputes all 184 Navier–Stokes and matched Stokes cases, retrains the selected POD correction, and compares the numerical reference and surrogate errors against this 25 × 25 baseline. The [three-page PDF addendum](../../lectures/week04_2_grid51_validation.pdf) includes the speed, streamline and vorticity figures. The 25-to-51 change remains substantial, so neither grid establishes mesh-independent CFD accuracy."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "flowmllab": {"week": "4.2", "topic": "Stokes-to-Navier-Stokes correction"},
    })
    execute(notebook, ROOT)
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)


def _p(c: canvas.Canvas, text: str, x: float, y: float, width: float, style: ParagraphStyle) -> float:
    para = Paragraph(text, style)
    _, height = para.wrap(width, 1000)
    para.drawOn(c, x, y - height)
    return y - height


def build_pdf(figures: dict[str, Path]) -> None:
    comparison = pd.read_csv(RESULTS / "comparison.csv")
    pressure = pd.read_csv(RESULTS / "pressure_metrics.csv")
    selection = pd.read_csv(RESULTS / "expanded/selection.csv")
    vortex = pd.read_csv(RESULTS / "vortex_comparison.csv")
    width, height = A4
    c = canvas.Canvas(str(PDF), pagesize=A4, pageCompression=1)
    c.setTitle("FlowMLLab Week 4.2 - Stokes-to-Navier-Stokes Correction")

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.2, leading=13.3,
                          textColor=TEXT, alignment=TA_LEFT, spaceAfter=4)
    small = ParagraphStyle("small", parent=body, fontSize=8.7, leading=11)
    note = ParagraphStyle("note", parent=body, fontName="Times-Italic", fontSize=9.2,
                          leading=12, textColor=NAVY)
    equation = ParagraphStyle("equation", parent=body, fontName="Times-Italic", fontSize=11,
                              leading=15, alignment=TA_CENTER, borderColor=MID,
                              borderWidth=0.5, borderPadding=7, backColor=LIGHT)
    h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=19, leading=22, textColor=NAVY)
    h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=13.5, leading=17, textColor=NAVY)
    center = ParagraphStyle("center", parent=body, alignment=TA_CENTER)

    def header(page: int, title: str) -> float:
        c.setFillColor(NAVY); c.rect(0, height - 23*mm, width, 23*mm, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Times-Bold", 15)
        c.drawString(16*mm, height - 14*mm, title)
        c.setFont("Times-Roman", 8.5)
        c.drawRightString(width - 16*mm, height - 14*mm, f"FlowMLLab | Week 4.2 | {page}/10")
        c.setFillColor(TEXT)
        return height - 31*mm

    def footer() -> None:
        c.setStrokeColor(MID); c.line(16*mm, 14*mm, width - 16*mm, 14*mm)
        c.setFillColor(colors.HexColor("#5A6873")); c.setFont("Times-Roman", 7.5)
        c.drawString(16*mm, 9.5*mm, "Stokes input + Reynolds number -> learned streamfunction correction -> recovered velocity and pressure")

    def finish() -> None:
        footer(); c.showPage()

    # Page 1
    y = header(1, "Stokes-to-Navier-Stokes correction")
    y = _p(c, "<b>A physical low-fidelity field becomes the coordinate for a nonlinear correction.</b>",
           16*mm, y, width-32*mm, h1) - 4*mm
    y = _p(c, "The network receives an actual Stokes solution at the same geometry and lid boundary condition, plus Reynolds number. It predicts a correction to streamfunction; velocity follows by differentiation and pressure is recovered from momentum afterwards.", 16*mm, y, width-32*mm, body) - 3*mm
    c.drawImage(str(figures["errors"]), 18*mm, y-82*mm, width=width-36*mm, height=78*mm,
                preserveAspectRatio=True, anchor="c")
    y -= 88*mm
    metrics = [("0.107%", "constant -> constant"), ("0.917%", "diverse -> diverse"),
               ("2.044%", "diverse -> constant"), ("19.920%", "constant -> diverse")]
    boxw = (width-37*mm)/4
    for i, (value, label) in enumerate(metrics):
        x = 16*mm + i*(boxw+1.7*mm)
        c.setFillColor(LIGHT); c.roundRect(x, y-25*mm, boxw, 23*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(TEAL if i < 3 else RED); c.setFont("Times-Bold", 15); c.drawCentredString(x+boxw/2, y-10*mm, value)
        c.setFillColor(TEXT); c.setFont("Times-Roman", 7.8); c.drawCentredString(x+boxw/2, y-17*mm, label)
    y -= 31*mm
    y = _p(c, "<b>Evidence boundary.</b> These are mean velocity relative L2 errors on unchanged, previously inspected n=25 regression tests. They demonstrate surrogate improvement against same-grid numerical labels, not mesh-independent CFD accuracy or a new neural architecture.", 16*mm, y, width-32*mm, note) - 5*mm
    stages = [("Stokes", "matched low-fidelity field"), ("MLP", "POD correction coefficients"),
              ("Velocity", "derivatives of corrected psi"), ("Pressure", "momentum recovery")]
    sw = (width-43*mm)/4
    for i, (title, detail) in enumerate(stages):
        x = 16*mm + i*(sw+3.7*mm)
        c.setFillColor(NAVY if i in (0,2) else BLUE); c.roundRect(x, y-22*mm, sw, 20*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Times-Bold", 10); c.drawCentredString(x+sw/2, y-9*mm, title)
        c.setFont("Times-Roman", 6.7); c.drawCentredString(x+sw/2, y-15*mm, detail)
    finish()

    # Page 2
    y = header(2, "Physical formulation")
    y = _p(c, "1. Steady incompressible flow", 16*mm, y, width-32*mm, h2) - 2*mm
    y = _p(c, "div(u) = 0, &nbsp;&nbsp; (u . grad)u = -grad(p) + Re<super>-1</super> Laplacian(u)", 22*mm, y, width-44*mm, equation) - 4*mm
    y = _p(c, "With streamfunction psi, u = d psi/dy and v = -d psi/dx, so incompressibility is satisfied by construction. The Navier-Stokes labels retain convection; the Stokes baseline removes only convection while keeping the same cavity, grid, and lid data.", 16*mm, y, width-32*mm, body) - 3*mm
    y = _p(c, "2. Reynolds number changes with speed", 16*mm, y, width-32*mm, h2) - 2*mm
    y = _p(c, "Re = U<sub>ref</sub> L / nu, &nbsp;&nbsp; U<sub>ref</sub> = Re nu / L, &nbsp;&nbsp; L = 1, &nbsp;&nbsp; nu = 0.0025", 22*mm, y, width-44*mm, equation) - 4*mm
    y = _p(c, "Changing lid speed therefore changes Reynolds number. A dimensionless Stokes velocity field by itself does not identify the nonlinear target; Re is supplied explicitly.", 16*mm, y, width-32*mm, body) - 3*mm
    y = _p(c, "3. Learned correction", 16*mm, y, width-32*mm, h2) - 2*mm
    y = _p(c, "psi<sub>NS</sub> = psi<sub>S</sub> + Delta psi<sub>theta</sub>(psi<sub>S</sub>, Re)", 22*mm, y, width-44*mm, equation) - 4*mm
    y = _p(c, "The prediction target is the streamfunction difference. Velocity is obtained from derivatives of the corrected streamfunction. Pressure is recovered through a least-squares momentum-gradient solve and is never a network output.", 16*mm, y, width-32*mm, body) - 5*mm
    y = _p(c, "Why this is useful: Stokes supplies geometry and boundary-condition response exactly for the linear problem; learning is reserved for the Reynolds-dependent nonlinear departure.", 18*mm, y, width-36*mm, note) - 5*mm
    data = [["Quantity", "Stokes baseline", "Learned correction", "Final field"],
            ["Convection", "omitted", "learned implicitly", "represented"],
            ["Boundary data", "exact input", "not relearned", "matched"],
            ["Incompressibility", "streamfunction", "streamfunction", "by construction"],
            ["Pressure", "not used as target", "not predicted", "recovered afterwards"]]
    table = Table(data, colWidths=[32*mm, 41*mm, 45*mm, 42*mm], rowHeights=8*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",8.6), ("FONT",(0,0),(-1,0),"Times-Bold",8.6),
                               ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.35,MID), ("ALIGN",(1,1),(-1,-1),"CENTER"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 45*mm); table.drawOn(c, 16*mm, y-42*mm)
    finish()

    # Page 3
    y = header(3, "Dataset design: constant and diverse lids")
    y = _p(c, "The two training families use identical Reynolds-number distributions. This isolates boundary-condition diversity rather than confounding it with a different Re range.", 16*mm, y, width-32*mm, body) - 2*mm
    y = _p(c, "Constant: U<sub>lid</sub>(x) = U<sub>ref</sub><br/>Diverse: U<sub>lid</sub>(x) = U<sub>ref</sub>[1 + sum a<sub>k</sub> sin(k pi x/L)] / max |1 + sum a<sub>k</sub> sin(k pi xi/L)|", 19*mm, y, width-38*mm, equation) - 3*mm
    c.drawImage(str(figures["lids"]), 18*mm, y-76*mm, width=width-36*mm, height=72*mm,
                preserveAspectRatio=True, anchor="c")
    y -= 80*mm
    data = [["Group", "Train", "Validation", "Test", "Purpose"],
            ["constant", "64", "16", "6", "Uniform lid"],
            ["diverse", "64", "16", "6", "Three-mode positive lids"],
            ["shape OOD", "0", "0", "6", "Five modes, larger amplitude"],
            ["Re OOD", "0", "0", "6", "450 < Re < 600"]]
    table = Table(data, colWidths=[26*mm, 20*mm, 24*mm, 17*mm, 68*mm], rowHeights=8*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",9), ("FONT",(0,0),(-1,0),"Times-Bold",9),
                               ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.35,MID), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 50*mm); table.drawOn(c, 16*mm, y-40*mm)
    y -= 46*mm
    _p(c, "Original test cases are frozen across the old, same-data refined, and expanded comparisons. Added cases enter training and validation only.", 16*mm, y, width-32*mm, note)
    finish()

    # Page 4
    y = header(4, "Numerical solvers and data isolation")
    y = _p(c, "Navier-Stokes labels", 16*mm, y, 78*mm, h2)
    y2 = _p(c, "Stokes input", 113*mm, height-31*mm, 80*mm, h2)
    left = "Streamfunction-vorticity finite differences; midpoint RK2; discrete-sine-transform Poisson solve; Thom wall-vorticity closure; n=25; steady PDE residual checked twice before a field becomes a label."
    right = "One sparse block finite-difference system with convection removed; same n=25 grid, lid samples, streamfunction convention, and wall closure; LU factorization reused across cases."
    yl = _p(c, left, 16*mm, y-2*mm, 78*mm, body)
    yr = _p(c, right, 113*mm, y2-2*mm, 80*mm, body)
    y = min(yl, yr) - 6*mm
    boxes = [
        ("1", "Generate complete CFD cases", "No spatial patches are split across train and test."),
        ("2", "Solve matched Stokes fields", "No Navier-Stokes target enters the input solver."),
        ("3", "Fit POD bases on training only", "Validation selects rank/width; test remains outside selection."),
        ("4", "Average three selected seeds", "Seeds 7, 17, and 27; ensemble rule fixed before test reporting."),
    ]
    for number, title, detail in boxes:
        c.setFillColor(LIGHT); c.roundRect(18*mm, y-25*mm, width-36*mm, 22*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(BLUE); c.circle(28*mm, y-14*mm, 6*mm, fill=1, stroke=0)
        c.setFillColor(colors.white); c.setFont("Times-Bold", 11); c.drawCentredString(28*mm, y-16*mm, number)
        c.setFillColor(NAVY); c.setFont("Times-Bold", 10.5); c.drawString(39*mm, y-10*mm, title)
        c.setFillColor(TEXT); c.setFont("Times-Roman", 9); c.drawString(39*mm, y-17*mm, detail)
        y -= 27*mm
    y -= 1*mm
    audit = json.loads((ROOT / "results/stokes_correction/solver_audit.json").read_text())
    refine = audit["manufactured_refinement"]
    timing = audit["timing"]
    data = [["Retained numerical audit", "Observed value", "Interpretation"],
            ["Manufactured psi error, n=17/33/65", "/".join(f"{r['relative_psi_l2']:.4f}" for r in refine), "decreases by about 4x"],
            ["Observed refinement order", "/".join(f"{v:.3f}" for v in audit["observed_orders"]), "second order"],
            ["Linear relative residual", f"{audit['linear_relative_residual']:.2e}", "sparse solve passed"],
            ["One cached Stokes solve", f"{timing['Stokes_cached_solve_seconds']*1e3:.3f} ms", "single CPU case"],
            ["Unmodified Stokes velocity error", f"{100*audit['stokes_unmodified_velocity_error']:.2f}%", "correction is necessary"]]
    table = Table(data, colWidths=[59*mm, 48*mm, 55*mm], rowHeights=7.2*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",8.2), ("FONT",(0,0),(-1,0),"Times-Bold",8.4),
                               ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.3,MID), ("ALIGN",(1,1),(1,-1),"CENTER"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 46*mm); table.drawOn(c, 16*mm, y-44*mm)
    finish()

    # Page 5
    y = header(5, "POD correction network and optimization")
    y = _p(c, "Architecture", 16*mm, y, width-32*mm, h2) - 2*mm
    y = _p(c, "Training-only POD compresses the Stokes field. A second training-only POD represents the correction. The selected MLP receives [Re/400, log(Re/400), normalized Stokes POD coefficients] and uses two 96-neuron tanh hidden layers with a linear rank-24 output.", 16*mm, y, width-32*mm, body) - 3*mm
    y = _p(c, "Loss = field error + velocity-gradient error + weighted vorticity error", 23*mm, y, width-46*mm, equation) - 3*mm
    c.drawImage(str(figures["history"]), 18*mm, y-76*mm, width=width-36*mm, height=72*mm,
                preserveAspectRatio=True, anchor="c")
    y -= 80*mm
    selected = selection[(selection["rank"]==24)&(selection["width"]==96)][["family","seed","validation","chosen_step","seconds"]]
    data = [["Family", "Seed", "Validation L2", "Selected step", "Fit time (s)"]] + [
        [r.family, str(int(r.seed)), f"{r.validation:.5f}", str(int(r.chosen_step)), f"{r.seconds:.2f}"] for r in selected.itertuples()
    ]
    table = Table(data, colWidths=[36*mm, 24*mm, 35*mm, 34*mm, 34*mm], rowHeights=6.5*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",8.3), ("FONT",(0,0),(-1,0),"Times-Bold",8.3),
                               ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.3,MID), ("ALIGN",(1,1),(-1,-1),"CENTER"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 60*mm); table.drawOn(c, 16*mm, y-48*mm)
    y -= 53*mm
    y = _p(c, "Adam runs for 1800 epochs with cosine decay. L-BFGS starts from the validation-selected Adam checkpoint, not the final Adam iterate. Architecture and checkpoint selection do not inspect test errors.", 16*mm, y, width-32*mm, note) - 5*mm
    items = [("Input POD", "computed from training Stokes fields only"),
             ("Correction POD", "computed from training NS - Stokes differences"),
             ("Ensemble", "three independently fitted selected networks")]
    bw = (width-39*mm)/3
    for i, (title, detail) in enumerate(items):
        x = 16*mm + i*(bw+3.5*mm)
        c.setFillColor(LIGHT); c.roundRect(x, y-26*mm, bw, 24*mm, 2*mm, fill=1, stroke=0)
        c.setFillColor(NAVY); c.setFont("Times-Bold", 9.5); c.drawCentredString(x+bw/2, y-10*mm, title)
        c.setFillColor(TEXT); c.setFont("Times-Roman", 7.4); c.drawCentredString(x+bw/2, y-17*mm, detail)
    finish()

    # Page 6
    y = header(6, "Quantitative results and generalization")
    c.drawImage(str(figures["errors"]), 18*mm, y-82*mm, width=width-36*mm, height=78*mm,
                preserveAspectRatio=True, anchor="c")
    y -= 87*mm
    expanded = comparison[comparison.model == "Refined / 64 train"].pivot(index="train", columns="test", values="velocity_percent")
    data = [["Train family", "constant", "diverse", "shape OOD", "Re OOD"]]
    for train in ["constant", "diverse"]:
        data.append([train] + [f"{expanded.loc[train,t]:.3f}%" for t in ["constant","diverse","ood_shape","ood_re"]])
    table = Table(data, colWidths=[40*mm, 30*mm, 30*mm, 32*mm, 30*mm], rowHeights=9*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",9), ("FONT",(0,0),(-1,0),"Times-Bold",9),
                               ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.4,MID), ("ALIGN",(1,1),(-1,-1),"CENTER"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 35*mm); table.drawOn(c, 16*mm, y-27*mm)
    y -= 34*mm
    y = _p(c, "<b>Main scientific lesson.</b> Boundary-condition diversity matters more than simply reducing in-family error. The diverse model reaches 2.044% on constant lids, 0.917% on diverse lids, 8.428% on shape OOD, and 10.204% on Reynolds OOD. Constant-only training remains at 19.920% on diverse lids and above 36% on both OOD groups.", 16*mm, y, width-32*mm, body) - 3*mm
    pmean = pressure.groupby(["train","test"])[["pressure_full_percent","pressure_core_percent"]].mean()
    y = _p(c, f"Pressure is a secondary consistency check. For diverse-to-diverse cases, mean full/core recovered-pressure errors are {pmean.loc[('diverse','diverse'),'pressure_full_percent']:.2f}% and {pmean.loc[('diverse','diverse'),'pressure_core_percent']:.2f}%. Both reference and prediction use the same recovery operator.", 16*mm, y, width-32*mm, note) - 4*mm
    worst = comparison[comparison.model == "Refined / 64 train"].pivot(index="train", columns="test", values="worst_percent")
    data = [["Worst retained case", "constant", "diverse", "shape OOD", "Re OOD"]]
    for train in ["constant", "diverse"]:
        data.append([train] + [f"{worst.loc[train,t]:.2f}%" for t in ["constant","diverse","ood_shape","ood_re"]])
    table = Table(data, colWidths=[40*mm, 30*mm, 30*mm, 32*mm, 30*mm], rowHeights=8*mm)
    table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",8.3), ("FONT",(0,0),(-1,0),"Times-Bold",8.3),
                               ("BACKGROUND",(0,0),(-1,0),BLUE), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                               ("GRID",(0,0),(-1,-1),0.35,MID), ("ALIGN",(1,1),(-1,-1),"CENTER"),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
    table.wrapOn(c, width-32*mm, 28*mm); table.drawOn(c, 16*mm, y-24*mm)
    finish()

    # Page 7
    y = header(7, "Held-out constant-lid field")
    y = _p(c, "First test case in manifest order; never selected by prediction quality. Left: Navier-Stokes reference. Center: Stokes plus learned correction. Right: absolute error. Shared reference/prediction scales are used within each row. Linear interpolation smooths display only; metrics use the original n=25 grid.", 16*mm, y, width-32*mm, small) - 2*mm
    c.drawImage(str(ROOT / "figures/Cavity_constant_velocity_pressure.png"), 19*mm, 26*mm,
                width=width-38*mm, height=y-29*mm, preserveAspectRatio=True, anchor="c")
    finish()

    # Page 8
    y = header(8, "Held-out diverse-lid field and claim boundary")
    y = _p(c, "The diverse model is evaluated on the first diverse-lid test case in manifest order. The pressure row is a recovered field rather than a network output. Linear interpolation smooths display only; metrics use the original n=25 grid.", 16*mm, y, width-32*mm, small) - 2*mm
    c.drawImage(str(ROOT / "figures/Cavity_diverse_velocity_pressure.png"), 23*mm, 73*mm,
                width=width-46*mm, height=y-78*mm, preserveAspectRatio=True, anchor="c")
    y = 67*mm
    y = _p(c, "What is established", 16*mm, y, 80*mm, h2)
    _p(c, "Actual Stokes input; consistent Re-speed coupling; complete-case splits; validation-only selection; three-seed ensemble; strong in-family correction; useful transfer from diverse to constant lids.", 16*mm, y-2*mm, 80*mm, small)
    y2 = _p(c, "What is not established", 111*mm, 67*mm, 82*mm, h2)
    _p(c, "Mesh independence; high-fidelity CFD validation; independent pressure validation; a new network architecture; universal transfer from a constant lid to unseen boundary shapes.", 111*mm, y2-2*mm, 82*mm, small)
    finish()

    # Pages 9-10: speed/streamline and vorticity evidence for both lid families.
    for page, family in ((9, "constant"), (10, "diverse")):
        y = header(page, f"{family.title()}-lid vortices and streamlines")
        y = _p(c, "Speed colors and white streamlines compare the same retained test case. The orange circle is the primary streamfunction minimum; the pink triangle marks a positive lower-right streamfunction maximum. Vorticity is the negative discrete Laplacian of streamfunction in both fields, with a shared symmetric-log color scale.",
               16*mm, y, width-32*mm, body) - 3*mm
        c.drawImage(str(ROOT / "figures" / f"Cavity_{family}_streamlines_vorticity.png"),
                    16*mm, y-127*mm, width=width-32*mm, height=124*mm,
                    preserveAspectRatio=True, anchor="c")
        y -= 133*mm
        group = vortex[vortex.family == family]
        primary = int(((group.primary_ref_x == group.primary_pred_x) &
                       (group.primary_ref_y == group.primary_pred_y)).sum())
        corner = int(((group.lower_right_ref_x == group.lower_right_pred_x) &
                      (group.lower_right_ref_y == group.lower_right_pred_y)).sum())
        mean_omega = 100 * group.interior_vorticity_rel_l2.mean()
        worst_omega = 100 * group.interior_vorticity_rel_l2.max()
        values = [["Six retained tests", "Observed", "Meaning"],
                  ["Primary-vortex grid node", f"{primary}/6 matched", "same sampled center"],
                  ["Lower-right local maximum", f"{corner}/6 matched", "small recirculation candidate"],
                  ["Interior vorticity relative L2", f"{mean_omega:.2f}% mean; {worst_omega:.2f}% worst", "same-grid comparison"]]
        table = Table(values, colWidths=[59*mm, 47*mm, 58*mm], rowHeights=9*mm)
        table.setStyle(TableStyle([("FONT",(0,0),(-1,-1),"Times-Roman",8.4),
                                   ("FONT",(0,0),(-1,0),"Times-Bold",8.4),
                                   ("BACKGROUND",(0,0),(-1,0),NAVY),
                                   ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                                   ("GRID",(0,0),(-1,-1),0.35,MID),
                                   ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                   ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,LIGHT])]))
        table.wrapOn(c, width-32*mm, 45*mm)
        table.drawOn(c, 16*mm, y-39*mm)
        y -= 45*mm
        _p(c, "Display colors are linearly interpolated for legibility. Streamlines follow the retained n=25 velocity field; vortex positions and vorticity errors are measured on original grid nodes. Agreement here does not establish mesh-independent capture of the small corner vortex.",
           16*mm, y, width-32*mm, note)
        finish()
    c.save()


def main() -> None:
    figures = make_figures()
    make_field_figures()
    make_vortex_figures()
    build_notebook(figures)
    build_pdf(figures)
    print(NOTEBOOK.relative_to(ROOT).as_posix())
    print(PDF.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
