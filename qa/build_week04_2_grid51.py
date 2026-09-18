#!/usr/bin/env python3
"""Build the 51x51 Week 4.2 comparison figures, notebook, and PDF addendum."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from scipy.interpolate import RegularGridInterpolator
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flowmllab.cavity_diversity import fields

R = ROOT / "results/stokes_grid51"
F = ROOT / "figures"
PDF = ROOT / "lectures/week04_2_grid51_validation.pdf"
NB = ROOT / "notebooks/week04/W4_Lab4_Grid51_Validation.ipynb"


def center_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    n = ref.shape[-1]
    y, x = np.mgrid[:n, :n]
    corner = (x > .625 * (n - 1)) & (x < n - 1) & (y > 0) & (y < .375 * (n - 1))
    primary_ref = np.unravel_index(np.argmin(ref), ref.shape)
    primary_pred = np.unravel_index(np.argmin(pred), pred.shape)
    corner_ref = np.unravel_index(np.argmax(np.where(corner, ref, -np.inf)), ref.shape)
    corner_pred = np.unravel_index(np.argmax(np.where(corner, pred, -np.inf)), pred.shape)
    h = 1 / (n - 1)
    return dict(primary_ref_xy=(primary_ref[1]*h, primary_ref[0]*h),
                primary_pred_xy=(primary_pred[1]*h, primary_pred[0]*h),
                corner_ref_xy=(corner_ref[1]*h, corner_ref[0]*h),
                corner_pred_xy=(corner_pred[1]*h, corner_pred[0]*h),
                corner_ref_psi=float(ref[corner_ref]),
                corner_pred_psi=float(pred[corner_pred]))


def figures() -> tuple[pd.DataFrame, pd.DataFrame]:
    F.mkdir(exist_ok=True)
    data = np.load(R / "dataset.npz")
    old = np.load(ROOT / "results/stokes_refined/expanded/expanded.npz")
    cases = json.loads((R / "cases.json").read_text())
    metrics = pd.read_csv(R / "metrics.csv")
    old_metrics = pd.read_csv(ROOT / "results/stokes_refined/expanded/metrics.csv")
    old_metrics = old_metrics[old_metrics.seed == "ensemble"]
    grouped = metrics.groupby(["train_family", "test_family"]).velocity_rel_l2.mean() * 100
    baseline = old_metrics.groupby(["train_family", "test_family"]).velocity_rel_l2.mean() * 100
    rows = []
    for key in grouped.index:
        rows.append(dict(train=key[0], test=key[1], error_25_percent=baseline[key],
                         error_51_percent=grouped[key]))
    compare = pd.DataFrame(rows)
    compare.to_csv(R / "grid_comparison.csv", index=False, float_format="%.8g")
    fig, ax = plt.subplots(figsize=(9.2, 4.7), layout="constrained")
    ticks = np.arange(len(compare)); w = .35
    labels = [f"{r.train[0].upper()} -> {r.test[0].upper()}" for r in compare.itertuples()]
    ax.bar(ticks-w/2, compare.error_25_percent, w, label="25 x 25", color="#93A8BA")
    ax.bar(ticks+w/2, compare.error_51_percent, w, label="51 x 51", color="#21867A")
    ax.set_xticks(ticks, labels); ax.set_ylabel("Mean velocity relative L2 error (%)")
    ax.set_title("Same cases, separately solved and trained at each grid")
    ax.set_yscale("symlog", linthresh=1); ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(F / "Cavity_grid25_grid51_errors.png", dpi=180); plt.close(fig)

    vortex_rows = []
    for family in ("constant", "diverse"):
        ids = [i for i, c in enumerate(cases) if c["family"] == family and c["split"] == "test"]
        pred = np.load(R / f"prediction_{family}_{family}.npz")["psi"]
        for j, i in enumerate(ids):
            values = center_metrics(data["psi"][i], pred[j])
            row = dict(family=family, case=cases[i]["id"], Re=cases[i]["Re"],
                       **{k: str(v) if isinstance(v, tuple) else v for k, v in values.items()})
            row["primary_distance"] = float(np.linalg.norm(np.subtract(
                values["primary_ref_xy"], values["primary_pred_xy"])))
            row["corner_distance"] = float(np.linalg.norm(np.subtract(
                values["corner_ref_xy"], values["corner_pred_xy"])))
            vortex_rows.append(row)
        i = ids[0]; psi_ref = data["psi"][i]; psi_pred = pred[0]
        lid = data["lid"][i]; n = psi_ref.shape[-1]; h = 1 / (n - 1)
        ur, vr = [z[0] for z in fields(psi_ref[None], lid[None])]
        up, vp = [z[0] for z in fields(psi_pred[None], lid[None])]
        speed_ref, speed_pred = np.hypot(ur, vr), np.hypot(up, vp)
        lap = lambda p: -(p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1]
                          + p[:-2, 1:-1] - 4*p[1:-1, 1:-1]) / h**2
        om_ref, om_pred = lap(psi_ref), lap(psi_pred)
        limit = max(np.abs(om_ref).max(), np.abs(om_pred).max())
        xx = np.linspace(0, 1, n)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), layout="constrained")
        fig.suptitle(f"{family.title()} lid, Re={cases[i]['Re']:.1f}, numerical grid 51 x 51")
        for col, (speed, psi, title) in enumerate(((speed_ref, psi_ref, "NS reference"),
                                                   (speed_pred, psi_pred, "Corrected prediction"))):
            ax = axes[0, col]
            im = ax.contourf(xx, xx, speed, levels=40, cmap="viridis")
            ax.contour(xx, xx, psi, levels=24, colors="white", linewidths=.55)
            centers = center_metrics(psi, psi)
            ax.scatter(*centers["primary_ref_xy"], c="#FFB547", edgecolor="black", s=65)
            ax.scatter(*centers["corner_ref_xy"], c="#F66DCC", edgecolor="black", s=70, marker="^")
            ax.set_title(title + " | speed, streamlines", fontsize=11)
            fig.colorbar(im, ax=ax)
            ax = axes[1, col]
            im = ax.contourf(xx[1:-1], xx[1:-1], lap(psi), levels=45,
                             cmap="RdBu_r", norm=SymLogNorm(linthresh=1, vmin=-limit, vmax=limit))
            ax.set_title(title + " | vorticity", fontsize=11)
            fig.colorbar(im, ax=ax)
        im = axes[0, 2].contourf(xx, xx, np.abs(speed_pred-speed_ref), levels=35, cmap="magma")
        axes[0, 2].set_title("Absolute speed error"); fig.colorbar(im, ax=axes[0, 2])
        im = axes[1, 2].contourf(xx[1:-1], xx[1:-1], np.abs(om_pred-om_ref),
                                 levels=35, cmap="magma")
        axes[1, 2].set_title("Absolute vorticity error"); fig.colorbar(im, ax=axes[1, 2])
        for ax in axes.flat:
            ax.set(xlabel="x/L", ylabel="y/L", aspect="equal", xlim=(0, 1), ylim=(0, 1))
        fig.savefig(F / f"Cavity_{family}_grid51_streamlines_vorticity.png", dpi=170)
        plt.close(fig)
    vortex = pd.DataFrame(vortex_rows)
    vortex.to_csv(R / "vortex_metrics.csv", index=False)

    # A refinement comparison of CFD labels, evaluated at the coarse nodes.
    xi = np.linspace(0, 1, 25); yy, xx = np.meshgrid(xi, xi, indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    shift = []
    for i, case in enumerate(cases):
        if case["split"] != "test": continue
        fine_u = RegularGridInterpolator((np.linspace(0, 1, 51),)*2, data["u"][i])(points).reshape(25,25)
        fine_v = RegularGridInterpolator((np.linspace(0, 1, 51),)*2, data["v"][i])(points).reshape(25,25)
        err = np.linalg.norm(np.stack((old["u"][i]-fine_u, old["v"][i]-fine_v)))
        den = np.linalg.norm(np.stack((fine_u, fine_v)))
        shift.append(dict(case=case["id"], family=case["family"],
                          velocity_grid_shift_rel_l2=err/den))
    shifts = pd.DataFrame(shift)
    shifts.to_csv(R / "grid_shift.csv", index=False)
    return compare, shifts

BOOTSTRAP_TEMPLATE = """# FLOWMLLAB_COLAB_BOOTSTRAP_V1
# In Colab this cell obtains the complete repository and installs the tested package.
# In a local checkout it leaves the active environment and working directory unchanged.
from pathlib import Path as _FlowMLLabPath
import os as _flowmllab_os
import subprocess as _flowmllab_subprocess
import sys as _flowmllab_sys

if "google.colab" in _flowmllab_sys.modules or _flowmllab_os.environ.get("COLAB_RELEASE_TAG"):
    _flowmllab_root = _FlowMLLabPath("/content/FlowMLLab")
    if not (_flowmllab_root / ".git").is_dir():
        _flowmllab_subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(_flowmllab_root),
            ],
            check=True,
        )
    _flowmllab_subprocess.run(
        [
            _flowmllab_sys.executable, "-m", "pip", "install", "-q", "-e",
            f"{_flowmllab_root}[test]",
        ],
        check=True,
    )
    _flowmllab_notebook_dir = _flowmllab_root / "{NOTEBOOK_DIR}"
    _flowmllab_os.chdir(_flowmllab_notebook_dir)
    for _flowmllab_path in (_flowmllab_root, _flowmllab_notebook_dir):
        if str(_flowmllab_path) not in _flowmllab_sys.path:
            _flowmllab_sys.path.insert(0, str(_flowmllab_path))
    print("FlowMLLab ready:", _flowmllab_root)
"""


def notebook(compare: pd.DataFrame, shifts: pd.DataFrame) -> None:
    summary = "| Train | Test | 25 x 25 | 51 x 51 |\n|---|---|---:|---:|\n" + "\n".join(
        f"| {row.train} | {row.test} | {row.error_25_percent:.3f}% | {row.error_51_percent:.3f}% |"
        for row in compare.itertuples())
    shift = "| Family | Mean difference |\n|---|---:|\n" + "\n".join(
        f"| {family} | {100*value:.2f}% |" for family, value in
        shifts.groupby("family").velocity_grid_shift_rel_l2.mean().items())
    colab = ("https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/"
             "notebooks/week04/W4_Lab4_Grid51_Validation.ipynb")
    bootstrap = BOOTSTRAP_TEMPLATE.replace("{NOTEBOOK_DIR}", "notebooks/week04")
    cells = [
        nbf.v4.new_markdown_cell(
            f'<a href="{colab}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>\n\n'
            "# Week 4.2 addendum: 51 × 51 grid validation\n\n"
            "<!-- MIE690A article-aligned validation v4 -->\n\n"
            "This addendum to Lab 4 (Stokes-to-Navier–Stokes correction) solves every case again on a 51 × 51 grid "
            "and retrains the same correction network there. The original 25 × 25 experiment remains the baseline. "
            "The case manifest is unchanged, so the test cases are regression cases, not a new blind test.\n\n"
            "**Question to answer before reading the tables:** if the CFD labels themselves change by roughly 20% "
            "when the grid is refined, what does a 0.1% surrogate error on the coarse grid measure, and what does it not measure?"),
        nbf.v4.new_code_cell(bootstrap),
        nbf.v4.new_markdown_cell("## Mean velocity relative L2 error (%)\n\n" + summary + "\n\nErrors at 25 and 51 nodes are measured against their respective same-grid Navier–Stokes solutions. A lower surrogate error does not by itself establish mesh convergence."),
        nbf.v4.new_markdown_cell("## Change in the CFD reference under refinement\n\nThe 51-node CFD velocity is linearly sampled at the 25-node locations and compared with the original 25-node CFD velocity. Mean relative differences (%):\n\n" + shift + "\n\nThis is one refinement step, not an asymptotic convergence study."),
        nbf.v4.new_markdown_cell("## Retained figures\n\n![25 versus 51 grid error comparison](../../figures/Cavity_grid25_grid51_errors.png)\n\n![51-node constant-lid speed, streamlines and vorticity](../../figures/Cavity_constant_grid51_streamlines_vorticity.png)\n\n![51-node diverse-lid speed, streamlines and vorticity](../../figures/Cavity_diverse_grid51_streamlines_vorticity.png)"),
        nbf.v4.new_code_cell("""from pathlib import Path
import pandas as pd
from IPython.display import Image, display
# Locate the repository root from any working directory (notebook folder or checkout root).
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'results/stokes_grid51').is_dir())
R = ROOT / 'results/stokes_grid51'
display(pd.read_csv(R/'grid_comparison.csv'))
display(pd.read_csv(R/'grid_shift.csv').groupby('family').velocity_grid_shift_rel_l2.mean())
display(pd.read_csv(R/'vortex_metrics.csv'))"""),
        nbf.v4.new_code_cell("""for name in ('Cavity_grid25_grid51_errors.png',
             'Cavity_constant_grid51_streamlines_vorticity.png',
             'Cavity_diverse_grid51_streamlines_vorticity.png'):
    display(Image(filename=str(ROOT/'figures'/name)))"""),
        nbf.v4.new_markdown_cell(
            "## What this addendum establishes\n\n"
            "Two different errors appear above and they must not be confused.\n\n"
            "1. **Surrogate error** (first table): how far the trained correction is from the Navier–Stokes solution "
            "*on the same grid*. It is about 0.1% (constant lids) and 0.8% (diverse lids) at 51 × 51, slightly better than at 25 × 25.\n"
            "2. **Label change under refinement** (second table): how far the 25 × 25 CFD solution is from the 51 × 51 solution "
            "sampled at the same nodes. It is about 18 to 20% for the two main families.\n\n"
            "The surrogate error is therefore roughly one hundred times smaller than the discretization error of the labels it was trained on. "
            "A surrogate cannot be more accurate than its labels: the 0.1% measures how well the network reproduces *this solver on this grid*, "
            "not how close either is to the converged cavity flow. This is why Lab 4 calls its results same-grid regression evidence and why a "
            "mesh-convergence study (at least three grids and an observed order) is required before any physical accuracy claim.\n\n"
            "**Exercise.** Using `grid_shift.csv`, find the family whose labels changed least under refinement and the one that changed most. "
            "Propose one reason for the difference that you could test with a third grid."),
        nbf.v4.new_markdown_cell("## Reproduce\n\nFrom the repository root, run `python qa/run_week04_2_grid51.py generate` and then `python qa/run_week04_2_grid51.py train`. The 51-node fields are solved afresh; the network uses the 25-node study's fixed selected architecture (POD rank 24, two tanh layers of width 96, seeds 7/17/27). Training and validation use separate complete cases. The plotted corner extrema are grid-resolved candidates and need finer-grid or independent CFD confirmation."),
    ]
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata.update({"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
                          "language_info": {"name":"python"}})
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(book, NB)


def pdf(compare: pd.DataFrame, shifts: pd.DataFrame) -> None:
    c = canvas.Canvas(str(PDF), pagesize=A4)
    w, h = A4
    c.setTitle("FlowMLLab Week 4.2 - 51 x 51 Grid Validation")
    def title(text: str, page: int):
        c.setFont("Helvetica-Bold", 16); c.drawString(17*mm, h-22*mm, text)
        c.setFont("Helvetica", 8); c.drawRightString(w-17*mm, h-22*mm, f"Week 4.2 | {page}/3")
    title("51 x 51 numerical refinement", 1)
    c.setFont("Helvetica", 10)
    lines = ["Matched Stokes fields and Navier-Stokes labels were solved again on 51 x 51 nodes.",
             "The retained case manifest is unchanged; tests are regression cases, not fresh blind data.",
             "The selected POD-MLP architecture was retrained from scratch at 51 x 51."]
    for j, line in enumerate(lines): c.drawString(17*mm, h-(34+7*j)*mm, line)
    c.drawImage(str(F/"Cavity_grid25_grid51_errors.png"), 18*mm, h-156*mm,
                width=w-36*mm, height=89*mm, preserveAspectRatio=True, anchor="c")
    y = h-169*mm
    c.setFont("Helvetica-Bold", 9)
    c.drawString(18*mm,y,"Train -> test"); c.drawString(80*mm,y,"25 x 25"); c.drawString(119*mm,y,"51 x 51")
    c.setFont("Helvetica", 9)
    for j, row in enumerate(compare.itertuples()):
        yy=y-(j+1)*7*mm
        c.drawString(18*mm,yy,f"{row.train} -> {row.test}")
        c.drawString(80*mm,yy,f"{row.error_25_percent:.3f}%")
        c.drawString(119*mm,yy,f"{row.error_51_percent:.3f}%")
    c.setFont("Helvetica", 8)
    c.drawString(17*mm, 20*mm, "Errors compare models with separate numerical labels on each grid.")
    c.showPage()
    for page, family in ((2,"constant"),(3,"diverse")):
        title(f"{family.title()} lid: speed, streamlines, vorticity", page)
        c.drawImage(str(F/f"Cavity_{family}_grid51_streamlines_vorticity.png"),
                    12*mm, 108*mm, width=w-24*mm, height=130*mm,
                    preserveAspectRatio=True, anchor="c")
        group = shifts[shifts.family == family]
        c.setFont("Helvetica", 8.5)
        c.drawString(17*mm, 91*mm,
                     f"Mean CFD 25-to-51 velocity change at coarse nodes: {100*group.velocity_grid_shift_rel_l2.mean():.2f}%")
        c.drawString(17*mm, 82*mm, "Vortex positions are measured on 51-node arrays; display contours do not add resolution.")
        c.drawString(17*mm, 73*mm, "One refinement step does not establish mesh-independent vortex capture.")
        c.showPage()
    c.save()


if __name__ == "__main__":
    if "--notebook-only" in sys.argv:
        # Rebuild the notebook from the retained CSV evidence without re-plotting or re-solving.
        comp = pd.read_csv(R / "grid_comparison.csv")
        diff = pd.read_csv(R / "grid_shift.csv")
        notebook(comp, diff)
        print(NB.relative_to(ROOT))
    else:
        comp, diff = figures()
        notebook(comp, diff)
        pdf(comp, diff)
        print(NB.relative_to(ROOT), PDF.relative_to(ROOT))
