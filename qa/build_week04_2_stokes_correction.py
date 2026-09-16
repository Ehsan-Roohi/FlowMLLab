#!/usr/bin/env python3
"""Build the executed Week 4.2 notebook, figures, and eight-page lecture PDF."""

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
from flowmllab.cavity_diversity import lid_profile


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
        nbf.v4.new_code_cell("""display(Image(filename=str(ROOT/'figures/Cavity_constant_velocity_pressure.png')))
display(Image(filename=str(ROOT/'figures/Cavity_diverse_velocity_pressure.png')))"""),
        nbf.v4.new_markdown_cell("""## 9. Interpretation and claim boundary

The diverse-lid model generalizes much better across boundary families and OOD groups. Constant-only training remains poor on diverse lids because it never learns dependence on lid shape. The best retained mean errors are 0.107% for constant-to-constant and 0.917% for diverse-to-diverse velocity prediction.

These are same-grid surrogate errors against an educational finite-difference solver. The study does not establish mesh-independent CFD accuracy, a new neural architecture, or greater physical fidelity than the numerical labels. The useful result is narrower: an actual Stokes field is a strong low-fidelity coordinate for learning the nonlinear Navier-Stokes correction when the training boundary conditions span the intended use."""),
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
        c.drawRightString(width - 16*mm, height - 14*mm, f"FlowMLLab | Week 4.2 | {page}/8")
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
    y = _p(c, "First test case in manifest order; never selected by prediction quality. Left: Navier-Stokes reference. Center: Stokes plus learned correction. Right: absolute error. Shared reference/prediction scales are used within each row.", 16*mm, y, width-32*mm, small) - 2*mm
    c.drawImage(str(ROOT / "figures/Cavity_constant_velocity_pressure.png"), 19*mm, 26*mm,
                width=width-38*mm, height=y-29*mm, preserveAspectRatio=True, anchor="c")
    finish()

    # Page 8
    y = header(8, "Held-out diverse-lid field and claim boundary")
    y = _p(c, "The diverse model is evaluated on the first diverse-lid test case in manifest order. The pressure row is a recovered field rather than a network output.", 16*mm, y, width-32*mm, small) - 2*mm
    c.drawImage(str(ROOT / "figures/Cavity_diverse_velocity_pressure.png"), 23*mm, 73*mm,
                width=width-46*mm, height=y-78*mm, preserveAspectRatio=True, anchor="c")
    y = 67*mm
    y = _p(c, "What is established", 16*mm, y, 80*mm, h2)
    _p(c, "Actual Stokes input; consistent Re-speed coupling; complete-case splits; validation-only selection; three-seed ensemble; strong in-family correction; useful transfer from diverse to constant lids.", 16*mm, y-2*mm, 80*mm, small)
    y2 = _p(c, "What is not established", 111*mm, 67*mm, 82*mm, h2)
    _p(c, "Mesh independence; high-fidelity CFD validation; independent pressure validation; a new network architecture; universal transfer from a constant lid to unseen boundary shapes.", 111*mm, y2-2*mm, 82*mm, small)
    finish()
    c.save()


def main() -> None:
    figures = make_figures()
    build_notebook(figures)
    build_pdf(figures)
    print(NOTEBOOK)
    print(PDF)


if __name__ == "__main__":
    main()
