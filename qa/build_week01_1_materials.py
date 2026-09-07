#!/usr/bin/env python3
"""Build the original Week 1.1 evidence, notebook, and lecture PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results" / "week01_1_scientific_software"
NOTEBOOK_DIR = ROOT / "notebooks" / "week01_1"
LECTURE = ROOT / "lectures" / "week01_1_ai_assisted_scientific_software.pdf"
OUTPUT_PDF = ROOT / "output" / "pdf" / LECTURE.name


def _science():
    from flowmllab.scientific_software import (
        audit_cavity_case,
        differential_diagnostics,
        evaluate_acceptance,
        load_cavity_case,
        manufactured_incompressible_field,
        verification_sweep,
        write_acceptance_record,
    )

    verification = verification_sweep()
    cavity = audit_cavity_case(ROOT, 100.0)
    record = evaluate_acceptance(verification, cavity)
    write_acceptance_record(record, RESULTS / "acceptance_record.json")
    case, _ = load_cavity_case(ROOT, 100.0)
    diagnostic = differential_diagnostics(case["x"], case["y"], case["u"], case["v"])
    x, y, u, v, exact = manufactured_incompressible_field(129)
    swapped = np.gradient(v, y, axis=0, edge_order=2) - np.gradient(
        u, x, axis=1, edge_order=2
    )
    interior = np.s_[2:-2, 2:-2]
    faulty_error = float(
        np.linalg.norm((swapped - exact)[interior]) / np.linalg.norm(exact[interior])
    )
    return verification, cavity, record, case, diagnostic, faulty_error


def build_figure(verification, record, case, diagnostic) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#9aa8ba",
            "axes.linewidth": 0.8,
        }
    )
    fig = plt.figure(figsize=(16, 5.8), facecolor="#f6f8fb", constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.0, 1.12])

    ax0 = fig.add_subplot(gs[0, 0])
    x, y = np.asarray(case["x"]), np.asarray(case["y"])
    w = diagnostic.vorticity
    limit = float(np.percentile(np.abs(w), 98.5))
    levels = np.linspace(-limit, limit, 25)
    contour = ax0.contourf(
        x,
        y,
        w,
        levels=levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        extend="both",
    )
    ax0.streamplot(x, y, case["u"], case["v"], color="#17233b", density=0.65, linewidth=0.45)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x/L")
    ax0.set_ylabel("y/L")
    ax0.set_title("A  Qualified Re=100 cavity diagnostic", loc="left", color="#17233b")
    cb = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=0.03)
    cb.set_label(r"$\omega_z L/U$")

    ax1 = fig.add_subplot(gs[0, 1])
    rows = verification["rows"]
    h = np.asarray([row["h"] for row in rows])
    error = np.asarray([row["vorticity_relative_l2"] for row in rows])
    reference = error[-1] * (h / h[-1]) ** 2
    ax1.loglog(h, error, "o-", color="#007f86", lw=2.4, ms=7, label="measured")
    ax1.loglog(h, reference, "--", color="#ef7d32", lw=1.8, label=r"$O(h^2)$ reference")
    for row in rows:
        ax1.annotate(
            f"{row['points']}x{row['points']}",
            (row["h"], row["vorticity_relative_l2"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax1.invert_xaxis()
    ax1.grid(True, which="both", alpha=0.25)
    ax1.set_xlabel("grid spacing h/L (finer to the right)")
    ax1.set_ylabel("relative L2 vorticity error")
    ax1.set_title(
        f"B  Manufactured verification: p={verification['observed_order']:.3f}",
        loc="left",
        color="#17233b",
    )
    ax1.legend(frameon=False, loc="lower left")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    ax2.set_title("C  Predeclared acceptance decision", loc="left", color="#17233b", pad=12)
    labels = [
        ("Dataset identity", str(record["cavity"]["dataset_sha256"])[:10] + "..."),
        ("Second-order verification", f"p = {verification['observed_order']:.3f}"),
        ("Fine-grid analytic error", f"{rows[-1]['vorticity_relative_l2']:.2e}"),
        ("Manufactured div. RMS", f"{max(r['divergence_rms'] for r in rows):.2e}"),
        ("CFD div. RMS", f"{record['cavity']['interior_divergence_rms']:.2e}"),
        ("CFD/archive vorticity", f"{record['cavity']['archive_vorticity_relative_l2']:.3%}"),
        ("Wall velocity max error", f"{record['cavity']['wall_velocity_max_abs_error']:.1e}"),
    ]
    y0 = 0.90
    for index, ((label, value), passed) in enumerate(zip(labels, record["gates"].values())):
        y_pos = y0 - index * 0.112
        ax2.add_patch(
            plt.Rectangle((0.02, y_pos - 0.065), 0.96, 0.095, color="white", ec="#d6dee8", lw=0.8)
        )
        ax2.text(0.06, y_pos - 0.018, label, va="center", color="#26364f", fontsize=10)
        ax2.text(0.77, y_pos - 0.018, value, va="center", ha="right", color="#26364f", fontsize=10)
        ax2.text(
            0.92,
            y_pos - 0.018,
            "PASS" if passed else "FAIL",
            va="center",
            ha="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#16826c" if passed else "#c8463b", ec="none"),
        )
    ax2.text(
        0.02,
        0.025,
        "Decision: ACCEPT  |  thresholds frozen before the CFD audit",
        fontsize=10.5,
        fontweight="bold",
        color="#007f86",
    )
    fig.suptitle(
        "Week 1.1 - Executable scientific contract for an AI-proposed diagnostic",
        fontsize=17,
        fontweight="bold",
        color="#17233b",
    )
    path = RESULTS / "week01_1_acceptance_summary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _cell_id(index: int) -> str:
    return f"w11-{index:02d}"


def build_notebook() -> Path:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    cells = []

    def md(text: str) -> None:
        cell = nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())
        cell["id"] = _cell_id(len(cells))
        cells.append(cell)

    def code(text: str) -> None:
        cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
        cell["id"] = _cell_id(len(cells))
        cells.append(cell)

    md(
        r"""
        # Week 1.1 - AI-assisted scientific software

        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01_1/W1_1_AI_Assisted_Scientific_Software.ipynb)

        <!-- MIE690A article-aligned validation v4 -->

        **Research question.** Can a velocity-field diagnostic proposed by a
        person or coding agent be accepted using evidence fixed before the final
        CFD result is inspected?

        This notebook is intentionally vendor-neutral and makes no remote model
        call. It audits the executable contract in `SCIENTIFIC_SPEC.md` through
        analytic verification, a real accepted cavity field, physical gates,
        and content-addressed provenance.

        **Learning outcomes**

        1. distinguish execution, verification, validation, and reproducibility;
        2. expose array-axis and sign conventions as testable requirements;
        3. predeclare thresholds and retain failures; and
        4. state which decisions remain the investigator's responsibility.
        """
    )
    code(
        """
        from pathlib import Path
        import os, subprocess, sys

        # FLOWMLLAB_COLAB_BOOTSTRAP_V1

        def find_root(start=Path.cwd()):
            for candidate in (start, *start.parents):
                if (candidate / "flowmllab" / "scientific_software.py").is_file():
                    return candidate
            return None

        ROOT = find_root()
        try:
            import google.colab  # noqa: F401
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False
        if ROOT is None and IN_COLAB:
            ROOT = Path("/content/FlowMLLab")
            if not ROOT.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1", "https://github.com/Ehsan-Roohi/FlowMLLab.git", str(ROOT)],
                    check=True,
                )
        if ROOT is None:
            raise FileNotFoundError("Run inside a complete FlowMLLab checkout")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        print("FlowMLLab checkout located; scientific module import enabled")
        """
    )
    code(
        """
        import json
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        from flowmllab.scientific_software import (
            DEFAULT_THRESHOLDS,
            audit_cavity_case,
            differential_diagnostics,
            evaluate_acceptance,
            load_cavity_case,
            manufactured_incompressible_field,
            verification_sweep,
        )
        plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.25})
        """
    )
    md(
        r"""
        ## 1. Freeze the contract before looking at the final case

        We define $u[j,i]=u(y_j,x_i)$ and

        $$\nabla\cdot\mathbf{u}=\partial_xu+\partial_yv,\qquad
        \omega_z=\partial_xv-\partial_yu.$$

        A correct implementation must therefore encode both the array axes and
        the vorticity sign. The limits below are part of the experiment, not
        plotting preferences.
        """
    )
    code(
        """
        pd.DataFrame(
            {"gate": list(DEFAULT_THRESHOLDS), "frozen threshold": list(DEFAULT_THRESHOLDS.values())}
        )
        """
    )
    md(
        r"""
        ## 2. Verify against a known solution

        Use $\psi=\sin^2(\pi x)\sin^2(\pi y)$,
        $u=\partial_y\psi$, and $v=-\partial_x\psi$. This field has exact zero
        divergence, vanishes at every wall, and has an analytic vorticity. Four
        grids test convergence; a single fine grid cannot establish order.
        """
    )
    code(
        """
        verification = verification_sweep((17, 33, 65, 129))
        verification_table = pd.DataFrame(verification["rows"])
        display(verification_table)
        print(f"Observed vorticity order p = {verification['observed_order']:.4f}")
        """
    )
    code(
        """
        h = verification_table["h"].to_numpy()
        error = verification_table["vorticity_relative_l2"].to_numpy()
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.loglog(h, error, "o-", lw=2.2, label="measured")
        ax.loglog(h, error[-1] * (h / h[-1])**2, "--", lw=1.8, label=r"$O(h^2)$")
        ax.invert_xaxis(); ax.set_xlabel("h/L (finer to the right)")
        ax.set_ylabel("relative L2 vorticity error"); ax.legend(frameon=False)
        ax.set_title("Manufactured-solution verification")
        plt.show()
        """
    )
    md(
        r"""
        ## 3. A plausible axis bug must fail

        The next expression executes and returns an array of the expected shape,
        yet differentiates with respect to the wrong coordinate directions. A
        shape test alone cannot detect it.
        """
    )
    code(
        """
        x, y, u, v, omega_exact = manufactured_incompressible_field(129)
        omega_axis_swapped = (
            np.gradient(v, y, axis=0, edge_order=2)
            - np.gradient(u, x, axis=1, edge_order=2)
        )
        interior = np.s_[2:-2, 2:-2]
        faulty_relative_l2 = np.linalg.norm((omega_axis_swapped-omega_exact)[interior]) / np.linalg.norm(omega_exact[interior])
        print(f"Axis-swapped relative L2 = {faulty_relative_l2:.3f}")
        print("Gate:", "PASS" if faulty_relative_l2 <= DEFAULT_THRESHOLDS["fine_vorticity_relative_l2_max"] else "FAIL (expected)")
        """
    )
    md(
        """
        ## 4. Open the real CFD case only after verification

        The `Re=100` case is an accepted 65 by 65 lid-driven-cavity field from
        the fixed Week-1 archive. We preserve its split label and SHA-256 digest,
        exclude two outer stencil layers from derivative comparisons, and test
        the walls separately. Agreement with archived vorticity is not expected
        to be exact because the stored solver quantity and this post-processing
        stencil differ near boundaries.
        """
    )
    code(
        """
        cavity = audit_cavity_case(ROOT, 100.0)
        display(pd.DataFrame([cavity]).T.rename(columns={0: "value"}))
        """
    )
    code(
        """
        case, _ = load_cavity_case(ROOT, 100.0)
        diagnostic = differential_diagnostics(case["x"], case["y"], case["u"], case["v"])
        fig, ax = plt.subplots(figsize=(5.6, 4.7))
        lim = np.percentile(np.abs(diagnostic.vorticity), 98.5)
        cf = ax.contourf(case["x"], case["y"], diagnostic.vorticity, 25, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.streamplot(case["x"], case["y"], case["u"], case["v"], color="k", density=.65, linewidth=.45)
        ax.set_aspect("equal"); ax.set_xlabel("x/L"); ax.set_ylabel("y/L")
        ax.set_title("Qualified Re=100 diagnostic"); fig.colorbar(cf, ax=ax, label=r"$\\omega_zL/U$")
        plt.show()
        """
    )
    md(
        """
        ## 5. Make one machine-readable decision

        Every failed gate remains visible. The final Boolean does not replace the
        underlying values, thresholds, dataset identity, or scientific review.
        """
    )
    code(
        """
        record = evaluate_acceptance(verification, cavity)
        gate_table = pd.DataFrame(
            [{"gate": gate, "passed": passed} for gate, passed in record["gates"].items()]
        )
        display(gate_table)
        print("FINAL DECISION:", record["decision"].upper())
        retained = json.loads((ROOT / "results/week01_1_scientific_software/acceptance_record.json").read_text())
        assert retained == record
        """
    )
    md(
        """
        ## 6. Required AI-use disclosure and manual review

        Copy this block into your report and complete it even if no agent was used:

        - **Tool/model/date:**
        - **Exact specification supplied:**
        - **Functions or files proposed by the tool:**
        - **Material human corrections and physical reason:**
        - **Independent commands executed:**
        - **Failed gates retained:**
        - **Bounded claim:**

        Manual review must address at least one scientific issue: axis order,
        units, sign convention, stencil support, boundary treatment, reference
        suitability, or threshold justification. Naming and formatting comments
        alone do not satisfy the requirement.
        """
    )
    md(
        """
        ## 7. Controlled extension

        Ask a coding agent, a collaborator, or yourself to make exactly one
        controlled change: support a stretched grid, add circulation, or refactor
        the derivative kernel. Do not change the frozen gates after opening the
        cavity result. Submit the specification, diff, complete test output,
        acceptance JSON, manual review, and a claim limited to the evidence.

        **Exit question:** Which passing gate would still be insufficient for an
        unstructured-mesh turbulent-flow claim, and what new reference would you
        require?
        """
    )
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": "W1_1_AI_Assisted_Scientific_Software.ipynb", "provenance": []},
    }
    path = NOTEBOOK_DIR / "W1_1_AI_Assisted_Scientific_Software.ipynb"
    nbf.write(nb, path)
    return path


def execute_notebook(path: Path) -> Path:
    """Execute the generated notebook in place from the repository root."""

    notebook = nbf.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    # nbclient records wall-clock timestamps and the executing interpreter's
    # patch version. They are useful operational logs but make a retained
    # teaching notebook change byte-for-byte on every clean execution.
    for cell in notebook.cells:
        cell.get("metadata", {}).pop("execution", None)
    notebook.metadata["language_info"] = {"name": "python", "version": "3.11"}
    nbf.write(notebook, path)
    return path


PAGE = landscape((13.333 * inch, 7.5 * inch))
NAVY = colors.HexColor("#14233B")
TEAL = colors.HexColor("#007F86")
ORANGE = colors.HexColor("#ED7D31")
INK = colors.HexColor("#26364F")
PALE = colors.HexColor("#F4F7FA")
MUTED = colors.HexColor("#607086")
GREEN = colors.HexColor("#16826C")
RED = colors.HexColor("#C8463B")


def _font_setup() -> None:
    """Use PDF base fonts so the deck renders identically across platforms."""


def _paragraph(c, text, x, y, w, h, size=18, color=INK, leading=None, bold=False, align=TA_LEFT):
    font = "Helvetica-Bold" if bold else "Helvetica"
    style = ParagraphStyle(
        "deck",
        fontName=font,
        fontSize=size,
        leading=leading or size * 1.28,
        textColor=color,
        alignment=align,
        spaceAfter=0,
    )
    para = Paragraph(text, style)
    para.wrapOn(c, w, h)
    para.drawOn(c, x, y + h - para.height)
    return para.height


def _header(c, title, number):
    c.setFillColor(PALE); c.rect(0, 0, PAGE[0], PAGE[1], fill=1, stroke=0)
    c.setFillColor(NAVY); c.rect(0, PAGE[1] - 0.12 * inch, PAGE[0], 0.12 * inch, fill=1, stroke=0)
    _paragraph(c, title, 0.62 * inch, PAGE[1] - 0.88 * inch, 11.8 * inch, 0.5 * inch, 23, NAVY, bold=True)
    c.setStrokeColor(colors.HexColor("#D5DEE8")); c.line(0.62 * inch, 0.48 * inch, 12.7 * inch, 0.48 * inch)
    _paragraph(c, "FlowMLLab | Week 1.1 | AI-assisted scientific software", 0.62 * inch, 0.1 * inch, 8 * inch, 0.26 * inch, 8.5, MUTED)
    _paragraph(c, str(number), 12.25 * inch, 0.1 * inch, 0.4 * inch, 0.26 * inch, 8.5, MUTED, align=TA_CENTER)


def _card(c, x, y, w, h, title, body, accent=TEAL):
    c.setFillColor(colors.white); c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
    c.setFillColor(accent); c.roundRect(x, y, 0.08 * inch, h, 4, fill=1, stroke=0)
    _paragraph(c, title, x + 0.25 * inch, y + h - 0.52 * inch, w - 0.45 * inch, 0.32 * inch, 14, NAVY, bold=True)
    _paragraph(c, body, x + 0.25 * inch, y + 0.18 * inch, w - 0.45 * inch, h - 0.75 * inch, 10.5, INK)


def _bullet_block(c, items, x, y, w, h, size=15):
    text = "<br/>".join(f'<font color="#007F86"><b>-</b></font>&nbsp;&nbsp;{item}' for item in items)
    _paragraph(c, text, x, y, w, h, size, INK, leading=size * 1.55)


def build_pdf(figure_path: Path, verification, cavity, record, faulty_error) -> Path:
    _font_setup()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT_PDF), pagesize=PAGE, pageCompression=1, invariant=1)
    c.setTitle("Week 1.1 - AI-Assisted Scientific Software")
    c.setAuthor("Ehsan Roohi - FlowMLLab")

    # 1
    c.setFillColor(NAVY); c.rect(0, 0, PAGE[0], PAGE[1], fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(0.65 * inch, 1.0 * inch, 0.11 * inch, 5.25 * inch, fill=1, stroke=0)
    _paragraph(c, "WEEK 1.1", 1.05 * inch, 5.75 * inch, 3 * inch, 0.35 * inch, 15, colors.HexColor("#67D3D6"), bold=True)
    _paragraph(c, "AI-Assisted<br/>Scientific Software", 1.05 * inch, 3.05 * inch, 8.2 * inch, 2.5 * inch, 35, colors.white, leading=40, bold=True)
    _paragraph(c, "Specification, verification, physical gates, and accountable human review", 1.08 * inch, 2.18 * inch, 8.8 * inch, 0.7 * inch, 17, colors.HexColor("#DCE5EF"))
    _paragraph(c, "A vendor-neutral FlowMLLab module using the accepted Re=100 cavity field", 1.08 * inch, 1.48 * inch, 8.8 * inch, 0.42 * inch, 11, colors.HexColor("#9EB0C5"))
    c.setFillColor(ORANGE); c.circle(11.2 * inch, 4.7 * inch, 0.72 * inch, fill=1, stroke=0)
    c.setFillColor(TEAL); c.circle(11.65 * inch, 3.75 * inch, 0.52 * inch, fill=1, stroke=0)
    c.setStrokeColor(colors.white); c.setLineWidth(2.5)
    c.line(10.2 * inch, 2.2 * inch, 12.25 * inch, 5.7 * inch)
    c.showPage()

    # 2
    _header(c, "The trust gap: executing code is the weakest gate", 2)
    cards = [
        ("Runs", "Imports resolve, shapes match, and a number is returned.", colors.HexColor("#6B7C93")),
        ("Verified", "The discretization converges toward a known mathematical answer.", TEAL),
        ("Physically valid", "Conservation, boundaries, units, and reference behavior are satisfied.", ORANGE),
        ("Reproducible", "Inputs, thresholds, environment, and decision evidence remain inspectable.", GREEN),
    ]
    for i, item in enumerate(cards):
        _card(c, 0.68 * inch + i * 3.08 * inch, 2.2 * inch, 2.72 * inch, 3.5 * inch, *item)
    _paragraph(c, "AI can accelerate implementation. It cannot silently choose the scientific contract.", 1.2 * inch, 1.05 * inch, 10.9 * inch, 0.5 * inch, 19, NAVY, bold=True, align=TA_CENTER)
    c.showPage()

    # 3
    _header(c, "Five failure modes a conventional unit test can miss", 3)
    failures = [
        ("Coordinate semantics", "field[y,x] is treated as field[x,y]"),
        ("Physics convention", "the vorticity sign or nondimensional scale changes"),
        ("Boundary support", "a first-order edge stencil contaminates a wall claim"),
        ("Reference leakage", "thresholds are tuned after the final case is opened"),
        ("Evidence drift", "the data file changes while the reported metric does not"),
    ]
    for i, (title, body) in enumerate(failures):
        y = 5.55 * inch - i * 0.95 * inch
        c.setFillColor(colors.white); c.roundRect(0.9 * inch, y - 0.47 * inch, 11.45 * inch, 0.7 * inch, 7, fill=1, stroke=0)
        c.setFillColor(RED); c.circle(1.3 * inch, y - 0.12 * inch, 0.17 * inch, fill=1, stroke=0)
        _paragraph(c, str(i + 1), 1.15 * inch, y - 0.23 * inch, 0.3 * inch, 0.25 * inch, 11, colors.white, bold=True, align=TA_CENTER)
        _paragraph(c, title, 1.7 * inch, y - 0.28 * inch, 2.3 * inch, 0.35 * inch, 13, NAVY, bold=True)
        _paragraph(c, body, 4.1 * inch, y - 0.28 * inch, 7.7 * inch, 0.35 * inch, 12, INK)
    c.showPage()

    # 4
    _header(c, "Specification before implementation", 4)
    _card(c, 0.75 * inch, 1.3 * inch, 3.7 * inch, 4.7 * inch, "1 | Scientific intent", "Quantity, sign, units, axes, domain, boundary interpretation, and intended decision.", TEAL)
    _card(c, 4.8 * inch, 1.3 * inch, 3.7 * inch, 4.7 * inch, "2 | Executable evidence", "Known solution, grid sequence, qualified CFD case, metrics, provenance digest, and failure behavior.", ORANGE)
    _card(c, 8.85 * inch, 1.3 * inch, 3.7 * inch, 4.7 * inch, "3 | Frozen decision", "Numerical thresholds selected before the final case, one accept/reject record, and a bounded claim.", GREEN)
    c.setStrokeColor(MUTED); c.setLineWidth(1.8)
    c.line(4.47 * inch, 3.65 * inch, 4.77 * inch, 3.65 * inch); c.line(8.52 * inch, 3.65 * inch, 8.82 * inch, 3.65 * inch)
    c.showPage()

    # 5
    _header(c, "The executable contract: thresholds are part of the method", 5)
    table_data = [
        ["Gate", "Threshold", "Scientific purpose"],
        ["Dataset identity", "SHA-256 exact", "fail closed on input drift"],
        ["Observed order", "p >= 1.90", "demonstrate convergence"],
        ["Fine-grid vorticity", "rel. L2 <= 5e-4", "bound analytic error"],
        ["Manufactured divergence", "RMS <= 1e-12", "check solenoidal identity"],
        ["CFD divergence", "RMS <= 1e-12", "check axes and field contract"],
        ["CFD/archive vorticity", "rel. L2 <= 4e-2", "check sign and stencil agreement"],
        ["Wall velocities", "max <= 1e-12", "protect no-slip and lid conditions"],
    ]
    table = Table(table_data, colWidths=[3.2 * inch, 2.2 * inch, 5.7 * inch], rowHeights=[0.52 * inch] + [0.55 * inch] * 7)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11), ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED8E4")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    table.wrapOn(c, 11.1 * inch, 4.5 * inch); table.drawOn(c, 1.1 * inch, 1.28 * inch)
    c.showPage()

    # 6
    _header(c, "Reference diagnostic: make semantics explicit", 6)
    _card(c, 0.8 * inch, 3.55 * inch, 5.8 * inch, 2.2 * inch, "Array contract", "u[j,i] = u(y[j], x[i])<br/>axis 0 is y; axis 1 is x<br/>coordinates are finite and strictly increasing", TEAL)
    _card(c, 6.75 * inch, 3.55 * inch, 5.8 * inch, 2.2 * inch, "Differential contract", "div(u) = du/dx + dv/dy<br/>omega_z = dv/dx - du/dy<br/>second-order edges; two-layer interior audit", ORANGE)
    _paragraph(c, "The test field is generated from a streamfunction", 0.95 * inch, 2.55 * inch, 11.4 * inch, 0.4 * inch, 18, NAVY, bold=True, align=TA_CENTER)
    _paragraph(c, "psi = sin(pi x)^2 sin(pi y)^2    |    u = dpsi/dy    |    v = -dpsi/dx", 1.1 * inch, 1.75 * inch, 11.1 * inch, 0.55 * inch, 20, TEAL, align=TA_CENTER)
    _paragraph(c, "Exact divergence is zero; exact vorticity is available analytically; all wall velocities vanish.", 1.2 * inch, 1.15 * inch, 10.9 * inch, 0.4 * inch, 13, INK, align=TA_CENTER)
    c.showPage()

    # 7
    _header(c, "Evidence: known answer first, accepted CFD second", 7)
    c.drawImage(str(figure_path), 0.58 * inch, 0.8 * inch, width=12.15 * inch, height=4.95 * inch, preserveAspectRatio=True, anchor="c")
    c.showPage()

    # 8
    _header(c, "The real-case audit is not a second verification study", 8)
    metrics = [
        ("Data identity", str(cavity["dataset_sha256"])[:12] + "...", TEAL),
        ("Interior div. RMS", f"{cavity['interior_divergence_rms']:.2e}", GREEN),
        ("Archive vorticity L2", f"{cavity['archive_vorticity_relative_l2']:.3%}", ORANGE),
        ("Wall max error", f"{cavity['wall_velocity_max_abs_error']:.1e}", GREEN),
    ]
    for i, (title, value, accent) in enumerate(metrics):
        x = 0.78 * inch + (i % 2) * 6.05 * inch
        y = 3.75 * inch if i < 2 else 1.55 * inch
        _card(c, x, y, 5.65 * inch, 1.72 * inch, title, f'<font size="22" color="#26364F"><b>{value}</b></font>', accent)
    _paragraph(c, "Agreement means the post-processor respects this archive's convention and support. It does not validate every flow, mesh, or solver.", 1.15 * inch, 0.75 * inch, 11.0 * inch, 0.48 * inch, 13, NAVY, bold=True, align=TA_CENTER)
    c.showPage()

    # 9
    _header(c, "Adversarial check: a plausible axis bug executes and fails", 9)
    _card(c, 0.9 * inch, 2.0 * inch, 5.25 * inch, 3.8 * inch, "Faulty proposal", "Differentiate v along axis 0 using y and u along axis 1 using x, then subtract. Shapes match. Values are finite. No exception is raised.", RED)
    _card(c, 7.15 * inch, 2.0 * inch, 5.25 * inch, 3.8 * inch, "Scientific verdict", f'<font size="24"><b>relative L2 = {faulty_error:.3f}</b></font><br/><br/>Required <= 5e-4<br/><br/><font color="#C8463B"><b>REJECT</b></font>', ORANGE)
    c.setStrokeColor(MUTED); c.setLineWidth(2); c.line(6.25 * inch, 3.9 * inch, 7.0 * inch, 3.9 * inch)
    _paragraph(c, "A conventional shape test would pass. The manufactured solution exposes the semantic error.", 1.2 * inch, 1.05 * inch, 10.9 * inch, 0.48 * inch, 17, NAVY, bold=True, align=TA_CENTER)
    c.showPage()

    # 10
    _header(c, "Human-agent loop with explicit authority boundaries", 10)
    steps = [
        ("Human", "Define physics and claim", NAVY),
        ("Agent or human", "Propose code and tests", TEAL),
        ("Harness", "Run frozen gates", ORANGE),
        ("Human", "Review semantics and evidence", NAVY),
        ("Record", "Accept, reject, or revise", GREEN),
    ]
    for i, (who, action, color) in enumerate(steps):
        x = 0.55 * inch + i * 2.55 * inch
        c.setFillColor(colors.white); c.roundRect(x, 2.65 * inch, 2.05 * inch, 2.1 * inch, 10, fill=1, stroke=0)
        c.setFillColor(color); c.roundRect(x, 4.18 * inch, 2.05 * inch, 0.57 * inch, 10, fill=1, stroke=0)
        _paragraph(c, who, x + 0.12 * inch, 4.19 * inch, 1.8 * inch, 0.35 * inch, 11, colors.white, bold=True, align=TA_CENTER)
        _paragraph(c, action, x + 0.18 * inch, 3.0 * inch, 1.68 * inch, 0.9 * inch, 13, INK, bold=True, align=TA_CENTER)
        if i < len(steps) - 1:
            c.setStrokeColor(MUTED); c.setLineWidth(1.8); c.line(x + 2.08 * inch, 3.7 * inch, x + 2.48 * inch, 3.7 * inch)
    _paragraph(c, "Only the investigator may change conventions, references, thresholds, or the final scientific claim.", 1.15 * inch, 1.2 * inch, 11.0 * inch, 0.5 * inch, 17, NAVY, bold=True, align=TA_CENTER)
    c.showPage()

    # 11
    _header(c, "Assessment: evidence quality, not AI usage", 11)
    rubric = [
        ["Deliverable", "Weight", "Non-negotiable evidence"],
        ["Specification", "20%", "axes, units, sign, support, thresholds"],
        ["Verification", "25%", "known solution and observed order"],
        ["Physical audit", "20%", "boundaries, conservation, reference case"],
        ["Manual review", "15%", "one scientific issue, not only style"],
        ["Provenance", "10%", "input digest and exact commands"],
        ["Claim boundary", "10%", "limits and retained failures"],
    ]
    table = Table(rubric, colWidths=[3.0 * inch, 1.2 * inch, 6.8 * inch], rowHeights=[0.56 * inch] + [0.62 * inch] * 6)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CED8E4")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    table.wrapOn(c, 11 * inch, 4.5 * inch); table.drawOn(c, 1.15 * inch, 1.32 * inch)
    c.showPage()

    # 12
    _header(c, "Bounded conclusion and further reading", 12)
    _card(c, 0.8 * inch, 3.65 * inch, 5.75 * inch, 2.1 * inch, "What passed", "One Cartesian differential diagnostic on an analytic field family and one accepted Re=100 cavity case.", GREEN)
    _card(c, 6.75 * inch, 3.65 * inch, 5.75 * inch, 2.1 * inch, "What remains open", "Stretched and unstructured grids, turbulence, other solvers, arbitrary boundary treatments, and correctness of any AI system.", ORANGE)
    _paragraph(c, "References", 0.9 * inch, 2.82 * inch, 2 * inch, 0.38 * inch, 16, NAVY, bold=True)
    _bullet_block(c, [
        "Wilson et al. (2017), Good Enough Practices in Scientific Computing, doi:10.1371/journal.pcbi.1005510",
        "Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research, doi:10.1371/journal.pcbi.1003285",
        "Roache (1998), Verification and Validation in Computational Science and Engineering",
        "Stanford CS146S consulted only for the public syllabus-level topic of agent-driven development",
    ], 1.0 * inch, 1.05 * inch, 11.2 * inch, 1.7 * inch, 10.5)
    _paragraph(c, "Exit ticket: What new reference and gate would be required before making an unstructured-mesh claim?", 1.1 * inch, 0.62 * inch, 11.0 * inch, 0.42 * inch, 14, TEAL, bold=True, align=TA_CENTER)
    c.showPage()

    c.save()
    shutil.copy2(OUTPUT_PDF, LECTURE)
    return OUTPUT_PDF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    verification, cavity, record, case, diagnostic, faulty_error = _science()
    figure = build_figure(verification, record, case, diagnostic)
    notebook = build_notebook()
    if args.execute:
        execute_notebook(notebook)
    pdf = None if args.no_pdf else build_pdf(figure, verification, cavity, record, faulty_error)
    print("decision:", record["decision"])
    print("notebook:", notebook.relative_to(ROOT))
    print("figure:", figure.relative_to(ROOT))
    if pdf:
        print("pdf:", pdf.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
