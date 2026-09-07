#!/usr/bin/env python3
"""Build the original Week 1.1 evidence, notebook, and lecture PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np


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

        The assessed assignment requires a coding agent of your choice.
        Complete [PROCESS_LOG.md](PROCESS_LOG.md) with actual interaction records:

        - **Tool/model/date:**
        - **Exact specification supplied:**
        - **Functions or files proposed by the tool:**
        - **Material human corrections and physical reason:**
        - **Independent commands executed:**
        - **Failed gates retained:**
        - **Bounded claim:**

        Review every changed code and test line using [LINE_REVIEW.md](LINE_REVIEW.md).
        Cover axis order, units, signs, quadrature, boundaries, invalid inputs and
        reference independence. Record actual human corrections; never fabricate them.
        """
    )
    md(
        """
        ## 7. Required GitHub implementation assignment

        Follow [ASSIGNMENT.md](ASSIGNMENT.md). First complete and commit [SPEC.md](SPEC.md),
        then ask a coding agent to add `net_volume_flux(x, y, u, v)` in
        `flowmllab/student_mass_balance.py`. The function is your new contribution;
        running the instructor's reference notebook is only preparation.

        Run `python qa/check_week01_1_candidate.py flowmllab/student_mass_balance.py`
        from the repository root, then the existing scientific-software tests.
        Report unit, numerical-regression, physical-invariant and independent-reference
        evidence separately. Preserve failed attempts and the original AI patch.
        Submit the diff, specification-first commit, process log, line review and
        [REPORT.md](REPORT.md), titled **What the agent produced, what the physicist
        corrected, and why**, as a PR in your own fork or the instructor-designated repo.

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


def build_pdf(figure_path: Path, verification, cavity, record, faulty_error) -> Path:
    from week01_1_lecture import build_lecture
    return build_lecture(OUTPUT_PDF, LECTURE, verification, cavity, record, faulty_error, ROOT)


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
