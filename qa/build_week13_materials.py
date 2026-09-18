"""Build the Week 13 research lecture and audit notebook from retained results.

This builder never trains a PINN.  It requires all four selected Unity result
directories and fails closed if a machine-readable audit or figure is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import re
import shutil
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "lectures/source/week13_rectangular_cavity_pinn.md"
RESULTS = ROOT / "results/week13_rectangular_pinn"
CASES = ((100, 1), (400, 1), (100, 2), (400, 2))
NAME = "week13_rectangular_cavity_pinn"

pdfmetrics.registerFont(TTFont("FlowSerif", font_manager.findfont("DejaVu Serif")))
pdfmetrics.registerFont(TTFont("FlowSerifBold", font_manager.findfont(
    font_manager.FontProperties(family="DejaVu Serif", weight="bold"))))

EQUATIONS = {
    "mapping": [r"$D=H/L,\qquad x=X/L,\qquad \eta=Y/H,\qquad y/L=D\eta$",
                r"$Re=UL/\nu,\qquad \partial_y=D^{-1}\partial_\eta,\qquad \partial_{yy}=D^{-2}\partial_{\eta\eta}$"],
    "ns": [r"$u u_x+v u_y+p_x-Re^{-1}(u_{xx}+u_{yy})=0$",
           r"$u v_x+v v_y+p_y-Re^{-1}(v_{xx}+v_{yy})=0,\qquad u_x+v_y=0$"],
    "streamfunction": [r"$u=\psi_y,\qquad v=-\psi_x,\qquad \nabla\!\cdot\mathbf{u}=\psi_{yx}-\psi_{xy}=0$",
                       r"$\omega_z=v_x-u_y=-\nabla^2\psi$"],
    "rectangular": [r"$u=D^{-1}\psi_\eta,\qquad v=-\psi_x$",
                    r"$r_x=u u_x+D^{-1}v u_\eta+p_x-Re^{-1}(u_{xx}+D^{-2}u_{\eta\eta})$",
                    r"$r_y=u v_x+D^{-1}v v_\eta+D^{-1}p_\eta-Re^{-1}(v_{xx}+D^{-2}v_{\eta\eta})$"],
    "lifting": [r"$B=16x(1-x)\eta(1-\eta),\qquad \psi_\theta=\psi_{lid}+B^2q_\theta$",
                r"$\psi_{lid}=D(\eta-1)\eta^2g(x)\exp[-(1-\eta)^2/\delta_\eta^2]$",
                r"$g(x)=(1-e^{-(1-x)^2/\delta_x^2})(1-e^{-x^2/\delta_x^2})$"],
    "objective": [r"$\mathcal{L}_{train}=\langle \widetilde r_x^2+\widetilde r_y^2\rangle_{\mathcal{C}},\qquad \widetilde r_i=m r_i$",
                  r"$\theta_0\longrightarrow\theta_A\longrightarrow\theta_Q\qquad[\mathrm{Adam};\ \mathrm{SSBroyden2}]$"],
    "errors": [r"$E_{\mathbf{u}}=\|\mathbf{u}_\theta-\mathbf{u}_{ref}\|_{2,\Omega'}/\|\mathbf{u}_{ref}\|_{2,\Omega'}$",
               r"$R_{full}=\sqrt{\langle r_x^2+r_y^2\rangle_{\mathcal{T}}},\qquad \mathcal{T}\cap\mathcal{C}=\varnothing$"],
}


def load_cases():
    records = []
    for re_value, depth in CASES:
        directory = RESULTS / f"re{re_value}-d{depth}"
        required = ("audit.json", "optimizer-history.jsonl", "fields.png", "loss.png", "environment-lock.txt")
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{directory}: missing {missing}")
        audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
        history = [json.loads(line) for line in (directory / "optimizer-history.jsonl").read_text(encoding="utf-8").splitlines() if line]
        records.append({"re": re_value, "depth": depth, "dir": directory,
                        "audit": audit, "history": history})
    return records


def result_rows(records):
    rows = [["Re", "D", "claim boundary", "full momentum RMS", "corner momentum RMS", "CFD field gates"]]
    for record in records:
        audit = record["audit"]; residual = audit["independent_residual"]
        full = np.hypot(residual["momentum_x_rms"], residual["momentum_y_rms"])
        corner = np.sqrt((residual["top_corner_momentum_x_rms"]**2
                          + residual["top_corner_momentum_y_rms"]**2))
        compare = audit.get("cfd_comparison")
        gate = "n/a: no raw matched field" if compare is None else ("pass" if compare["all_pass"] else "fail")
        claim = ("square: CFD gates pass" if audit["claim_status"] == "field-qualified-square-case"
                 else "deep: no field reference")
        rows.append([str(record["re"]), str(record["depth"]), claim,
                     f"{full:.3g}", f"{corner:.3g}", gate])
    return rows


def build_pdf(records, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    width = A4[0] - 104
    styles = {
        "body": ParagraphStyle("body", fontName="FlowSerif", fontSize=9.7, leading=12.8,
                               alignment=4, spaceAfter=5.5),
        "title": ParagraphStyle("title", fontName="FlowSerifBold", fontSize=23, leading=27,
                                textColor=colors.HexColor("#17384d"), spaceAfter=8, keepWithNext=True),
        "subtitle": ParagraphStyle("subtitle", fontName="FlowSerif", fontSize=16, leading=20,
                                   spaceAfter=11, keepWithNext=True),
        "heading": ParagraphStyle("heading", fontName="FlowSerifBold", fontSize=12.5, leading=15.5,
                                  textColor=colors.HexColor("#17384d"), spaceBefore=9, spaceAfter=5,
                                  keepWithNext=True),
        "caption": ParagraphStyle("caption", fontName="FlowSerif", fontSize=8.8, leading=11.1,
                                  spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="FlowSerif", fontSize=8.0, leading=9.8),
        "reference": ParagraphStyle("reference", fontName="FlowSerif", fontSize=7.7, leading=9.3,
                                    spaceAfter=2.5),
    }

    def para(text, style="body"):
        value = escape(text)
        value = re.sub(r"(https://[^\s]+)", r'<link href="\1" color="#14648a">\1</link>', value)
        return Paragraph(value, styles[style])

    def raster(fig, max_height=245):
        stream = BytesIO()
        fig.savefig(stream, format="png", dpi=300, bbox_inches="tight", pad_inches=.08, facecolor="white")
        plt.close(fig); stream.seek(0)
        image = Image(stream)
        scale = min(width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth *= scale; image.drawHeight *= scale
        return image

    def equation(key, number):
        lines = EQUATIONS[key]
        fig = plt.figure(figsize=(7.0, .47 * len(lines) + .18))
        for index, line in enumerate(lines):
            fig.text(.5, 1 - (index + .55) / len(lines), line, ha="center", va="center", fontsize=13.2)
        image = raster(fig, max_height=34 * len(lines))
        table = Table([[image, para(f"({number})", "cell")]], colWidths=[width - 28, 28])
        table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("ALIGN", (0, 0), (0, 0), "CENTER"),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        return table

    tables = {
        "representations": [["Representation", "Structural property", "Numerical cost / risk"],
            ["primitive u,v,p", "continuity and walls are explicit residuals or transforms", "coupled loss; second derivatives"],
            ["primitive FOSLS", "four derivative auxiliaries lower base-network derivative order", "seven outputs; compatibility residuals"],
            ["streamfunction psi,p", "continuity exact for smooth psi", "third derivatives of psi; pressure remains coupled"]],
        "literature": [["Study", "Tested space", "Result relevant here"],
            ["McDevitt et al. (2022)", "Re and geometry-parametric cavity", "motivates hard streamfunction representation"],
            ["Służalec et al. (2026)", "square; Re 1,10,100,1000; FOSLS PINN/CRVPINN", "reports unacceptable error from Re=100"],
            ["Cheng & Hung (2006)", "rectangular D=0.1--7; Re=0.01--5000", "topology changes jointly with Re and D"]],
        "restart": [["Restart state", "Why it is retained"],
            ["model weights", "represented field"], ["Adam moments / quasi-Newton H and x", "optimizer trajectory"],
            ["collocation points", "deterministic objective"], ["CPU and GPU RNG", "reproducible continuation"],
            ["config, commit, digest, job id", "provenance and incompatibility guard"]],
        "gates": [["Square-case metric", "Frozen threshold", "Role"],
            ["u(0.5,y) relative L2", "10%", "primary recirculation profile"],
            ["v(x,0.5) relative L2", "15%", "cross-stream profile"],
            ["interior velocity-vector relative L2", "15%", "8192 seeded points"],
            ["full/corner residual and walls", "reported, not tuned", "failure localization"]],
        "results": result_rows(records),
        "paperdesign": [["Factor", "Pilot", "Paper requirement"],
            ["representation", "streamfunction only", "streamfunction vs primitive/FOSLS"],
            ["optimizer", "one staged path", "Adam-only vs staged at matched budget"],
            ["Re and D", "2 by 2 feasibility matrix", "cross failure region and topology transitions"],
            ["initialization", "one seed", "multiple preregistered seeds; retain failures"],
            ["reference", "near-matched square CFD", "grid-converged smooth-lid CFD/FEM for every case"]],
    }

    def make_table(key):
        data = [[para(str(value), "cell") for value in row] for row in tables[key]]
        columns = len(data[0]); widths = [width / columns] * columns
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0f4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17384d")),
            ("LINEABOVE", (0, 0), (-1, 0), .7, colors.HexColor("#29475b")),
            ("LINEBELOW", (0, 0), (-1, 0), .55, colors.HexColor("#29475b")),
            ("LINEBELOW", (0, -1), (-1, -1), .7, colors.HexColor("#29475b")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        return KeepTogether([table, Spacer(1, 8)])

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    def figure(key):
        if key in ('deep_fields', 'deep_loss', 'deep_streamfunction'):
            files={'deep_fields':'fields.png','deep_loss':'loss_continuation.png','deep_streamfunction':'streamfunction.png'}
            captions={
                'deep_fields':'Deep-cavity retained field: Re=1000, D/W=2.2, checkpoint 55118. Speed with streamlines, mean-zero pressure and vorticity. Symmetric-log scales retain the full pressure/vorticity range; equal physical aspect ratio is preserved.',
                'deep_loss':'Observed continuation training MSEs through checkpoint 65711. The field shown above is checkpoint 55118. Logged steps are local to the resumed process, not a from-scratch history. These curves are not independent test loss or CFD error.',
                'deep_streamfunction':'Integrated streamfunction and lower-cavity detail from checkpoint 55118. Candidate recirculation features are field diagnostics, not independently validated vortex centres.'}
            path=ROOT/'results/week13_deep_cavity'/files[key]
            pixels=plt.imread(path); h,w=pixels.shape[:2]
            scale=min(490/w,350/h)
            return KeepTogether([Image(str(path),width=w*scale,height=h*scale),Spacer(1,4),para(captions[key],'caption')])
        if key == "geometry":
            fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), constrained_layout=True)
            for ax, depth in zip(axes, (1, 2)):
                ax.add_patch(plt.Rectangle((0, 0), 1, depth, fc="#eef4f6", ec="#17384d", lw=1.5))
                ax.arrow(.18, depth + .08, .64, 0, width=.012, head_width=.07, color="#D55E00",
                         length_includes_head=True)
                ax.text(.5, depth + .18, "moving regularized lid", ha="center", fontsize=9)
                ax.set(xlim=(-.12, 1.12), ylim=(-.12, depth + .35), aspect="equal",
                       xlabel="x = X/L", ylabel="y/L", title=f"D = H/L = {depth}")
            caption = "Figure 1. Width-based nondimensionalization for the square and deep cavities. The network coordinates are (x, eta), while physical y/L=D eta. Equal data scales preserve geometry."
        elif key == "lid":
            x = np.linspace(0, 1, 1601); delta = np.sqrt(1e-3)
            g = (1 - np.exp(-(1-x)**2/delta**2)) * (1 - np.exp(-x*x/delta**2))
            fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.7), constrained_layout=True)
            axes[0].plot(x, g, color="#0072B2", lw=1.8); axes[0].plot(x, np.ones_like(x), "--", c=".55")
            axes[1].plot(x, g, color="#0072B2", lw=1.8); axes[1].plot(x, np.ones_like(x), "--", c=".55")
            axes[0].set(xlim=(0, 1), title="whole lid"); axes[1].set(xlim=(0, .13), title="corner detail")
            for ax in axes: ax.set(xlabel="x/L", ylabel="u_lid/U", ylim=(-.03, 1.06)); ax.grid(alpha=.2)
            caption = "Figure 2. Smooth lid profile used by the hard lifting (blue) versus an ideal unit lid (gray). The regularization width is part of the boundary-value problem."
        elif key == "losses":
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
            for ax, record in zip(axes.ravel(), records):
                latest = {row["global_step"]: row for row in record["history"]}
                rows = [latest[key] for key in sorted(latest)]
                x = [row["global_step"] for row in rows]
                train = [np.sqrt(row["train_rx_mse"] + row["train_ry_mse"]) for row in rows]
                ax.semilogy(x, train, color="#0072B2", label="masked train")
                ax.semilogy(x, [row["heldout_momentum_rms"] for row in rows], "--", color="black", label="held-out full")
                ax.semilogy(x, [np.sqrt(2)*row["top_corner_momentum_rms"] for row in rows], ":", color="#009E73", label="top corners (vector RMS)")
                ax.axvline(1000, color="#D55E00", lw=.9, alpha=.8)
                ax.set(title=f"Re={record['re']}, D={record['depth']}", xlabel="optimizer step", ylabel="residual RMS")
                ax.grid(which="both", alpha=.2)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.savefig(RESULTS / "week13_optimizer_history.png", dpi=260,
                        bbox_inches="tight", facecolor="white")
            caption = "Figure 3. Adam (left of the orange line) followed by SSBroyden2. Held-out full-domain and top-corner residuals prevent masked training loss from being mistaken for global accuracy."
        else:
            # Each retained render is a wide three-panel strip.  Match the
            # montage canvas to that shape so ReportLab does not inherit large
            # bands of empty axes space between the two rows.
            fig, axes = plt.subplots(2, 2, figsize=(8.2, 4.4), constrained_layout=True)
            for ax, record in zip(axes.ravel(), records):
                field = plt.imread(record["dir"] / "fields.png")
                rgb = field[..., :3]
                content = np.any(rgb < .985, axis=2)
                rows, cols = np.where(content)
                if rows.size:
                    pad = 8
                    r0, r1 = max(0, rows.min()-pad), min(field.shape[0], rows.max()+pad+1)
                    c0, c1 = max(0, cols.min()-pad), min(field.shape[1], cols.max()+pad+1)
                    field = field[r0:r1, c0:c1]
                ax.imshow(field); ax.axis("off"); ax.set_title(f"Re={record['re']}, D={record['depth']}", fontsize=10)
            fig.savefig(RESULTS / "week13_matrix.png", dpi=260,
                        bbox_inches="tight", facecolor="white")
            caption = "Figure 4. Retained case renders from the exact Unity jobs. These panels preserve geometric aspect ratio inside each source render. Square cases have CFD gates; deep cases are residual-audited hypotheses, not field validations."
        image = raster(fig, max_height=(325 if key == "losses" else 225) if key in ("losses", "fields") else (165 if key == "geometry" else 220))
        return KeepTogether([image, Spacer(1, 4), para(caption, "caption")])

    story = []; equation_number = 0; references = False
    for block in SOURCE.read_text(encoding="utf-8").strip().split("\n\n"):
        if block.startswith("@equation "):
            equation_number += 1; story.append(equation(block.split()[1], equation_number))
        elif block.startswith("@table "):
            story.append(make_table(block.split()[1]))
        elif block.startswith("@figure "):
            story.append(figure(block.split()[1]))
        elif block.startswith("### "):
            references = block[4:].strip() == "References and provenance"
            story.append(para(block[4:], "heading"))
        elif block.startswith("## "):
            story.append(para(block[3:], "subtitle"))
        elif block.startswith("# "):
            story.append(para(block[2:], "title"))
        else:
            story.append(para(block.replace("\n", " "), "reference" if references else "body"))

    def footer(canvas, document):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#b7c3cd")); canvas.setLineWidth(.4)
        canvas.line(52, 43, A4[0]-52, 43); canvas.setFont("FlowSerif", 8.5)
        canvas.drawString(52, 29, "FlowMLLab | Week 13 | Rectangular-cavity PINNs")
        canvas.drawRightString(A4[0]-52, 29, str(document.page)); canvas.restoreState()

    def canvas(*args, **kwargs):
        kwargs["invariant"] = 1; return Canvas(*args, **kwargs)

    target = outdir / f"{NAME}.pdf"
    SimpleDocTemplate(str(target), pagesize=A4, leftMargin=52, rightMargin=52,
        topMargin=45, bottomMargin=58,
        title="Physics-Informed Neural Networks for Rectangular Cavity Flow",
        author="Ehsan Roohi").build(story, onFirstPage=footer, onLaterPages=footer, canvasmaker=canvas)
    return target


def build_notebook(target):
    """Delegate to build_week13_notebook.py, which assembles the notebook in teaching order."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_week13_notebook import build  # noqa: PLC0415
    return build(target, keep_outputs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/pdf")
    parser.add_argument("--publish-copy", action="store_true")
    args = parser.parse_args()
    records = load_cases()
    pdf = build_pdf(records, args.output_dir)
    notebook = build_notebook(ROOT / "notebooks/week13/W13_Rectangular_Cavity_PINN_Research.ipynb")
    manifest = {"builder": str(Path(__file__).relative_to(ROOT)), "source": str(SOURCE.relative_to(ROOT)),
                "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                "case_audits": {f"re{r['re']}-d{r['depth']}": hashlib.sha256((r['dir']/"audit.json").read_bytes()).hexdigest() for r in records}}
    (args.output_dir / f"{NAME}_build.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.publish_copy:
        shutil.copy2(pdf, ROOT / "lectures" / pdf.name)
    print(pdf); print(notebook)


if __name__ == "__main__":
    main()
