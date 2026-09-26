"""Build the Week 16 educational LaTeX lecture from retained numerical evidence.

This generator intentionally uses line/scatter plots rather than bar charts. The PDF
is compiled with XeLaTeX; Times New Roman is used when installed, with Tinos as
the metric-compatible fallback on Linux CI.
"""
from pathlib import Path
import json
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/week16_lowboom/reference"
ASSETS = ROOT / "lectures/source/week16_tex_assets"
TEX = ROOT / "lectures/source/week16_supersonic_shape_optimization.tex"
PDF = ROOT / "lectures/week16_supersonic_shape_optimization.pdf"


def build_figures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = np.load(R / "clean_dataset_v801.npz", allow_pickle=False)

    # Mach cone schematic.
    M = 1.8
    mu = np.arcsin(1.0 / M)
    x = np.linspace(0.0, 6.0, 200)
    y = np.tan(mu) * (6.0 - x)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([0, 6], [0, 0], lw=1)
    ax.plot(x, y, lw=2)
    ax.plot(x, -y, lw=2)
    ax.scatter([6], [0], s=80, zorder=3)
    ax.annotate("supersonic body", xy=(6, 0), xytext=(4.8, 0.75),
                arrowprops=dict(arrowstyle="->"))
    ax.text(2.4, 0.55, rf"$\mu=\sin^{{-1}}(1/M)={np.degrees(mu):.1f}^\circ$")
    ax.set_xlim(0, 6.4)
    ax.set_ylim(-2.7, 2.7)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "mach_cone.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Conceptual N-wave.
    xi = np.linspace(0, 1, 500)
    p = np.zeros_like(xi)
    mid = (xi >= 0.15) & (xi <= 0.85)
    p[mid] = 0.9 - 1.8 * (xi[mid] - 0.15) / 0.70
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(xi, p, lw=2)
    ax.axhline(0, lw=0.8)
    ax.axvline(0.15, ls="--", lw=0.8)
    ax.axvline(0.85, ls="--", lw=0.8)
    ax.set_xlabel("retarded time")
    ax.set_ylabel("overpressure")
    ax.set_title("Conceptual far-field N-wave")
    ax.text(0.18, 0.72, "compression shock")
    ax.text(0.60, -0.72, "expansion region")
    fig.tight_layout()
    fig.savefig(ASSETS / "n_wave.png", dpi=220)
    plt.close(fig)

    # Fixed-volume geometry family.
    s = np.linspace(0, 1, 1500)
    volume = np.pi * 0.06**2 / 2

    def radius(a, b):
        f = np.sin(np.pi * s) * np.exp(a * (2 * s - 1) + b * np.cos(2 * np.pi * s))
        c = np.sqrt(volume / (np.pi * np.trapezoid(f * f, s)))
        return c * f

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for a, b, label in [
        (0.0, 0.0, "baseline"),
        (0.3249532654588658, 0.12237788693256269, "retained candidate"),
        (-0.25, 0.12, "example"),
    ]:
        rr = radius(a, b)
        line, = ax.plot(s, rr, label=label)
        ax.plot(s, -rr, color=line.get_color())
    ax.set_aspect("equal")
    ax.set_xlabel("x/L")
    ax.set_ylabel("r/L")
    ax.legend(ncol=3)
    ax.set_title("Fixed-volume two-parameter body family")
    fig.tight_layout()
    fig.savefig(ASSETS / "geometry_family.png", dpi=220)
    plt.close(fig)

    # Split map.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    markers = {"train": "o", "validation": "s", "test": "^", "extrapolation": "x"}
    for split, marker in markers.items():
        mask = data["splits"] == split
        ax.scatter(data["parameters"][mask, 0], data["parameters"][mask, 1],
                   marker=marker, s=45, label=split)
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title("Frozen geometry split")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "split_map.png", dpi=220)
    plt.close(fig)

    # Representative pressure waveforms.
    fig, ax = plt.subplots(figsize=(8, 4))
    for split in ["train", "validation", "test", "extrapolation"]:
        ids = np.where(data["splits"] == split)[0]
        i = ids[len(ids) // 2]
        ax.plot(data["x"], data["waveforms"][i], label=split)
    ax.set_xlabel("x/L")
    ax.set_ylabel("Cp")
    ax.set_title("Representative near-field CFD signatures")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "waveforms_by_split.png", dpi=220)
    plt.close(fig)

    # Training-only POD.
    train = data["splits"] == "train"
    Y = data["waveforms"][train]
    mean = Y.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(Y - mean, full_matrices=False)
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(1, len(singular_values) + 1), energy, "o-")
    ax.axvline(12, ls="--", lw=1)
    ax.set_ylim(0, 1.005)
    ax.set_xlabel("number of POD modes")
    ax.set_ylabel("cumulative retained energy")
    ax.set_title("Training-only POD compression")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "pod_energy.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(3):
        ax.plot(data["x"], vt[k], label=f"mode {k + 1}")
    ax.set_xlabel("x/L")
    ax.set_ylabel("POD mode amplitude")
    ax.set_title("First three pressure-signature POD modes")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "pod_modes.png", dpi=220)
    plt.close(fig)

    # Taylor-Maccoll refinement.
    cone = json.loads((R / "cone_refinement_v801.json").read_text())
    cells = [row["mesh"]["cells"] for row in cone["levels"]]
    errors = [100 * row["analytical_comparison"]["relative_error"] for row in cone["levels"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(cells, errors, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("fluid cells")
    ax.set_ylabel("cone Cp error [%]")
    ax.set_title("Taylor-Maccoll verification under mesh refinement")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "cone_refinement.png", dpi=220)
    plt.close(fig)

    # Clean retained model: lines, no bars.
    audit = json.loads((R / "clean_model_audit_v801.json").read_text())
    splits = ["train", "validation", "test", "extrapolation"]
    xx = range(len(splits))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for key, label in [
        ("wave_relative_l2", "wave L2"),
        ("peak_mean_relative_error", "mean peak error"),
        ("drag_mean_relative_error", "mean drag error"),
    ]:
        yy = [100 * audit["same_mesh"][s][key] for s in splits]
        ax.plot(xx, yy, "o-", label=label)
    ax.set_xticks(list(xx), splits)
    ax.set_ylabel("relative error [%]")
    ax.set_title("Clean retained model: interpolation vs extrapolation")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "model_metrics_lines.png", dpi=220)
    plt.close(fig)

    errs = [100 * value for value in audit["finer_mesh"]["per_case_wave_relative_l2"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(range(1, len(errs) + 1), errs, "o-")
    ax.axhline(10, ls="--", lw=1, label="10% reference line")
    ax.set_xlabel("finer-mesh test case")
    ax.set_ylabel("waveform relative L2 error [%]")
    ax.set_title("Finer-CFD audit: individual waveform errors")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "finer_case_errors.png", dpi=220)
    plt.close(fig)


def compile_pdf():
    for _ in range(2):
        subprocess.run([
            "xelatex", "-interaction=nonstopmode", "-halt-on-error",
            "-output-directory=lectures", str(TEX)
        ], cwd=ROOT, check=True)

    if not PDF.is_file():
        raise RuntimeError("Week 16 lecture PDF was not generated.")

    # PDF must be a full lecture, not the old short handout.
    from pypdf import PdfReader
    pages = len(PdfReader(str(PDF)).pages)
    if pages < 20:
        raise RuntimeError(f"Week 16 lecture unexpectedly short: {pages} pages")

    for suffix in [".aux", ".log", ".out", ".toc"]:
        p = ROOT / "lectures" / ("week16_supersonic_shape_optimization" + suffix)
        p.unlink(missing_ok=True)

    print(f"Built {PDF} ({pages} pages)")


if __name__ == "__main__":
    build_figures()
    compile_pdf()
