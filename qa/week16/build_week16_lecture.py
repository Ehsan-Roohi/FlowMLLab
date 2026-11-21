"""Validate a Week 16 lecture build without replacing reviewed publication files.

The default compiles isolated copies of the reviewed TeX and figures. Explicit
--rebuild-figures reconstructs retained-data figures only in the isolated tree.
XeLaTeX and genuine Times New Roman are required; no font fallback is accepted.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lecture_typography import configure_matplotlib
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/week16_lowboom/reference"
ASSETS = ROOT / "lectures/source/week16_tex_assets"
TEX = ROOT / "lectures/source/week16_supersonic_shape_optimization.tex"
PDF = ROOT / "lectures/week16_supersonic_shape_optimization.pdf"


def isolated_directory(path):
    """Reject publication paths and checkout ancestors before any output write."""
    path = Path(path).expanduser().resolve()
    root = ROOT.resolve()
    if path == root or root.is_relative_to(path):
        raise ValueError("Output directory cannot be the checkout or its ancestor")
    if path.is_relative_to(root) and path.relative_to(root).parts[0] not in {"build", "tmp"}:
        raise ValueError("Inside the checkout, outputs belong only under build/ or tmp/")
    return path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contained_output(output_dir, relative):
    output_dir = isolated_directory(output_dir)
    candidate = output_dir / relative
    if candidate.is_symlink() or (candidate.is_file() and candidate.stat().st_nlink > 1):
        raise ValueError(f"Output file must not alias another file: {relative}")
    path = candidate.resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError(f"Output path escapes the isolated directory: {relative}")
    return path


def validate_isolated_tree(output_dir):
    """A reused build must not expose external files through links to TeX."""
    for path in output_dir.rglob("*"):
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
            raise ValueError(f"Linked file is not allowed in the isolated build tree: {path}")


def lecture_dependencies():
    """Collect this lecture's local TeX and image inputs, preserving their paths."""
    root = ROOT.resolve()
    found = set()
    pending = [TEX]
    while pending:
        source = pending.pop().resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Invalid lecture input: {source}")
        if source in found:
            continue
        found.add(source)
        text = source.read_text(encoding="utf-8")
        for name in re.findall(r"\\input\{([^}]+)\}", text):
            path = root / name
            if not path.suffix:
                path = path.with_suffix(".tex")
            pending.append(path)
        for name in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text):
            candidates = [root / name, ASSETS / name, R / name]
            image = next((p.resolve() for p in candidates if p.is_file()), None)
            if image is None or not image.is_relative_to(root):
                raise ValueError(f"Missing or external lecture figure: {name}")
            found.add(image)
    return sorted(found)


def stage_sources(output_dir):
    output_dir = isolated_directory(output_dir)
    stage = contained_output(output_dir, "source")
    dependencies = lecture_dependencies()
    for source in dependencies:
        destination = contained_output(output_dir, Path("source") / source.relative_to(ROOT.resolve()))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_hash(source) != file_hash(destination):
            raise RuntimeError(f"Staged source mismatch: {source}")
    return stage, dependencies


def build_figures(output_assets):
    # A local output path deliberately shadows the read-only publication folder.
    ASSETS = isolated_directory(output_assets)
    configure_matplotlib()
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
    ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,1.45),frameon=False)
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
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
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
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
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
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "pod_modes.png", dpi=220)
    plt.close(fig)

    # Taylor-Maccoll refinement.
    cone = json.loads((R / "cone_refinement_v801.json").read_text(encoding="utf-8"))
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
    audit = json.loads((R / "clean_model_audit_v801.json").read_text(encoding="utf-8"))
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
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
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
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.14),ncol=3,frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "finer_case_errors.png", dpi=220)
    plt.close(fig)


def compile_pdf(output_dir, source_tree):
    output_dir = isolated_directory(output_dir)
    source_tree = Path(source_tree).resolve()
    if not source_tree.is_relative_to(output_dir):
        raise ValueError("Compilation source tree must be inside the isolated output directory")
    validate_isolated_tree(output_dir)
    configure_matplotlib()
    xelatex = shutil.which("xelatex")
    if not xelatex:
        raise RuntimeError("Week 16 lecture validation requires XeLaTeX")
    tex_output = contained_output(output_dir, "tex")
    tex_output.mkdir(parents=True, exist_ok=True)
    stem = TEX.stem
    # Only this isolated build's disposable auxiliaries may be replaced.
    for suffix in (".aux", ".out", ".toc"):
        (tex_output / (stem + suffix)).unlink(missing_ok=True)
    for _ in range(2):
        subprocess.run([
            xelatex, "-interaction=nonstopmode", "-halt-on-error",
            f"-output-directory={tex_output}", str(TEX.relative_to(ROOT))
        ], cwd=source_tree, check=True)

    compiled = tex_output / (stem + ".pdf")
    if not compiled.is_file():
        raise RuntimeError("Week 16 lecture PDF was not generated.")

    # PDF must be a full lecture, not the old short handout.
    from pypdf import PdfReader
    pages = len(PdfReader(str(compiled)).pages)
    if pages < 20:
        raise RuntimeError(f"Week 16 lecture unexpectedly short: {pages} pages")

    destination = contained_output(output_dir, PDF.name)
    shutil.copyfile(compiled, destination)
    print(f"Validated isolated {destination} ({pages} pages)")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/week16-lecture",
                        help="Isolated build directory; publication paths are rejected")
    parser.add_argument("--rebuild-figures", action="store_true",
                        help="Regenerate retained-data figures only in the isolated source tree")
    args = parser.parse_args(argv)
    output_dir = isolated_directory(args.output_dir)
    dependencies = lecture_dependencies()
    numerical_inputs = [R / "clean_dataset_v801.npz", R / "cone_refinement_v801.json",
                        R / "clean_model_audit_v801.json"]
    protected = {p: file_hash(p) for p in [*dependencies, PDF, *numerical_inputs]}
    stage, _ = stage_sources(output_dir)
    if args.rebuild_figures:
        build_figures(stage / ASSETS.relative_to(ROOT))
    pdf = compile_pdf(output_dir, stage)
    for path, expected in protected.items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Reviewed input was modified: {path}")
    receipt = {
        "mode": "isolated retained-data figure regeneration" if args.rebuild_figures else "compile-only with reviewed figures",
        "builder_sha256": file_hash(Path(__file__)),
        "reviewed_inputs_unchanged": True,
        "reviewed_inputs": {p.relative_to(ROOT).as_posix(): value for p, value in protected.items()},
        "staged_inputs": {p.relative_to(ROOT).as_posix(): file_hash(stage / p.relative_to(ROOT)) for p in dependencies},
        "generated_pdf": pdf.name,
        "generated_pdf_sha256": file_hash(pdf),
        "publication_policy": "Build output is for validation/review only; no publication file is replaced.",
    }
    contained_output(output_dir, "build_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    return receipt


if __name__ == "__main__":
    main()
