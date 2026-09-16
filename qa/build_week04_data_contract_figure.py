"""Build the Week 4 operator-learning data-contract figure for the README."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pod_deeponet" / "week04_data_contract.png"


def box(ax, xy, width, height, text, face, edge="#17324D", size=10):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=size)


def arrow(ax, start, end, color="#355C7D"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.6, color=color))


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    # (a) Generic operator-learning contract.
    ax = axes[0]
    ax.set_title("(a) Branch sensors, trunk queries, targets", fontsize=13, weight="bold")
    box(ax, (0.05, 0.58), 0.25, 0.26, "Branch input\n$f(s_1),…,f(s_m)$", "#DDEBF7")
    for x in np.linspace(0.055, 0.235, 7):
        ax.plot(x, 0.63, "o", ms=4, color="#0072B2")
    box(ax, (0.05, 0.16), 0.25, 0.23, "Trunk query\n$y_i=(x_i,y_i)$", "#FFF2CC")
    box(ax, (0.39, 0.36), 0.23, 0.28, "paired sample\n$(f_j,y_i)$", "#E2F0D9")
    box(ax, (0.73, 0.36), 0.24, 0.28, "Target\n$G(f_j)(y_i)$", "#FCE4D6")
    arrow(ax, (0.30, 0.71), (0.39, 0.54))
    arrow(ax, (0.30, 0.275), (0.39, 0.46))
    arrow(ax, (0.62, 0.50), (0.73, 0.50))
    ax.text(0.50, 0.08, "case ID + physical coordinate must follow every target",
            ha="center", color="#8B1E1E", fontsize=9.5, weight="bold")

    # (b) Model hierarchy taught in Week 4.
    ax = axes[1]
    ax.set_title("(b) Three different surrogate contracts", fontsize=13, weight="bold")
    rows = [
        (0.72, "Pointwise MLP", "$(Re,x,y) \\rightarrow q$", "one queried point"),
        (0.43, "Week 4 model", "$Re \\rightarrow$ POD coefficients $\\rightarrow q(x,y)$", "parameter to field"),
        (0.14, "Full DeepONet", "$f(s_1),…,f(s_m);\\ y \\rightarrow G(f)(y)$", "function to function"),
    ]
    colors = ["#EDEDED", "#D9EAD3", "#DDEBF7"]
    for (y, title, mapping, scope), face in zip(rows, colors):
        box(ax, (0.03, y), 0.29, 0.18, title, face, size=10.5)
        arrow(ax, (0.33, y + 0.09), (0.43, y + 0.09))
        box(ax, (0.44, y), 0.52, 0.18, mapping + "\n" + scope, "white", size=9.5)
    ax.text(0.50, 0.025, "Same word ‘DeepONet’ can hide different input/output contracts.",
            ha="center", fontsize=9.5, color="#555555")

    # (c) Executable audit and intentional negative control.
    ax = axes[2]
    ax.set_title("(c) data_alignment_audit before fit", fontsize=13, weight="bold")
    checks = [
        ("branch case ID = target case ID", True),
        ("trunk coordinate = target coordinate", True),
        ("branch sensors fixed across cases", True),
        ("train / validation / test cases disjoint", True),
    ]
    y = 0.78
    for label, passed in checks:
        ax.add_patch(Rectangle((0.06, y - 0.035), 0.055, 0.055, facecolor="#2E7D32", edgecolor="none"))
        ax.text(0.0875, y - 0.007, "OK", color="white", ha="center", va="center", fontsize=7.5, weight="bold")
        ax.text(0.14, y - 0.007, label, va="center", fontsize=9.5)
        y -= 0.135
    box(ax, (0.08, 0.035), 0.84, 0.16,
        "Intentional negative control: shuffle one trunk block\nAudit must fail before training starts",
        "#F4CCCC", edge="#A61C00", size=10)
    ax.text(0.50, 0.255, "PASS: aligned data  •  FAIL: shape-correct physical mismatch",
            ha="center", fontsize=9.5, weight="bold", color="#17324D")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle("Week 4 — An explicit data contract for operator learning", fontsize=17, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
