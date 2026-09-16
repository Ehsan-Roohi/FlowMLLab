"""Build the Week 9 moving-throat data-alignment audit figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "nozzle_alignment_audit" / "nozzle_data_alignment_audit.png"
OUT_SVG = OUT.with_suffix(".svg")
GAMMA = 1.4
SENSORS = np.linspace(0.0, 1.0, 41)
CASES = [
    (0.35, 2.0, "A"),
    (0.40, 2.5, "B"),
    (0.46, 3.5, "C"),
]


def area(x: np.ndarray, throat: float, exit_ratio: float) -> np.ndarray:
    x = np.asarray(x)
    return np.where(
        x <= throat,
        1 + 0.6 * (1 + np.cos(np.pi * x / throat)),
        1 + 0.5 * (exit_ratio - 1) * (1 - np.cos(np.pi * (x - throat) / (1 - throat))),
    )


def area_mach(mach: float | np.ndarray) -> float | np.ndarray:
    return (
        2 / (GAMMA + 1) * (1 + (GAMMA - 1) * np.asarray(mach) ** 2 / 2)
    ) ** ((GAMMA + 1) / (2 * (GAMMA - 1))) / np.asarray(mach)


def reference(throat: float, exit_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_left = round(40 * throat) + 1
    x = np.r_[np.linspace(0, throat, n_left), np.linspace(throat, 1, 42 - n_left)[1:]]
    cross_section = area(x, throat, exit_ratio)
    mach = np.array(
        [
            1.0
            if abs(xj - throat) < 1e-12
            else brentq(
                lambda value: area_mach(value) - aj,
                *((1e-7, 1.0) if xj < throat else (1.0, 10.0)),
            )
            for xj, aj in zip(x, cross_section)
        ]
    )
    return x, cross_section, mach


def main() -> None:
    records = [reference(throat, ratio) for throat, ratio, _ in CASES]
    wrong_x = np.tile(records[0][0], (3, 1))
    correct_x = np.stack([record[0] for record in records])
    displacement = np.abs(wrong_x - correct_x)
    assert displacement.max() > 0
    for _, cross_section, mach in records:
        residual = np.max(np.abs(area_mach(mach) - cross_section))
        mass_flux = cross_section * mach * (
            1 + (GAMMA - 1) * mach**2 / 2
        ) ** (-(GAMMA + 1) / (2 * (GAMMA - 1)))
        assert residual < 1e-9
        assert mass_flux.std() / mass_flux.mean() < 1e-12

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]

    ax = axes[0]
    for (throat, ratio, label), color in zip(CASES, colors):
        ax.plot(SENSORS, area(SENSORS, throat, ratio), color=color, lw=3.0,
                label=rf"case {label}: $x_t/L={throat:.2f}$")
        ax.plot(SENSORS, area(SENSORS, throat, ratio), ".", color=color, ms=4.5, alpha=0.70)
    ax.set(title="(a) Geometry sampled at fixed sensors", xlabel=r"sensor position, $x/L$", ylabel=r"area ratio, $A/A_t$")
    ax.legend(frameon=True, fontsize=12, loc="upper left", facecolor="white", framealpha=0.94)
    ax.grid(alpha=0.25)

    ax = axes[1]
    index = np.arange(displacement.shape[1])
    ax.fill_between(index, 0, 1e3 * displacement[2], color="#E69F00", alpha=0.22)
    ax.plot(index, 1e3 * displacement[2], "o-", color="#D55E00", lw=2.5,
            ms=4.5, label="case C target vs reused case-A trunk")
    ax.axhline(0, color="0.25", lw=1.3)
    ax.set(title="(b) Coordinate mismatch detected", xlabel="query index",
           ylabel=r"coordinate error, $10^3|x_{reused}-x_{target}|/L$")
    ax.legend(frameon=True, fontsize=11.5, loc="upper left", facecolor="white", framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.text(0.97, 0.08, f"maximum error = {displacement.max():.3f} L\ntraining blocked before fit",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12.5, weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF2CC", "edgecolor": "#C87900"})

    ax = axes[2]
    for (throat, _, label), (_, _, mach), color in zip(CASES, records, colors):
        ax.plot(correct_x[ord(label) - ord("A")], mach, color=color, lw=3.0, label=f"case {label}")
        ax.plot(throat, 1.0, "o", color=color, ms=8)
    ax.axhline(1.0, color="0.25", lw=1.3, ls="--")
    ax.set(title="(c) Repaired pairs pass physics gates", xlabel=r"target coordinate, $x/L$", ylabel="Mach number")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True, fontsize=12, loc="upper left", facecolor="white", framealpha=0.94)
    ax.text(0.97, 0.06,
            "PASS: alignment = 0\nmass-flow CV < 10⁻¹²\nsonic error < 10⁻¹²\narea–Mach residual < 10⁻⁹",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#E8F5E9", "edgecolor": "#2E7D32", "linewidth": 1.4})

    fig.suptitle("Week 9 Lab 3 — Moving-throat nozzle data-alignment audit", fontsize=20, weight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor="white")
    fig.savefig(OUT_SVG, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
