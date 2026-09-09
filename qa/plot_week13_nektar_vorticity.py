"""Plot Cheng--Hung-style clockwise vorticity contours from Nektar++ VTU.

The plotted quantity is omega_c = du/dy - dv/dx, matching the sign convention
visible in Cheng & Hung (2006), Fig. 10(b'). Derivatives are second-order finite
differences on the boundary-inclusive visualization grid; they are not Nektar++
spectral derivatives and must be described as such.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_week13_nektar import structured
from check_week13_nektar_vtu import read_vtu


# Levels transcribed from the published D/W=5, Re=100 panel and completed
# symmetrically at the very weak magnitudes so both rotation senses remain visible.
PAPER_LEVELS = np.array([
    -5.0, -3.0, -1.0, -0.5, -0.1, -0.03, -0.01, -0.001, -3e-4,
    -1e-4, -5e-5, -1e-5, -5e-6, -1e-6, -5e-7, -3e-7, -1e-7,
    -5e-8, -1e-8, 0.0, 1e-8, 5e-8, 1e-7, 3e-7, 5e-7, 1e-6,
    5e-6, 1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 0.01, 0.03, 0.1, 0.5,
    1.0, 3.0, 5.0,
])


def clockwise_vorticity(x, y, u, v):
    return (
        np.gradient(u, y, axis=0, edge_order=2)
        - np.gradient(v, x, axis=1, edge_order=2)
    )


def plot(vtu, output):
    x, y, fields, _ = structured(read_vtu(vtu))
    omega = clockwise_vorticity(x, y, fields["u"], fields["v"])
    levels = PAPER_LEVELS[(PAPER_LEVELS >= omega.min()) & (PAPER_LEVELS <= omega.max())]

    fig, ax = plt.subplots(figsize=(3.45, 10.2), constrained_layout=True)
    cs = ax.contour(
        x, y, omega, levels=levels, colors="#111827", linewidths=0.55,
        linestyles="solid",
    )
    # Match the paper's labelled black contours while keeping labels legible.
    label_levels = levels[np.isin(levels, [
        -5, -1, -0.5, -0.1, -0.03, -0.01, -0.001, -1e-4, -1e-5,
        -1e-6, -1e-7, -1e-8, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
        1e-3, 0.01, 0.1, 0.5, 1, 3, 5,
    ])]
    ax.clabel(cs, levels=label_levels, inline=True, fontsize=5.2, fmt="%g")
    ax.set(xlim=(0, 1), ylim=(0, 5), xlabel=r"$x/W$", ylabel=r"$y/W$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(r"Nektar++: $Re=100$, $D/W=5$, $tU/W=80$", fontsize=10)
    ax.text(
        0.03, 0.02,
        r"Contours of $\omega_c W/U$; $\omega_c=\partial u/\partial y-\partial v/\partial x$",
        transform=ax.transAxes, fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.92, pad=2.5),
    )
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)
    return float(omega.min()), float(omega.max()), int(len(levels))


def self_test():
    x = np.linspace(0, 1, 81)
    y = np.linspace(0, 5, 401)
    xx, yy = np.meshgrid(x, y)
    u = xx**2 * yy
    v = xx * yy**2
    exact = xx**2 - yy**2
    error = float(np.max(np.abs(clockwise_vorticity(x, y, u, v) - exact)))
    assert error < 1e-10, error
    print({"manufactured_vorticity_test": "passed", "max_abs_error": error})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtu", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.vtu or not args.output:
            parser.error("vtu and output are required unless --self-test is used")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(dict(zip(("omega_min", "omega_max", "n_levels"), plot(args.vtu, output))))
