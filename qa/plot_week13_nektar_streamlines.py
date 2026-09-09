"""Plot streamfunction contours from a validated Nektar++ ASCII VTU export."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from audit_week13_nektar import integrate_paths, structured
from check_week13_nektar_vtu import read_vtu


def plot(vtu, output):
    x, y, fields, _ = structured(read_vtu(vtu))
    psi_h, psi_v = integrate_paths(x, y, fields["u"], fields["v"])
    psi = 0.5 * (psi_h + psi_v)

    negative = -np.geomspace(1e-8, 1e-1, 22)[::-1]
    positive = np.geomspace(1e-8, 1e-3, 16)
    levels = np.unique(np.r_[negative, 0.0, positive])
    levels = levels[(levels >= psi.min()) & (levels <= psi.max())]

    fig, ax = plt.subplots(figsize=(3.45, 10.2), constrained_layout=True)
    ax.contour(x, y, psi, levels=levels, colors="#17202a", linewidths=0.62)
    ax.contour(x, y, psi, levels=[0.0], colors="#c44536", linewidths=1.05)
    ax.set(xlim=(0, 1), ylim=(0, 5), xlabel=r"$x/W$", ylabel=r"$y/W$")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(r"Nektar++: $Re=100$, $D/W=5$, $tU/W=80$", fontsize=10)
    ax.text(0.03, 0.02, r"Contours of $\psi/(UW)$; red: $\psi=0$",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.9, pad=2.5))
    fig.savefig(output, dpi=400, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vtu")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot(args.vtu, output)
