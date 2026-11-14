"""Render the three README examples from retained course data and notebook output."""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/assets/showcase"


def cavity():
    with np.load(ROOT / "data/cavity_data.npz", allow_pickle=False) as field:
        index, = np.flatnonzero(field["Re"] == 400)
        x, y = field["x"], field["y"]
        psi = field["psi"][index]

    fig, ax = plt.subplots(figsize=(5.1, 5.1))
    interior_levels = -np.r_[0.1065, 0.104, 0.100, np.linspace(0.095, 0.005, 23),
                             0.003, 0.0015]
    corner_levels = np.r_[1.3e-6, 3.0e-6, 5.4e-6, 7.5e-6,
                          np.linspace(5e-5, 5.5e-4, 8)]
    ax.contour(x, y, psi, levels=np.r_[interior_levels, corner_levels], colors="black",
               linewidths=0.52, linestyles="solid")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$x/L$", ylabel="$y/L$")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    fig.subplots_adjust(left=0.125, right=0.965, bottom=0.12, top=0.965)
    fig.savefig(OUTPUT / "week01-cavity-streamlines.png", dpi=200)
    plt.close(fig)


def cylinder():
    with np.load(ROOT / "results/cylinder_d40/re100_D040.npz", allow_pickle=False) as field:
        setup = json.loads(str(field["metadata"]))["config"]
        diameter, inflow = setup["diameter"], setup["inflow_velocity"]
        x = (field["x"] - setup["center_x"]) / diameter
        y = (field["y"] - setup["center_y"]) / diameter
        values = np.ma.masked_array(field["vorticity"] * diameter / inflow,
                                    mask=field["solid"].astype(bool))

    norm = SymLogNorm(linthresh=0.5, vmin=-18, vmax=18)
    assert np.max(np.abs(values)) < norm.vmax
    levels = norm.inverse(np.linspace(0, 1, 129))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    contour = ax.contourf(x, y, values, levels=levels, norm=norm, cmap="RdBu_r")
    ax.add_patch(Circle((0, 0), radius=0.5, facecolor="#25313d", edgecolor="none", zorder=3))
    ax.set(xlim=(-1.2, 12), ylim=(-3.5, 3.5), aspect="equal",
           xlabel=r"$(x-x_c)/D$", ylabel=r"$(y-y_c)/D$")
    ax.set_xticks(range(0, 13, 2))
    ax.set_yticks([-3, 0, 3])
    fig.subplots_adjust(left=0.09, right=0.875, bottom=0.13, top=0.97)
    colorbar = fig.colorbar(contour, ax=ax, pad=0.025, fraction=0.030, aspect=27,
                            format="%g",
                            ticks=[-18, -3, -0.5, 0, 0.5, 3, 18])
    colorbar.set_label(r"$\omega_z D/U_\infty$", labelpad=6)
    fig.savefig(OUTPUT / "week07-cylinder-vorticity.png", dpi=190)
    plt.close(fig)


def optimizers():
    notebook = ROOT / "notebooks/week01_3/W1_3_Keras_Optimizers.ipynb"
    doc = json.loads(notebook.read_text(encoding="utf-8"))
    comparison = next(cell for cell in doc["cells"] if cell.get("id") == "week01-3-opt-08")
    outputs = comparison["outputs"]
    table = next(i for i, output in enumerate(outputs)
                 if "".join(output.get("data", {}).get("text/markdown", [])).startswith("**Localized**"))
    encoded = "".join(outputs[table + 1]["data"]["image/png"])
    (OUTPUT / "week01-3-optimizer-comparison.png").write_bytes(base64.b64decode(encoded, validate=True))


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8, "savefig.facecolor": "white"})
    cavity()
    cylinder()
    optimizers()
