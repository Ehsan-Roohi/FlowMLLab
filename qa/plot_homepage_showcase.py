"""Render README figures from retained course data and notebook output."""

import argparse
import base64
import json
import sys
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
    sys.path.insert(0, str(ROOT / "common"))
    from w4utils import GHIA, ghia_valid_mask

    with np.load(ROOT / "data/cavity_data.npz", allow_pickle=False) as field:
        index, = np.flatnonzero(field["Re"] == 400)
        x, y = field["x"], field["y"]
        psi, u, v = (field[key][index] for key in ("psi", "u", "v"))
    reference = GHIA[400]
    valid_v = ghia_valid_mask(400, "v")
    mid_x = int(np.argmin(np.abs(x - 0.5)))
    mid_y = int(np.argmin(np.abs(y - 0.5)))

    fig = plt.figure(figsize=(11.3, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1), left=0.075, right=0.975,
                           bottom=0.105, top=0.86, wspace=0.30, hspace=0.51)
    field_ax = fig.add_subplot(grid[:, 0])
    u_ax = fig.add_subplot(grid[0, 1])
    v_ax = fig.add_subplot(grid[1, 1])
    interior_levels = -np.r_[0.1065, 0.104, 0.100, np.linspace(0.095, 0.005, 23),
                             0.003, 0.0015]
    corner_levels = np.r_[1.3e-6, 3.0e-6, 5.4e-6, 7.5e-6,
                          np.linspace(5e-5, 5.5e-4, 8)]
    field_ax.contour(x, y, psi, levels=np.r_[interior_levels, corner_levels],
                     colors="black", linewidths=0.54, linestyles="solid")
    field_ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="$x/L$", ylabel="$y/L$",
                 title="(a) Streamlines, Re = 400")
    field_ax.set_xticks(np.linspace(0, 1, 6))
    field_ax.set_yticks(np.linspace(0, 1, 6))

    cfd, = u_ax.plot(u[:, mid_x], y, color="#1261a0", lw=1.8, label="FlowMLLab CFD")
    ghia = u_ax.scatter(reference["u"], reference["y"], s=24, color="#c3282d",
                        edgecolor="white", linewidth=0.35, label="Ghia et al.", zorder=3)
    u_ax.set(title="(b) Vertical centerline", xlabel=r"$u/U_{lid}$ at $x/L=0.5$",
             ylabel="$y/L$", ylim=(-0.015, 1.025))

    v_ax.plot(x, v[mid_y, :], color="#1261a0", lw=1.8)
    v_ax.scatter(reference["x"][valid_v], reference["v"][valid_v], s=24, color="#c3282d",
                 edgecolor="white", linewidth=0.35, zorder=3)
    anomalous = v_ax.scatter(reference["x"][~valid_v], reference["v"][~valid_v],
                             s=45, color="#9b5500", marker="x", linewidth=1.3,
                             label="Tabulated point excluded from error norm", zorder=4)
    v_ax.set(title="(c) Horizontal centerline", xlabel=r"$x/L$",
             ylabel=r"$v/U_{lid}$ at $y/L=0.5$", xlim=(-0.02, 1.02))
    for ax in (u_ax, v_ax):
        ax.grid(alpha=0.23)
    fig.legend(handles=[cfd, ghia, anomalous], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, 0.977), fontsize=11)
    fig.savefig(OUTPUT / "week01-cavity-streamlines.png", dpi=180)
    plt.close(fig)


def coupling():
    methods = ("streamfunction", "simple", "piso", "pimple")
    labels = ("Streamfunction–vorticity", "SIMPLE", "PISO", "PIMPLE")
    levels = np.r_[-np.r_[0.101, 0.097, np.linspace(0.09, 0.004, 17), 0.0015],
                   0.7e-6, 1.5e-6, 2.2e-6, 4e-6, 7e-6, 11e-6, 14e-6]
    fig, axes = plt.subplots(1, 4, figsize=(11.3, 3.25), sharex=True, sharey=True,
                             layout="constrained")
    for ax, method, label in zip(axes, methods, labels):
        path = ROOT / f"results/week01_2_pressure_velocity/{method}-re100-n64.npz"
        with np.load(path, allow_pickle=False) as field:
            psi = field["psi"]
            # FV streamfunctions are on cell corners; the FD reference uses boundary-inclusive nodes.
            y_nodes = np.linspace(0, 1, psi.shape[0])
            x_nodes = np.linspace(0, 1, psi.shape[1])
            ax.contour(x_nodes, y_nodes, psi, levels=levels,
                       colors="black", linewidths=0.62, linestyles="solid")
        ax.set(aspect="equal", xlim=(0, 1), ylim=(0, 1), xlabel="$x/L$", title=label,
               xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    axes[0].set_ylabel("$y/L$")
    fig.savefig(OUTPUT / "week01-2-coupling-streamlines.png", dpi=185)
    plt.close(fig)


def cylinder(case_path):
    import hashlib
    from matplotlib.colors import Normalize
    sys.path.insert(0, str(ROOT))
    from flowmllab import cylinder_cnn
    from run_cylinder_multiscale_cnn import _full_inputs, GRID, HISTORY

    published_hash = "fd6bed163b1fcb94d1157b9046db60a214841ad86d91346627fe7a5106b647ee"
    with case_path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != published_hash:
            raise ValueError("Expected the published cylinder-cfd-v1 Re105 NPZ")
    with np.load(case_path, allow_pickle=False) as data:
        case = {key: data[key] for key in ("x", "y", "solid", "u", "v", "p", "snapshot_time")}
    case["solid"] = case["solid"].astype(bool)
    target = 142
    windows = cylinder_cnn.temporal_windows(len(case["u"]), history=HISTORY)
    window = next(entry for entry in windows if entry.target == target)
    features, truth, _ = _full_inputs(case, [window], 105)
    weights = ROOT / "results/cylinder_cnn/multiscale_cnn.weights.h5"
    model = cylinder_cnn.build_multiscale_predictor(filters=24)
    model.load_weights(weights)
    prediction = np.asarray(model(features, training=False))
    diameter = float(GRID["diameter"])
    center_x, center_y = GRID["center"]
    x = (case["x"] - center_x) / diameter
    y = (case["y"] - center_y) / diameter
    exact = cylinder_cnn.vorticity(truth[0, ..., 0], truth[0, ..., 1], diameter=diameter)
    fitted = cylinder_cnn.vorticity(prediction[0, ..., 0], prediction[0, ..., 1], diameter=diameter)
    fluid = ~case["solid"]
    shown = (x >= -1.2) & (x <= 12)
    shown_y = (y >= -3.5) & (y <= 3.5)
    region = fluid & shown[None, :] & shown_y[:, None]
    scale = np.sqrt(np.mean(exact[region]**2))
    error = 100 * (fitted - exact) / scale
    relative = np.linalg.norm((fitted - exact)[region]) / np.linalg.norm(exact[region])
    assert np.isfinite(error[region]).all() and 0 < relative < 0.05
    print(f"Re = 105, frame {target}, displayed-window vorticity relative L2: {100*relative:.3f}%")

    field_limit = np.ceil(max(np.max(np.abs(exact[region])), np.max(np.abs(fitted[region]))))
    error_limit = np.ceil(np.max(np.abs(error[region])))
    field_norm = SymLogNorm(linthresh=0.5, vmin=-field_limit, vmax=field_limit)
    error_norm = Normalize(vmin=-error_limit, vmax=error_limit)
    fig = plt.figure(figsize=(11.4, 3.65), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 0.07))
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    color_axes = (fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2]))
    panels = (
        (exact, field_norm, "RdBu_r", "(a) CFD reference (LBM)"),
        (fitted, field_norm, "RdBu_r", "(b) CNN prediction"),
        (error, error_norm, "PuOr_r", "(c) Signed relative error"),
    )
    mapped = []
    for ax, (values, norm, cmap, title) in zip(axes, panels):
        levels = norm.inverse(np.linspace(0, 1, 129))
        plotted = np.ma.masked_where(~fluid, values)
        mapped.append(ax.contourf(x, y, plotted, levels=levels, norm=norm, cmap=cmap))
        ax.add_patch(Circle((0, 0), radius=0.5, color="#25313d", linewidth=0))
        ax.set(xlim=(-1.2, 12), ylim=(-3.5, 3.5), aspect="equal", title=title,
               xlabel=r"$(x-x_c)/D$", xticks=[0, 4, 8, 12], yticks=[-3, 0, 3])
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel(r"$(y-y_c)/D$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.colorbar(mapped[0], cax=color_axes[0], orientation="horizontal",
                 ticks=[-field_limit, -2, 0, 2, field_limit], format="%g",
                 label=r"Vorticity $\omega_z D/U_\infty$")
    fig.colorbar(mapped[2], cax=color_axes[1], orientation="horizontal",
                 ticks=np.linspace(-error_limit, error_limit, 5), format="%g",
                 label="Signed error (%)")
    fig.savefig(OUTPUT / "week07-cylinder-vorticity.png", dpi=170)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cylinder-case", type=Path, help="Published cylinder_cfd_re105.npz from cylinder-cfd-v1")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.linewidth": 0.8, "savefig.facecolor": "white"})
    cavity()
    coupling()
    optimizers()
    if args.cylinder_case is not None:
        cylinder(args.cylinder_case)
