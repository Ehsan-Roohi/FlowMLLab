"""Evaluate a retained checkpoint from the original parametric deep-cavity PINN.

The private, author-supplied training source is loaded at run time and is not
copied into this repository.  This script reconstructs the same network and
hard output transform, restores a TensorFlow checkpoint, and writes a regular
grid field for one physical case.  Local streamfunction extrema are reported as
vortex *candidates*; they are not validation against CFD or a paper figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import maximum_filter, minimum_filter


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("chris_deep_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized(value: float, lower: float, upper: float, name: str) -> float:
    if not lower <= value <= upper:
        raise ValueError(f"{name}={value} lies outside [{lower}, {upper}]")
    return (value - lower) / (upper - lower)


def vortex_candidates(psi: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[dict]:
    size = max(5, 2 * (min(psi.shape) // 50) + 1)
    interior = np.ones_like(psi, dtype=bool)
    margin = max(3, size // 2 + 1)
    interior[:margin] = interior[-margin:] = False
    interior[:, :margin] = interior[:, -margin:] = False
    scale = max(float(np.max(np.abs(psi))), np.finfo(float).tiny)
    extrema = ((psi == maximum_filter(psi, size=size)) |
               (psi == minimum_filter(psi, size=size)))
    indices = np.argwhere(extrema & interior & (np.abs(psi) > 1e-8 * scale))
    ordered = sorted(indices, key=lambda ij: abs(psi[tuple(ij)]), reverse=True)
    kept: list[tuple[int, int]] = []
    for iy, ix in ordered:
        if all((iy - jy) ** 2 + (ix - jx) ** 2 >= size**2 for jy, jx in kept):
            kept.append((int(iy), int(ix)))
        if len(kept) == 12:
            break
    return [
        {"x": float(x[ix]), "y": float(y[iy]), "psi": float(psi[iy, ix])}
        for iy, ix in kept
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--re", type=float, default=1000.0)
    parser.add_argument("--depth", type=float, default=2.2)
    parser.add_argument("--tri", type=float, default=0.0)
    parser.add_argument("--nx", type=int, default=301)
    parser.add_argument("--ny", type=int, default=661)
    parser.add_argument("--parameters", type=Path,
                        help="Training case-parameters.json for adapted parameter boxes")
    args = parser.parse_args()

    src = load_source(args.source.resolve())
    if args.parameters:
        parameters = json.loads(args.parameters.read_text())
        for name in ('ReMin', 'ReMax', 'DMin', 'DMax'):
            value = float(parameters[name])
            if not np.isfinite(value):
                raise ValueError('Nonfinite training parameter')
            setattr(src, name, value)
            src.pde.__globals__[name] = value
            src.output_transform_cavity_flow.__globals__[name] = value
    dde = src.dde
    re_n = normalized(args.re, src.ReMin, src.ReMax, "Re")
    depth_n = normalized(args.depth, src.DMin, src.DMax, "depth")
    tri_n = normalized(args.tri, src.triMin, src.triMax, "tri")

    # A minimal PDE dataset builds the original graph without resampling the
    # 100,000 training points.  Network widths and the hard transform are exact.
    geom = dde.geometry.Rectangle([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
    net = dde.maps.FNN([5] + [32] * 6 + [3], "tanh", "Glorot normal")
    net.apply_output_transform(src.output_transform_cavity_flow)
    data = dde.data.PDE(geom, src.pde, [], num_domain=2, num_boundary=0,
                        num_test=2, train_distribution="Hammersley")
    model = dde.Model(data, net)
    model.compile("adam", lr=src.lr, loss=["MSE"] * 2, loss_weights=[1, 1])
    model.restore(str(args.checkpoint.resolve()))

    x = np.linspace(0.0, 1.0, args.nx)
    eta = np.linspace(0.0, 1.0, args.ny)
    xx, ee = np.meshgrid(x, eta)
    points = np.column_stack((xx.ravel(), ee.ravel(),
                              np.full(xx.size, re_n),
                              np.full(xx.size, depth_n),
                              np.full(xx.size, tri_n)))
    prediction = model.predict(points).reshape(args.ny, args.nx, 3)
    u, v, pressure = np.moveaxis(prediction, -1, 0)
    y = args.depth * eta
    psi = cumulative_trapezoid(u, y, axis=0, initial=0.0)
    du_dx = np.gradient(u, x, axis=1, edge_order=2)
    dv_dy = np.gradient(v, y, axis=0, edge_order=2)
    omega = np.gradient(v, x, axis=1, edge_order=2) - np.gradient(
        u, y, axis=0, edge_order=2
    )
    candidates = vortex_candidates(psi, x, y)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "field.npz", x=x, y=y, u=u, v=v,
                        p=pressure, psi=psi, omega=omega)
    report = {
        "source": str(args.source),
        "checkpoint": str(args.checkpoint),
        "parameters": str(args.parameters) if args.parameters else None,
        "case": {"Re": args.re, "depth_over_width": args.depth, "tri": args.tri},
        "grid": {"nx": args.nx, "ny": args.ny},
        "finite": bool(np.isfinite(prediction).all()),
        "max_abs_divergence_fd": float(np.max(np.abs(du_dx + dv_dy))),
        "wall_max_abs_velocity": {
            "bottom": float(np.max(np.hypot(u[0], v[0]))),
            "left": float(np.max(np.hypot(u[:, 0], v[:, 0]))),
            "right": float(np.max(np.hypot(u[:, -1], v[:, -1]))),
        },
        "vortex_candidates": candidates,
        "limitations": [
            "Streamfunction is reconstructed by integrating u from the bottom wall.",
            "Candidate extrema are not CFD or publication validation.",
            "Finite-difference divergence includes differentiation error near walls.",
        ],
    }
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 8), constrained_layout=True)
    levels = np.linspace(float(np.min(psi)), float(np.max(psi)), 41)
    axes[0].contour(x, y, psi, levels=levels, colors="black", linewidths=0.65)
    for candidate in candidates:
        axes[0].plot(candidate["x"], candidate["y"], "o", ms=4, color="#d55e00")
    axes[0].set_title("Streamfunction and candidate centres")
    speed = np.hypot(u, v)
    image = axes[1].pcolormesh(x, y, speed, shading="auto", cmap="viridis")
    axes[1].streamplot(x, y, u, v, color="white", density=1.1,
                       linewidth=0.45, arrowsize=0.65)
    axes[1].set_title("Speed and streamlines")
    fig.colorbar(image, ax=axes[1], shrink=0.72, label="speed / lid speed")
    vort_lim = float(np.quantile(np.abs(omega), 0.995))
    image = axes[2].pcolormesh(x, y, omega, shading="auto", cmap="RdBu_r",
                               vmin=-vort_lim, vmax=vort_lim)
    axes[2].set_title("Vorticity")
    fig.colorbar(image, ax=axes[2], shrink=0.72, label="vorticity")
    for axis in axes:
        axis.set(xlabel="x/W", ylabel="y/W", aspect="equal")
        axis.tick_params(labelsize=10)
    fig.suptitle(f"Original deep-cavity PINN checkpoint: Re={args.re:g}, D/W={args.depth:g}")
    fig.savefig(args.output / "checkpoint_diagnostics.png", dpi=220)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
