#!/usr/bin/env python3
"""Reproduce the manuscript's solver-validation figures from retained evidence.

This module is the single source of truth shared by the Week-1 cavity notebook,
the Week-3 DSMC notebook, and the manuscript.  It never fabricates a comparison
curve: every line comes from a retained solver output and every reference marker
comes from a cited tabulation or a vector extraction of the published markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_course_root(start: str | Path | None = None) -> Path:
    """Locate the FlowMLLab root from a notebook, script, or Colab directory."""
    starts = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve()]
    for origin in starts:
        for candidate in (origin, *origin.parents):
            if (candidate / "common" / "w4utils.py").exists() and (candidate / "data").exists():
                return candidate
    raise FileNotFoundError("Could not locate the FlowMLLab repository root.")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.15,
            "lines.linewidth": 2.5,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[suffix] = path
    plt.close(fig)
    return paths


def build_ghia_velocity_validation(
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Build manuscript Figure 2 and report the two Ghia centerline errors."""
    root = find_course_root(root)
    output_dir = Path(output_dir or root / "results" / "article_figures")
    sys.path.insert(0, str(root / "common"))
    from w4utils import GHIA  # noqa: PLC0415

    data = _load_npz(root / "data" / "cavity_data.npz")
    index = int(np.where(np.isclose(data["Re"], 400.0))[0][0])
    x, y = data["x"], data["y"]
    u, v = data["u"][index], data["v"][index]
    speed = np.hypot(u, v)
    X, Y = np.meshgrid(x, y)
    mid_x = int(np.argmin(np.abs(x - 0.5)))
    mid_y = int(np.argmin(np.abs(y - 0.5)))
    ref = GHIA[400]
    u_ref_pred = np.interp(ref["y"], y, u[:, mid_x])
    v_ref_pred = np.interp(ref["x"], x, v[mid_y, :])
    u_error = float(np.linalg.norm(u_ref_pred - ref["u"]) / np.linalg.norm(ref["u"]))
    v_error = float(np.linalg.norm(v_ref_pred - ref["v"]) / np.linalg.norm(ref["v"]))

    _style()
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.1,
        }
    )
    fig = plt.figure(figsize=(13.6, 8.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0), height_ratios=(1, 1))
    ax_field = fig.add_subplot(grid[:, 0])
    ax_u = fig.add_subplot(grid[0, 1])
    ax_v = fig.add_subplot(grid[1, 1])
    levels = np.linspace(0.0, 1.0, 25)
    contour = ax_field.contourf(X, Y, speed, levels=levels, cmap="viridis", extend="max")
    ax_field.streamplot(x, y, u, v, color="white", density=1.25, linewidth=0.8, arrowsize=0.9)
    ax_field.set(
        title=r"(a) $Re=400$: speed and streamlines",
        xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal", xlim=(0, 1), ylim=(0, 1),
    )
    cbar = fig.colorbar(contour, ax=ax_field, orientation="horizontal", pad=0.05, shrink=0.88)
    cbar.set_label(r"Speed $|\mathbf{u}|/U_{lid}$", fontsize=13)
    ax_u.plot(u[:, mid_x], y, color="#1261A0", label="FlowMLLab solver")
    ax_u.scatter(ref["u"], ref["y"], s=48, color="#E87500", edgecolor="white", linewidth=0.6, label="Ghia et al.", zorder=3)
    ax_u.set(title=r"(b) Vertical centerline $u(x/L=0.5,y)$", xlabel=r"$u/U_{lid}$", ylabel=r"$y/L$", ylim=(0, 1.04))
    ax_v.plot(x, v[mid_y, :], color="#1261A0", label="FlowMLLab solver")
    ax_v.scatter(ref["x"], ref["v"], s=48, color="#E87500", edgecolor="white", linewidth=0.6, label="Ghia et al.", zorder=3)
    ax_v.set(title=r"(c) Horizontal centerline $v(x,y/L=0.5)$", xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$", xlim=(-0.02, 1.02))
    for axis in (ax_u, ax_v):
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=True)
    paths = _save(fig, output_dir, "fig02_cavity_benchmark")
    metrics = {"Re": 400, "u_centerline_relative_l2": u_error, "v_centerline_relative_l2": v_error}
    (output_dir / "fig02_cavity_benchmark_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"figure": paths, "metrics": metrics}


def _center_gauge(case: dict[str, np.ndarray]) -> np.ndarray:
    p = np.asarray(case["p"], dtype=float).copy()
    mid = p.shape[0] // 2
    return p - p[mid, mid]


def _gradient_mismatch(case: dict[str, np.ndarray], crop: int, w4utils) -> float:
    x, y = np.asarray(case["x"]), np.asarray(case["y"])
    u, v, p = np.asarray(case["u"]), np.asarray(case["v"]), np.asarray(case["p"])
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    gx, gy = w4utils.momentum_pressure_gradient(u, v, float(case["Re"]), dx, dy)
    pgx = np.gradient(p, dx, axis=1, edge_order=2)
    pgy = np.gradient(p, dy, axis=0, edge_order=2)
    sl = (slice(crop, -crop), slice(crop, -crop)) if crop else (...,)
    numerator = np.sqrt(np.mean((pgx[sl] - gx[sl]) ** 2 + (pgy[sl] - gy[sl]) ** 2))
    denominator = np.sqrt(np.mean(gx[sl] ** 2 + gy[sl] ** 2))
    return float(numerator / (denominator + 1.0e-30))


def build_pressure_validation(
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Build the Re=1000 pressure field and Botella--Peyret comparisons."""
    root = find_course_root(root)
    output_dir = Path(output_dir or root / "results" / "article_figures")
    source = root / "results" / "article_validation"
    sys.path.insert(0, str(root / "common"))
    import w4utils  # noqa: PLC0415

    coarse = _load_npz(source / "re1000_n65.npz")
    fine = _load_npz(source / "re1000_n129.npz")
    reference = pd.read_csv(source / "botella_pressure_reference.csv")
    vertical = reference[reference["orientation"] == "vertical"]
    horizontal = reference[reference["orientation"] == "horizontal"]
    pc, pf = _center_gauge(coarse), _center_gauge(fine)
    xc, yc, xf, yf = coarse["x"], coarse["y"], fine["x"], fine["y"]
    mc, mf = len(xc) // 2, len(xf) // 2
    vc = np.interp(vertical["coordinate"], yc, pc[:, mc])
    vf = np.interp(vertical["coordinate"], yf, pf[:, mf])
    hc = np.interp(horizontal["coordinate"], xc, pc[mc, ::-1])
    hf = np.interp(horizontal["coordinate"], xf, pf[mf, ::-1])
    evc = float(np.linalg.norm(vc - vertical["p_over_rhoU2"]) / np.linalg.norm(vertical["p_over_rhoU2"]))
    evf = float(np.linalg.norm(vf - vertical["p_over_rhoU2"]) / np.linalg.norm(vertical["p_over_rhoU2"]))
    ehc = float(np.linalg.norm(hc - horizontal["p_over_rhoU2"]) / np.linalg.norm(horizontal["p_over_rhoU2"]))
    ehf = float(np.linalg.norm(hf - horizontal["p_over_rhoU2"]) / np.linalg.norm(horizontal["p_over_rhoU2"]))
    global_mismatch = _gradient_mismatch(fine, 0, w4utils)
    interior_mismatch = _gradient_mismatch(fine, 4, w4utils)

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.1), constrained_layout=True)
    levels = np.linspace(np.min(pf), np.max(pf), 33)
    im = axes[0, 0].contourf(*np.meshgrid(xf, yf), pf, levels=levels, cmap="coolwarm", extend="both")
    axes[0, 0].set(title=r"(a) Recovered pressure, $Re=1000$, $129^2$ grid", xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.9, label=r"$(p-p_c)/(\rho U_{lid}^2)$")
    axes[0, 1].plot(pc[:, mc], yc, "--", color="#94A3B8", label=rf"FlowMLLab $65^2$ ($E={100*evc:.1f}\%$)")
    axes[0, 1].plot(pf[:, mf], yf, color="#1D4ED8", label=rf"FlowMLLab $129^2$ ($E={100*evf:.2f}\%$)")
    axes[0, 1].scatter(vertical["p_over_rhoU2"], vertical["coordinate"], s=42, facecolors="white", edgecolors="black", linewidths=1.2, zorder=3, label="Botella--Peyret")
    axes[0, 1].set(title="(b) Vertical centerline", xlabel=r"$(p-p_c)/(\rho U_{lid}^2)$", ylabel=r"$y/L$")
    axes[1, 0].plot(xc, pc[mc, ::-1], "--", color="#94A3B8", label=rf"FlowMLLab $65^2$ ($E={100*ehc:.1f}\%$)")
    axes[1, 0].plot(xf, pf[mf, ::-1], color="#B45309", label=rf"FlowMLLab $129^2$ ($E={100*ehf:.2f}\%$)")
    axes[1, 0].scatter(horizontal["coordinate"], horizontal["p_over_rhoU2"], s=42, facecolors="white", edgecolors="black", linewidths=1.2, zorder=3, label="Botella--Peyret")
    axes[1, 0].set(title="(c) Horizontal centerline (lid-direction mapped)", xlabel=r"$x/L$", ylabel=r"$(p-p_c)/(\rho U_{lid}^2)$")
    positions, width = np.arange(2), 0.34
    axes[1, 1].bar(positions - width / 2, 100 * np.array([evc, ehc]), width, color="#94A3B8", label=r"$65^2$")
    axes[1, 1].bar(positions + width / 2, 100 * np.array([evf, ehf]), width, color=["#1D4ED8", "#B45309"], label=r"$129^2$")
    axes[1, 1].set(title="(d) Literature error decreases under refinement", ylabel=r"centerline relative $L_2$ error (\%)", xticks=positions, xticklabels=["vertical", "horizontal"], ylim=(0, 24))
    axes[1, 1].text(0.98, 0.95, f"129² grid\nsteady residual: {float(fine['final_residual']):.2e}\ngradient mismatch: {100*interior_mismatch:.2f}% interior\n({100*global_mismatch:.1f}% including corners)", transform=axes[1, 1].transAxes, ha="right", va="top", fontsize=12, bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.65"})
    for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.grid(alpha=0.25)
        axis.legend(loc="best", framealpha=0.96)
    fig.suptitle("Pressure recovery validated against the Re=1000 benchmark", fontsize=19, fontweight="bold")
    paths = _save(fig, output_dir, "fig08_pressure_recovery")
    metrics = {"vertical_relative_l2_n65": evc, "vertical_relative_l2_n129": evf, "horizontal_relative_l2_n65": ehc, "horizontal_relative_l2_n129": ehf, "gradient_mismatch_interior": interior_mismatch, "gradient_mismatch_global": global_mismatch}
    (output_dir / "fig08_pressure_recovery_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"figure": paths, "metrics": metrics}


def build_dsmc_wall_pressure_validation(
    root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """Validate the HS--NTC solver against Mohammadzadeh Fig. 3."""
    root = find_course_root(root)
    output_dir = Path(output_dir or root / "results" / "article_figures")
    validation = root / "results" / "dsmc_validation"
    reference = pd.read_csv(validation / "mohammadzadeh_fig3_dsmc_points.csv")
    coarse_names = ("ref_n40_seed07", "ref_n40_seed19", "ref_n40_seed31")
    coarse_cases = [pd.read_csv(validation / "runs" / name / "wall_pressure.csv") for name in coarse_names]
    fine = pd.read_csv(validation / "runs" / "ref_n60_seed07" / "wall_pressure.csv")
    metadata = json.loads((validation / "runs" / "ref_n60_seed07" / "metadata.json").read_text())
    expected = {"Kn": 0.1, "reference_reported_Re": 1.5, "reference_Ma": 0.09, "nx": 60, "ny": 60, "ppc": 50}
    for key, value in expected.items():
        if not np.isclose(float(metadata[key]), float(value)):
            raise ValueError(f"Retained DSMC run has {key}={metadata[key]!r}; expected {value!r}.")
    if metadata.get("backend_used") != "numpy-cpu-vectorized":
        raise ValueError("The retained paper curve is not tagged as the validated vectorized CPU configuration.")
    xc = coarse_cases[0]["s_over_L"].to_numpy()
    yc = np.stack([case["p_five_point_filtered_over_p0"].to_numpy() for case in coarse_cases])
    coarse_mean, coarse_std = yc.mean(axis=0), yc.std(axis=0, ddof=1)
    xf, yf = fine["s_over_L"].to_numpy(), fine["p_five_point_filtered_over_p0"].to_numpy()
    xr, yr = reference["s_over_L"].to_numpy(), reference["p_over_p0"].to_numpy()
    fine_error = np.interp(xr, xf, yf) - yr
    coarse_error = np.interp(xr, xc, coarse_mean) - yr
    relative_l2 = float(np.linalg.norm(fine_error) / np.linalg.norm(yr))
    rmse = float(np.sqrt(np.mean(fine_error**2)))
    maximum = float(np.max(np.abs(fine_error)))
    grid_change = float(np.linalg.norm(np.interp(xc, xf, yf) - coarse_mean) / np.linalg.norm(np.interp(xc, xf, yf)))

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.5), gridspec_kw={"width_ratios": (1.22, 0.78)})
    axes[0].fill_between(xc, coarse_mean - coarse_std, coarse_mean + coarse_std, color="#F59E0B", alpha=0.20, label=r"$40^2$: run-to-run $\pm1\sigma$")
    axes[0].plot(xc, coarse_mean, color="#B45309", ls="--", lw=2.3, label=r"our DSMC, $40^2$ (3-seed mean)")
    axes[0].plot(xf, yf, color="#1D4ED8", lw=2.8, label=r"our DSMC, $60^2$ (reported)")
    axes[0].scatter(xr, yr, s=47, color="black", edgecolor="white", linewidth=0.65, zorder=5, label="Mohammadzadeh et al. DSMC")
    axes[0].set(title=r"(a) Direct solver validation: $Re=1.5$, $Kn=0.1$, $Ma=0.09$", xlabel=r"wall coordinate $s/L$", ylabel=r"$p/p_0$", xlim=(0, 4), ylim=(0.895, 1.125), xticks=[0, 1, 2, 3, 4], xticklabels=["A", "B", "C", "D", "A"])
    axes[0].legend(loc="best", framealpha=0.97, fontsize=11.2)
    axes[1].axhline(0.0, color="#334155", lw=1.2)
    axes[1].plot(xr, coarse_error, "^--", color="#B45309", ms=6.0, lw=1.8, label=r"$40^2$ mean")
    axes[1].plot(xr, fine_error, "s-", color="#1D4ED8", ms=5.4, lw=2.0, label=r"$60^2$")
    axes[1].set(title="(b) Pointwise difference from published DSMC markers", xlabel=r"wall coordinate $s/L$", ylabel=r"$(p_{ours}-p_{published})/p_0$", xlim=(0, 4), ylim=(-0.021, 0.021), xticks=[0, 1, 2, 3, 4], xticklabels=["A", "B", "C", "D", "A"])
    axes[1].legend(loc="lower right", framealpha=0.97)
    axes[1].text(0.04, 0.96, rf"$60^2$ relative $L_2$: {100*relative_l2:.3f}\%" "\n" rf"RMSE: {rmse:.4f} $p_0$" "\n" rf"max $|\Delta|$: {maximum:.4f} $p_0$" "\n" rf"$40^2\rightarrow60^2$ change: {100*grid_change:.3f}\%" "\n" r"50 particles/cell; 6000 field samples", transform=axes[1].transAxes, ha="left", va="top", fontsize=12.2, bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#64748B", "alpha": 0.97})
    for axis in axes:
        axis.grid(alpha=0.24)
        for boundary in (1, 2, 3):
            axis.axvline(boundary, color="#94A3B8", ls=":", lw=1.0, zorder=0)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.13, top=0.84, wspace=0.22)
    fig.suptitle("HS--NTC solver validation against Mohammadzadeh DSMC wall pressure", y=0.98, fontsize=18, fontweight="bold")
    paths = _save(fig, output_dir, "fig10a_mohammadzadeh_validation")
    metrics = {"relative_l2": relative_l2, "rmse_over_p0": rmse, "max_abs_over_p0": maximum, "grid_change_40_to_60": grid_change, "fine_run": "ref_n60_seed07", "coarse_runs": list(coarse_names)}
    (output_dir / "fig10a_mohammadzadeh_validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"figure": paths, "metrics": metrics}


def build_all(root: str | Path | None = None) -> dict:
    root = find_course_root(root)
    return {
        "ghia_velocity": build_ghia_velocity_validation(root),
        "cavity_pressure": build_pressure_validation(root),
        "dsmc_wall_pressure": build_dsmc_wall_pressure_validation(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("ghia", "pressure", "dsmc", "all", "check"), nargs="?", default="all")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    root = find_course_root(args.root)
    if args.target == "ghia":
        report = build_ghia_velocity_validation(root)
    elif args.target == "pressure":
        report = build_pressure_validation(root)
    elif args.target == "dsmc":
        report = build_dsmc_wall_pressure_validation(root)
    else:
        report = build_all(root)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
