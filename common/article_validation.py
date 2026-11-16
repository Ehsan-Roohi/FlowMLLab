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
    from w4utils import GHIA, ghia_valid_mask  # noqa: PLC0415
    sys.path.insert(0, str(root))
    from flowmllab.cavity_visuals import RGB, flow_structure

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
    valid_v = ghia_valid_mask(400, "v")
    v_error_all = float(np.linalg.norm(v_ref_pred - ref["v"]) / np.linalg.norm(ref["v"]))
    v_error = float(np.linalg.norm((v_ref_pred - ref["v"])[valid_v]) / np.linalg.norm(ref["v"][valid_v]))

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
    contour = ax_field.contourf(X, Y, speed, levels=levels, cmap=RGB, extend="max")
    flow_structure(ax_field, x, y, u, v)
    ax_field.set(
        title=r"(a) $Re=400$: speed and streamlines",
        xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal", xlim=(0, 1), ylim=(0, 1),
    )
    cbar = fig.colorbar(contour, ax=ax_field, orientation="horizontal", pad=0.05, shrink=0.88)
    cbar.set_label(r"Speed $|\mathbf{u}|/U_{lid}$", fontsize=13)
    ax_u.plot(u[:, mid_x], y, color="#1261A0", label="FlowMLLab solver")
    ax_u.scatter(ref["u"], ref["y"], s=48, color="#ed2024", edgecolor="white", linewidth=0.6, label="Ghia et al.", zorder=3)
    ax_u.set(title=r"(b) Vertical centerline $u(x/L=0.5,y)$", xlabel=r"$u/U_{lid}$", ylabel=r"$y/L$", ylim=(0, 1.04))
    ax_v.plot(x, v[mid_y, :], color="#1261A0", label="FlowMLLab solver")
    ax_v.scatter(ref["x"][valid_v], ref["v"][valid_v], s=48, color="#ed2024", edgecolor="white", linewidth=0.6, label="Ghia et al. (qualified)", zorder=3)
    ax_v.scatter(ref["x"][~valid_v], ref["v"][~valid_v], s=80, marker="x", color="#9b5500", linewidth=1.8, label="Published anomalous point (not scored)", zorder=4)
    ax_v.set(title=r"(c) Horizontal centerline $v(x,y/L=0.5)$", xlabel=r"$x/L$", ylabel=r"$v/U_{lid}$", xlim=(-0.02, 1.02))
    for axis in (ax_u, ax_v):
        axis.grid(alpha=0.25)
        axis.legend(loc="best", frameon=True)
    paths = _save(fig, output_dir, "fig02_cavity_benchmark")
    metrics = {"Re": 400, "u_centerline_relative_l2": u_error,
               "v_centerline_relative_l2": v_error,
               "v_centerline_relative_l2_including_anomalous_point": v_error_all,
               "excluded_Ghia_v_x_over_L": 0.9063}
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
    sys.path.insert(0, str(root))
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
    from matplotlib.lines import Line2D
    import hashlib
    from PIL import Image as PILImage

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

    # Cyan and yellow anchors avoid the muddy green-to-red blend. This map is
    # local to this figure; earlier course figures keep their rendering rules.
    rainbow = LinearSegmentedColormap.from_list(
        "pressure_rgb_rainbow",
        ["#173ed5", "#00bfff", "#00cf72", "#e6ef00", "#ffad00", "#e51f29"],
        N=256,
    )
    norm = SymLogNorm(linthresh=.01, linscale=1., base=10,
                      vmin=float(pf.min()), vmax=float(pf.max()))
    levels = np.unique(np.r_[norm.inverse(np.linspace(0., 1., 161)), 0.])
    levels[0], levels[-1] = float(pf.min()), float(pf.max())
    style = {
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 14, "axes.labelsize": 16,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.0,
        "xtick.direction": "out", "ytick.direction": "out", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13.6, 7.4))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.35, 1),
                               left=.065, right=.98, bottom=.12, top=.91,
                               wspace=.35, hspace=.40)
        field = fig.add_subplot(grid[:, 0])
        ax_vertical = fig.add_subplot(grid[0, 1])
        ax_horizontal = fig.add_subplot(grid[1, 1])
        im = field.contourf(xf, yf, pf, levels=levels, cmap=rainbow, norm=norm)
        # Matching polygon edges suppress PDF-viewer hairline seams without
        # smoothing, resampling, or rasterizing the native pressure contours.
        im.set_edgecolor("face")
        im.set_linewidth(.5)
        field.streamplot(xf, yf, fine['u'], fine['v'], density=1.45,
                         color="#161b22", linewidth=.60, arrowsize=.72,
                         minlength=.025, zorder=4)
        field.set(xlabel=r"$x/L$", ylabel=r"$y/L$", aspect="equal",
                  xlim=(0, 1), ylim=(0, 1), xticks=np.linspace(0, 1, 6),
                  yticks=np.linspace(0, 1, 6))
        field.grid(False)
        cbar = fig.colorbar(im, ax=field, orientation="horizontal", pad=.10,
                            fraction=.045, shrink=.98, aspect=32)
        cbar.set_ticks([float(pf.min()), -.01, 0., .01, .1, float(pf.max())])
        cbar.set_ticklabels([f"{pf.min():.3f}", "-0.01", "0", "0.01", "0.1", f"{pf.max():.3f}"])
        cbar.set_label(r"$p^*=(p-p_c)/(\rho U_{\mathrm{lid}}^2)$  (symmetric-log)", fontsize=14, labelpad=8)
        cbar.ax.minorticks_off()
        cbar.solids.set_edgecolor("face")
        cbar.solids.set_rasterized(False)
        coarse_style = dict(color="#0878c9", ls="--", lw=1.8)
        fine_style = dict(color="#e4242b", lw=2.25)
        markers = dict(s=33, facecolors="white", edgecolors="#18202a", linewidths=1.1, zorder=5)
        ax_vertical.plot(pc[:, mc], yc, **coarse_style)
        ax_vertical.plot(pf[:, mf], yf, **fine_style)
        ax_vertical.scatter(vertical["p_over_rhoU2"], vertical["coordinate"], **markers)
        ax_vertical.set(xlabel=r"$p^*\quad (x/L=0.5)$", ylabel=r"$y/L$", ylim=(-.02, 1.02))
        # Retain the established reflected abscissa for the published lid convention.
        ax_horizontal.plot(xc, pc[mc, ::-1], **coarse_style)
        ax_horizontal.plot(xf, pf[mf, ::-1], **fine_style)
        ax_horizontal.scatter(horizontal["coordinate"], horizontal["p_over_rhoU2"], **markers)
        ax_horizontal.set(xlabel=r"$x_{\mathrm{ref}}/L\quad (y/L=0.5)$", ylabel=r"$p^*$", xlim=(-.02, 1.02))
        for axis in (ax_vertical, ax_horizontal):
            axis.grid(color="#dce2e8", lw=.65, alpha=.65)
            axis.set_axisbelow(True)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
        handles = [Line2D([], [], **coarse_style, label=r"CFD $65\times65$"),
                   Line2D([], [], **fine_style, label=r"CFD $129\times129$"),
                   Line2D([], [], color="#18202a", marker='o', mfc='white',
                          ls='none', ms=5.5, label="Botella-Peyret")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.779, .99),
                   ncol=3, frameon=False, fontsize=11.2, handlelength=2.0,
                   columnspacing=1.0, handletextpad=.5)
        paths = _save(fig, output_dir, "fig08_pressure_recovery")
    with PILImage.open(paths['png']) as rendered:
        rendered.convert('RGB').save(paths['png'], dpi=(300, 300))
    metrics = {"vertical_relative_l2_n65": evc, "vertical_relative_l2_n129": evf, "horizontal_relative_l2_n65": ehc, "horizontal_relative_l2_n129": ehf, "gradient_mismatch_interior": interior_mismatch, "gradient_mismatch_global": global_mismatch}
    (output_dir / "fig08_pressure_recovery_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    render = {
        "Re": 1000, "field_grid": [129, 129], "gauge": "p(0.5,0.5)=0",
        "pressure_definition": "p*=(p-p_c)/(rho U_lid^2)",
        "color_map": "blue-cyan-green-yellow-orange-red", "png_mode": "RGB",
        "color_normalization": "SymLogNorm, base 10, linthresh 0.01, linscale 1",
        "color_limits": [float(pf.min()), float(pf.max())],
        "clipping": False, "field_smoothing": False, "regridding": False,
        "contours": "161 levels spaced uniformly in color coordinates, plus zero; native-cell level crossings",
        "streamlines": "integrated from retained native u,v; density 1.45",
        "horizontal_profile_mapping": "x_ref/L=1-x/L; unchanged from the retained validation protocol",
        "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (source / 're1000_n65.npz', source / 're1000_n129.npz',
                            source / 'botella_pressure_reference.csv')},
        "outputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
    }
    (output_dir / "fig08_pressure_recovery_render.json").write_text(json.dumps(render, indent=2) + "\n")
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
