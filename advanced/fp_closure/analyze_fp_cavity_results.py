#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-process exact-FP and ML-FP held-out cavity fields for Track 6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

K_B = 1.380e-23
MASS_AR = 66.3e-27


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1.0e-300))


def field_metrics(name, a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return {
        "quantity": name,
        "relative_L2": rel_l2(a, b),
        "RMSE": float(np.sqrt(np.mean((a - b) ** 2))),
        "MAE": float(np.mean(np.abs(a - b))),
        "max_abs_error": float(np.max(np.abs(a - b))),
    }


def reshape(field, nx, ny):
    arr = np.asarray(field)
    if arr.shape == (ny, nx):
        return arr
    if arr.shape == (nx, ny):
        return arr.T
    if arr.ndim == 1 and arr.size == nx * ny:
        return arr.reshape(ny, nx)
    raise ValueError(f"Cannot reshape {arr.shape} into ({ny},{nx})")


def save_both(fig, base):
    fig.savefig(str(base) + ".png", dpi=250, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", bbox_inches="tight")


def make_triptych(x, y, phys, ml, title, outbase):
    err = np.abs(ml - phys)
    vmin = min(float(np.min(phys)), float(np.min(ml)))
    vmax = max(float(np.max(phys)), float(np.max(ml)))
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    im0 = axes[0].contourf(x, y, phys, levels=40, vmin=vmin, vmax=vmax)
    axes[0].set_title("Exact cubic-FP")
    im1 = axes[1].contourf(x, y, ml, levels=40, vmin=vmin, vmax=vmax)
    axes[1].set_title("ML-FP")
    im2 = axes[2].contourf(x, y, err, levels=40)
    axes[2].set_title("Absolute error")
    fig.colorbar(im1, ax=axes[:2], shrink=0.88)
    fig.colorbar(im2, ax=axes[2], shrink=0.88)
    for ax in axes:
        ax.set_xlabel("x/L")
        ax.set_ylabel("y/L")
        ax.set_aspect("equal")
    fig.suptitle(title)
    save_both(fig, outbase)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--physics", required=True)
    ap.add_argument("--ml", required=True)
    ap.add_argument("--outdir", default="fp_cavity_analysis")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    P = load_npz(args.physics)
    M = load_npz(args.ml)

    nx = int(np.asarray(P["nx"]))
    ny = int(np.asarray(P["ny"]))
    x = np.asarray(P["x_coords"], dtype=np.float64)
    y = np.asarray(P["y_coords"], dtype=np.float64)
    x_nd = (x - x.min()) / max(x.max() - x.min(), 1.0e-300)
    y_nd = (y - y.min()) / max(y.max() - y.min(), 1.0e-300)

    U_p = np.asarray(P["U"], dtype=np.float64)
    U_m = np.asarray(M["U"], dtype=np.float64)
    speed_p = np.linalg.norm(U_p[:, :2], axis=1)
    speed_m = np.linalg.norm(U_m[:, :2], axis=1)
    T_p = np.asarray(P["T"], dtype=np.float64)
    T_m = np.asarray(M["T"], dtype=np.float64)
    rho_p_raw = np.asarray(P["rho"], dtype=np.float64)
    rho_m_raw = np.asarray(M["rho"], dtype=np.float64)
    rho0 = max(float(np.mean(rho_p_raw)), 1.0e-300)
    rho_p = rho_p_raw / rho0
    rho_m = rho_m_raw / rho0
    p_p_raw = rho_p_raw / MASS_AR * K_B * T_p
    p_m_raw = rho_m_raw / MASS_AR * K_B * T_m
    p0 = max(float(np.mean(p_p_raw)), 1.0e-300)
    p_p = p_p_raw / p0
    p_m = p_m_raw / p0

    rows = [
        field_metrics("speed", speed_p, speed_m),
        field_metrics("temperature", T_p, T_m),
        field_metrics("density_normalized", rho_p, rho_m),
        field_metrics("pressure_normalized", p_p, p_m),
        field_metrics("velocity_vector", U_p[:, :2], U_m[:, :2]),
        field_metrics("C_coefficients_closed_loop", P["A"], M["A"]),
        field_metrics("Gamma_coefficients_closed_loop", P["B"], M["B"]),
    ]

    high_keys = ["sigma_norm", "q_norm", "m3_norm", "Rij_norm", "Delta4_norm", "DM6_norm"]
    for key in high_keys:
        if key in P and key in M:
            rows.append(field_metrics(key, P[key], M[key]))

    with (outdir / "fp_cavity_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    speed_p2 = reshape(speed_p, nx, ny)
    speed_m2 = reshape(speed_m, nx, ny)
    T_p2 = reshape(T_p / float(np.asarray(P["T_wall"])), nx, ny)
    T_m2 = reshape(T_m / float(np.asarray(M["T_wall"])), nx, ny)
    rho_p2 = reshape(rho_p, nx, ny)
    rho_m2 = reshape(rho_m, nx, ny)

    make_triptych(x_nd, y_nd, speed_p2, speed_m2, "Speed", outdir / "fp_speed_comparison")
    make_triptych(x_nd, y_nd, T_p2, T_m2, "Temperature T/Twall", outdir / "fp_temperature_comparison")
    make_triptych(x_nd, y_nd, rho_p2, rho_m2, "Normalized density", outdir / "fp_density_comparison")

    ix = int(np.argmin(np.abs(x_nd - 0.5)))
    U_lid = float(np.asarray(P["U_lid"]))
    u_p = reshape(U_p[:, 0] / U_lid, nx, ny)[:, ix]
    u_m = reshape(U_m[:, 0] / U_lid, nx, ny)[:, ix]
    v_p = reshape(U_p[:, 1] / U_lid, nx, ny)[ny // 2, :]
    v_m = reshape(U_m[:, 1] / U_lid, nx, ny)[ny // 2, :]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    axes[0].plot(u_p, y_nd, label="Exact FP")
    axes[0].plot(u_m, y_nd, "--", label="ML-FP")
    axes[0].set_xlabel("u/U_lid at x/L≈0.5")
    axes[0].set_ylabel("y/L")
    axes[0].legend()
    axes[1].plot(x_nd, v_p, label="Exact FP")
    axes[1].plot(x_nd, v_m, "--", label="ML-FP")
    axes[1].set_xlabel("x/L")
    axes[1].set_ylabel("v/U_lid at y/L≈0.5")
    axes[1].legend()
    axes[2].plot(T_p2[:, ix], y_nd, label="Exact FP")
    axes[2].plot(T_m2[:, ix], y_nd, "--", label="ML-FP")
    axes[2].set_xlabel("T/Twall at x/L≈0.5")
    axes[2].set_ylabel("y/L")
    axes[2].legend()
    save_both(fig, outdir / "fp_centerlines")
    plt.close(fig)

    runtime_p = float(np.asarray(P["runtime_s"]))
    runtime_m = float(np.asarray(M["runtime_s"]))
    speedup = runtime_p / runtime_m if runtime_m > 0 else np.nan

    top_mask = np.repeat(y_nd[:, None] >= 0.8, nx, axis=1).ravel()
    lower_mask = np.repeat(y_nd[:, None] <= 0.4, nx, axis=1).ravel()
    checks = {
        "temperature_positive": bool(np.all(T_m > 0)),
        "density_positive": bool(np.all(rho_m_raw > 0)),
        "nonfinite_prediction_count": int(np.sum(~np.isfinite(U_m)) + np.sum(~np.isfinite(T_m)) + np.sum(~np.isfinite(rho_m_raw))),
        "mean_top_u_over_Ulid": float(np.mean(U_m[top_mask, 0] / U_lid)),
        "minimum_lower_u_over_Ulid": float(np.min(U_m[lower_mask, 0] / U_lid)),
        "physics_runtime_s": runtime_p,
        "ml_runtime_s": runtime_m,
        "online_speedup": speedup,
    }
    (outdir / "fp_physical_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    summary_lines = [
        "# Track 6 held-out cavity summary",
        "",
        f"- U_lid: {U_lid:g} m/s",
        f"- nominal Kn: {float(np.asarray(P['Kn'])):.4g}",
        f"- exact-FP runtime: {runtime_p:.3f} s",
        f"- ML-FP runtime: {runtime_m:.3f} s",
        f"- online speedup: {speedup:.3f}x",
        "",
        "## Numerical metrics",
    ]
    for row in rows:
        summary_lines.append(
            f"- {row['quantity']}: relative L2={100*row['relative_L2']:.3f}%, RMSE={row['RMSE']:.4e}"
        )
    summary_lines += ["", "## Physical checks"]
    for k, v in checks.items():
        summary_lines.append(f"- {k}: {v}")
    summary_lines += [
        "",
        "## Scope statement",
        "These are reduced educational runs. They test the complete train-export-deploy workflow, but do not replace the production-resolution convergence and independent-validation studies required for a conference or journal claim.",
    ]
    (outdir / "fp_project_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Saved analysis in", outdir)
    print("Online speedup:", speedup)
    for row in rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
