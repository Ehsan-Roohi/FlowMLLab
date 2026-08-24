"""Shared utilities for the M&I-ENG 690A unified cavity project.

The CFD solver is adapted from the Week-1 educational streamfunction-vorticity
notebook. It remains an educational finite-difference solver; quality gates and
benchmark checks are mandatory before a field is admitted to the shared dataset.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np


GHIA = {
    100: {
        "y": np.array([1.0000, .9766, .9688, .9609, .9531, .8516, .7344, .6172,
                       .5000, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0]),
        "u": np.array([1.0, .84123, .78871, .73722, .68717, .23151, .00332,
                       -.13641, -.20581, -.21090, -.15662, -.10150, -.06434,
                       -.04775, -.04192, -.03717, 0.0]),
        "x": np.array([1.0000, .9688, .9609, .9531, .9453, .9063, .8594, .8047,
                       .5000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.0]),
        "v": np.array([0.0, -.05906, -.07391, -.08864, -.10313, -.16914,
                       -.22445, -.24533, .05454, .17527, .17507, .16077,
                       .12317, .10890, .10091, .09233, 0.0]),
    },
    400: {
        "y": np.array([1.0000, .9766, .9688, .9609, .9531, .8516, .7344, .6172,
                       .5000, .4531, .2813, .1719, .1016, .0703, .0625, .0547, 0.0]),
        "u": np.array([1.0, .75837, .68439, .61756, .55892, .29093, .16256,
                       .02135, -.11477, -.17119, -.32726, -.24299, -.14612,
                       -.10338, -.09266, -.08186, 0.0]),
        "x": np.array([1.0000, .9688, .9609, .9531, .9453, .9063, .8594, .8047,
                       .5000, .2344, .2266, .1563, .0938, .0781, .0703, .0625, 0.0]),
        "v": np.array([0.0, -.12146, -.15663, -.19254, -.22847, -.23827,
                       -.44993, -.38598, .05186, .30174, .30203, .28124,
                       .22965, .20920, .19713, .18360, 0.0]),
    },
}


CASE_MAP = {
    1:  {"Re": 100, "split": "train"},
    2:  {"Re": 150, "split": "train"},
    3:  {"Re": 175, "split": "test"},
    4:  {"Re": 200, "split": "train"},
    5:  {"Re": 225, "split": "train"},
    6:  {"Re": 250, "split": "train"},
    7:  {"Re": 275, "split": "test"},
    8:  {"Re": 300, "split": "train"},
    9:  {"Re": 350, "split": "train"},
    10: {"Re": 375, "split": "test"},
    11: {"Re": 400, "split": "train"},
}


def build_grid(N: int):
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y, x[1] - x[0], y[1] - y[0]


def apply_vorticity_bc(psi, omega, U, dx, dy):
    omega[0, 1:-1] = -2.0 * psi[1, 1:-1] / dy**2
    omega[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / dy**2 - 2.0 * U / dy
    omega[1:-1, 0] = -2.0 * psi[1:-1, 1] / dx**2
    omega[1:-1, -1] = -2.0 * psi[1:-1, -2] / dx**2
    omega[0, 0] = .5 * (omega[0, 1] + omega[1, 0])
    omega[0, -1] = .5 * (omega[0, -2] + omega[1, -1])
    omega[-1, 0] = .5 * (omega[-1, 1] + omega[-2, 0])
    omega[-1, -1] = .5 * (omega[-1, -2] + omega[-2, -1])
    return omega


def solve_poisson(psi, omega, dx, dy, iterations=60):
    dx2, dy2 = dx**2, dy**2
    denom = 2.0 * (dx2 + dy2)
    for _ in range(iterations):
        old = psi.copy()
        psi[1:-1, 1:-1] = (
            (old[1:-1, 2:] + old[1:-1, :-2]) * dy2
            + (old[2:, 1:-1] + old[:-2, 1:-1]) * dx2
            + omega[1:-1, 1:-1] * dx2 * dy2
        ) / denom
        psi[[0, -1], :] = 0.0
        psi[:, [0, -1]] = 0.0
    return psi


def compute_velocity(psi, U, dx, dy):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dy)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dx)
    u[-1, :] = U
    u[:, [0, -1]] = 0.0
    v[[0, -1], :] = 0.0
    v[:, [0, -1]] = 0.0
    return u, v


def advance_vorticity(omega, u, v, Re, dt, dx, dy):
    old = omega.copy()
    dwdx = (old[1:-1, 2:] - old[1:-1, :-2]) / (2.0 * dx)
    dwdy = (old[2:, 1:-1] - old[:-2, 1:-1]) / (2.0 * dy)
    lap = (
        (old[1:-1, 2:] - 2.0 * old[1:-1, 1:-1] + old[1:-1, :-2]) / dx**2
        + (old[2:, 1:-1] - 2.0 * old[1:-1, 1:-1] + old[:-2, 1:-1]) / dy**2
    )
    conv = u[1:-1, 1:-1] * dwdx + v[1:-1, 1:-1] * dwdy
    omega[1:-1, 1:-1] = old[1:-1, 1:-1] + dt * (-conv + lap / Re)
    return omega, old


def run_cavity(Re=100, N=65, dt=1e-3, max_steps=20000, poisson_iters=60,
               U=1.0, check_every=250, min_steps=5000, tol=2e-6,
               consecutive_required=3, verbose=True):
    """Run the Week-1 solver with early stopping, safety checks, and metadata."""
    if Re <= 0 or N < 17 or dt <= 0:
        raise ValueError("Require Re>0, N>=17, and dt>0.")
    x, y, X, Y, dx, dy = build_grid(N)
    diffusion_limit = Re * min(dx, dy)**2 / 4.0
    if dt > .8 * diffusion_limit:
        warnings.warn(f"dt={dt:g} is close to/above the explicit diffusion limit {diffusion_limit:g}.")
    psi = np.zeros((N, N), dtype=float)
    omega = np.zeros_like(psi)
    omega = apply_vorticity_bc(psi, omega, U, dx, dy)
    history_steps, history_residual = [], []
    passed = 0
    converged = False
    final_step = max_steps
    for step in range(1, max_steps + 1):
        psi = solve_poisson(psi, omega, dx, dy, iterations=poisson_iters)
        omega = apply_vorticity_bc(psi, omega, U, dx, dy)
        u, v = compute_velocity(psi, U, dx, dy)
        omega, old = advance_vorticity(omega, u, v, Re, dt, dx, dy)
        omega = apply_vorticity_bc(psi, omega, U, dx, dy)
        if not np.isfinite(omega).all():
            raise FloatingPointError("Non-finite vorticity: reduce dt and inspect the run.")
        if step % check_every == 0:
            residual = np.linalg.norm(omega - old) / (np.linalg.norm(omega) + 1e-30)
            history_steps.append(step)
            history_residual.append(residual)
            if verbose:
                print(f"step {step:7d} | residual {residual:.3e}")
            passed = passed + 1 if (step >= min_steps and residual < tol) else 0
            if passed >= consecutive_required:
                converged, final_step = True, step
                break
    psi = solve_poisson(psi, omega, dx, dy, iterations=2 * poisson_iters)
    omega = apply_vorticity_bc(psi, omega, U, dx, dy)
    u, v = compute_velocity(psi, U, dx, dy)
    return {
        "x": x, "y": y, "X": X, "Y": Y, "psi": psi, "omega": omega,
        "u": u, "v": v, "Re": float(Re), "N": int(N), "dt": float(dt),
        "U": float(U), "steps": int(final_step), "converged": bool(converged),
        "residual_steps": np.asarray(history_steps),
        "residual_values": np.asarray(history_residual),
        "final_residual": float(history_residual[-1]) if history_residual else np.nan,
        "poisson_iters": int(poisson_iters),
    }


def centerline_profiles(result):
    mid = len(result["x"]) // 2
    return result["u"][:, mid] / result["U"], result["v"][mid, :] / result["U"]


def ghia_errors(result):
    Re_key = int(round(float(result["Re"])))
    if Re_key not in GHIA:
        return np.nan, np.nan
    ref = GHIA[Re_key]
    uc, vc = centerline_profiles(result)
    ui = np.interp(ref["y"], result["y"], uc)
    vi = np.interp(ref["x"], result["x"], vc)
    return (float(np.linalg.norm(ui-ref["u"]) / np.linalg.norm(ref["u"])),
            float(np.linalg.norm(vi-ref["v"]) / np.linalg.norm(ref["v"])))


def physical_metrics(result):
    x, y, u, v, psi = result["x"], result["y"], result["u"], result["v"], result["psi"]
    dx, dy = x[1]-x[0], y[1]-y[0]
    div = np.gradient(u, dx, axis=1) + np.gradient(v, dy, axis=0)
    interior = div[1:-1, 1:-1]
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    eu, ev = ghia_errors(result)
    return {
        "div_l2": float(np.sqrt(np.mean(interior**2))),
        "div_linf": float(np.max(np.abs(interior))),
        "vortex_x": float(x[i]), "vortex_y": float(y[j]),
        "psi_min": float(psi[j, i]), "speed_max": float(np.max(np.hypot(u, v))),
        "ghia_Eu": eu, "ghia_Ev": ev,
    }


def quality_gate(result, residual_limit=1e-5, ghia_limit=0.20):
    m = physical_metrics(result)
    checks = {
        "finite_fields": bool(all(np.isfinite(result[k]).all() for k in ("u", "v", "psi", "omega"))),
        "residual_below_limit": bool(result["final_residual"] < residual_limit),
        "reasonable_speed": bool(m["speed_max"] <= 1.25 * result["U"]),
    }
    if int(round(result["Re"])) in GHIA:
        checks["Ghia_Eu"] = bool(m["ghia_Eu"] < ghia_limit)
        checks["Ghia_Ev"] = bool(m["ghia_Ev"] < ghia_limit)
    return {"accepted": bool(all(checks.values())), "checks": checks, "metrics": m}


def save_case(result, student_id, split, outdir="case_outputs"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    Re, N = int(round(result["Re"])), int(result["N"])
    stem = f"cavity_Re{Re:04d}_N{N:03d}_student{int(student_id):02d}"
    q = quality_gate(result)
    np.savez_compressed(out / f"{stem}.npz",
        x=result["x"], y=result["y"], u=result["u"], v=result["v"],
        psi=result["psi"], omega=result["omega"], Re=result["Re"], N=result["N"],
        dt=result["dt"], U=result["U"], steps=result["steps"],
        final_residual=result["final_residual"], split=str(split), student_id=int(student_id))
    meta = {"student_id": int(student_id), "split": str(split), "Re": Re, "N": N,
            "dt": result["dt"], "steps": result["steps"],
            "final_residual": result["final_residual"], **q}
    (out / f"{stem}_quality.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out / f"{stem}.npz", meta


def load_case(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def merge_cases(paths, output="combined_cavity_dataset.npz", require_accepted=True):
    records = []
    for p in sorted(map(Path, paths)):
        qpath = p.with_name(p.stem + "_quality.json")
        if require_accepted:
            if not qpath.exists() or not json.loads(qpath.read_text())["accepted"]:
                print("SKIP unaccepted:", p.name); continue
        records.append(load_case(p))
    if not records:
        raise ValueError("No accepted case files were found.")
    x0, y0 = records[0]["x"], records[0]["y"]
    for r in records:
        if r["u"].shape != records[0]["u"].shape or not np.allclose(r["x"], x0) or not np.allclose(r["y"], y0):
            raise ValueError("Production cases must use the same grid. Keep grid-study files separate.")
    order = np.argsort([float(r["Re"]) for r in records])
    records = [records[i] for i in order]
    data = {"x": x0, "y": y0,
            "Re": np.array([float(r["Re"]) for r in records]),
            "u": np.stack([r["u"] for r in records]),
            "v": np.stack([r["v"] for r in records]),
            "psi": np.stack([r["psi"] for r in records]),
            "omega": np.stack([r["omega"] for r in records]),
            "split": np.array([str(r["split"]) for r in records]),
            "student_id": np.array([int(r["student_id"]) for r in records])}
    np.savez_compressed(output, **data)
    return data


def load_dataset(path="combined_cavity_dataset.npz"):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def field_errors(true_u, true_v, pred_u, pred_v):
    du, dv = pred_u-true_u, pred_v-true_v
    denom = np.sqrt(np.sum(true_u**2 + true_v**2)) + 1e-30
    return {"relative_L2_uv": float(np.sqrt(np.sum(du**2+dv**2))/denom),
            "MAE_u": float(np.mean(np.abs(du))), "MAE_v": float(np.mean(np.abs(dv))),
            "max_vector_error": float(np.max(np.hypot(du, dv)))}


def save_predictions(path, Re, x, y, u_true, v_true, u_pred, v_pred, model_name):
    np.savez_compressed(path, Re=float(Re), x=x, y=y, u_true=u_true, v_true=v_true,
                        u_pred=u_pred, v_pred=v_pred, model_name=str(model_name))
