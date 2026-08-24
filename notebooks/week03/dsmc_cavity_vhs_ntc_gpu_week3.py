#!/usr/bin/env python3
"""GPU-first DSMC lid-driven cavity solver for the Week-3 optional extension.

Physical model
--------------
* monatomic argon in a square micro/nano cavity;
* Variable Hard Sphere (VHS) collision kernel;
* Bird-style No Time Counter (NTC) candidate selection;
* fully diffuse, fully accommodating walls;
* isothermal walls by default; the top wall moves in +x;
* particle-time sampling of density, velocity, temperature, pressure;
* incident+reflected surface sampling on the moving lid;
* vorticity, streamfunction, and primary-vortex diagnostics.

The default physical setup is the isothermal Mohammadzadeh-type cavity:
Kn=0.05, L=1 micrometre, T=300 K, and U_wall=100 m/s.

Production runs are GPU-only and require CuPy.  ``--debug-cpu`` exists only
for tiny syntax/development checks and must not be used for assignment results.

Examples
--------
GPU preflight:
    python -u dsmc_cavity_vhs_ntc_gpu_week3.py --check-gpu

Smoke test:
    python -u dsmc_cavity_vhs_ntc_gpu_week3.py --Kn 0.05 --nx 40 --ny 40 \
        --ppc 8 --steps 500 --sample-start 250 --case-name smoke

Week-3 long run:
    python -u dsmc_cavity_vhs_ntc_gpu_week3.py --Kn 0.05 --Uwall 100 \
        --nx 80 --ny 80 --ppc 32 --steps 20000 --sample-start 5000 \
        --sample-stride 5 --smoothing-passes 3 --case-name final_long
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

KB = 1.380649e-23
PI = math.pi


def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/CuPy values to a NumPy array without importing CuPy eagerly."""
    try:
        import cupy as cp  # type: ignore

        if isinstance(value, cp.ndarray):
            return cp.asnumpy(value)
        if isinstance(value, cp.generic):
            return np.asarray(value.get())
    except Exception:
        pass
    return np.asarray(value)


def _scalar(value: Any) -> float:
    arr = _to_numpy(value)
    return float(arr.reshape(-1)[0])


def get_backend(debug_cpu: bool = False):
    """Return (array_module, backend_name)."""
    if debug_cpu:
        print("[WARNING] --debug-cpu is for tiny development checks only.")
        return np, "numpy-debug"
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on CUDA environment
        raise RuntimeError(
            "CuPy/CUDA is required for production runs. Select a GPU runtime and "
            "install cupy-cuda12x (or the matching CUDA build)."
        ) from exc
    try:
        n_gpu = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("CuPy is installed, but no usable CUDA GPU was found.") from exc
    if n_gpu < 1:  # pragma: no cover
        raise RuntimeError("No visible CUDA GPU. Do not run production cases on a CPU node.")
    return cp, "cupy"


def check_gpu() -> None:
    """Print a concise GPU preflight report and execute a small device calculation."""
    try:
        import cupy as cp  # type: ignore

        n_gpu = int(cp.cuda.runtime.getDeviceCount())
        print("CuPy version:", cp.__version__)
        print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())
        print("Visible GPUs:", n_gpu)
        for device_id in range(n_gpu):
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            print(f"GPU {device_id}: {name}")
        a = cp.arange(1_000_000, dtype=cp.float64)
        result = float(cp.asnumpy(cp.sum(a * a)))
        print("Device test sum:", f"{result:.6e}")
        print("GPU preflight: PASS")
    except Exception as exc:
        print("GPU preflight: FAIL")
        print(type(exc).__name__ + ":", exc)
        raise SystemExit(2)


@dataclass
class DSMCConfig:
    # Physical case
    Kn: float = 0.05
    L: float = 1.0e-6
    T0: float = 300.0
    Uwall: float = 100.0
    m: float = 6.63e-26
    d_ref: float = 4.17e-10
    T_ref: float = 300.0
    omega: float = 0.81
    depth_factor: float = 1.0

    # Numerical setup
    nx: int = 80
    ny: int = 80
    ppc: int = 32
    steps: int = 20_000
    sample_start: int = 5_000
    sample_stride: int = 5
    dt_factor: float = 0.20
    gmax_safety: float = 1.35
    smoothing_passes: int = 3
    streamfunction_iterations: int = 5_000

    # Reproducibility/output
    seed: int = 7
    output_dir: str = "week3_gpu_output"
    case_name: str = "mohammadzadeh_Kn0p05_long"
    progress_every: int = 500
    monitor_every: int = 1_000
    debug_cpu: bool = False

    @property
    def R(self) -> float:
        return KB / self.m

    @property
    def gamma(self) -> float:
        return 5.0 / 3.0

    @property
    def lambda0(self) -> float:
        return self.Kn * self.L

    @property
    def n0(self) -> float:
        # Standard HS-like reference relation used to set the intended global Kn.
        return 1.0 / (math.sqrt(2.0) * PI * self.d_ref**2 * self.lambda0)

    @property
    def c_mean(self) -> float:
        return math.sqrt(8.0 * self.R * self.T0 / PI)

    @property
    def c_mp(self) -> float:
        return math.sqrt(2.0 * self.R * self.T0)

    @property
    def dx(self) -> float:
        return self.L / self.nx

    @property
    def dy(self) -> float:
        return self.L / self.ny

    @property
    def depth(self) -> float:
        return self.depth_factor * self.L

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.depth

    @property
    def dt(self) -> float:
        length_scale = min(self.dx, self.dy, self.lambda0)
        return self.dt_factor * length_scale / max(self.c_mean + abs(self.Uwall), 1.0e-30)

    @property
    def Np(self) -> int:
        return int(self.nx * self.ny * self.ppc)

    @property
    def particle_weight(self) -> float:
        volume = self.L * self.L * self.depth
        return self.n0 * volume / self.Np

    @property
    def tau_collision_est(self) -> float:
        return self.lambda0 / max(self.c_mean, 1.0e-30)

    def validate(self) -> None:
        if self.Kn <= 0 or self.L <= 0 or self.T0 <= 0:
            raise ValueError("Kn, L, and T0 must be positive.")
        if self.nx < 4 or self.ny < 4 or self.ppc < 2:
            raise ValueError("Use nx, ny >= 4 and ppc >= 2.")
        if self.steps < 1:
            raise ValueError("steps must be positive.")
        if not (0 <= self.sample_start < self.steps):
            raise ValueError("sample_start must satisfy 0 <= sample_start < steps.")
        if self.sample_stride < 1 or self.progress_every < 1 or self.monitor_every < 1:
            raise ValueError("stride/progress/monitor intervals must be positive.")
        if self.omega < 0.5 or self.omega > 1.0:
            raise ValueError("For this teaching solver, use 0.5 <= omega <= 1.0.")
        if self.debug_cpu and self.Np > 20_000:
            raise ValueError("CPU debug mode is capped at 20,000 particles.")


def vhs_sigma_g(g: Any, cfg: DSMCConfig, xp) -> Any:
    """Return sigma_T(g)*g for the VHS teaching model."""
    sigma_ref = PI * cfg.d_ref**2
    alpha = 2.0 * (cfg.omega - 0.5)
    g_ref = math.sqrt(4.0 * KB * cfg.T_ref / cfg.m)
    g_safe = xp.maximum(g, 1.0e-30)
    return sigma_ref * (g_ref / g_safe) ** alpha * g_safe


def initial_majorant(cfg: DSMCConfig) -> float:
    # Maxwellian tails are unbounded. Ten thermal standard deviations plus lid speed
    # is a deliberately conservative initial bound, updated adaptively if exceeded.
    g_bound = 2.0 * (abs(cfg.Uwall) + 10.0 * math.sqrt(cfg.R * cfg.T0))
    value = vhs_sigma_g(np.asarray([g_bound]), cfg, np)[0]
    return float(value) * cfg.gmax_safety


def initialize_particles(cfg: DSMCConfig, xp) -> Tuple[Any, Any]:
    xp.random.seed(cfg.seed)
    x = xp.empty((cfg.Np, 2), dtype=xp.float64)
    x[:, 0] = cfg.L * xp.random.random(cfg.Np)
    x[:, 1] = cfg.L * xp.random.random(cfg.Np)
    v = math.sqrt(cfg.R * cfg.T0) * xp.random.standard_normal((cfg.Np, 3))
    # Remove the finite-sample initial drift without changing thermal fluctuations much.
    v -= xp.mean(v, axis=0, keepdims=True)
    return x, v


def create_wall_accumulator(cfg: DSMCConfig, xp) -> Dict[str, Any]:
    return {
        "sum_w": xp.zeros(cfg.nx, dtype=xp.float64),
        "sum_wvx": xp.zeros(cfg.nx, dtype=xp.float64),
        "sum_wvy": xp.zeros(cfg.nx, dtype=xp.float64),
        "sum_wvz": xp.zeros(cfg.nx, dtype=xp.float64),
        "sum_wv2": xp.zeros(cfg.nx, dtype=xp.float64),
        "hits": xp.zeros(cfg.nx, dtype=xp.float64),
    }


def accumulate_top_surface(x_hit: Any, velocities: Any, cfg: DSMCConfig, wall_accum: Dict[str, Any], xp) -> None:
    if int(velocities.shape[0]) == 0:
        return
    bins = xp.clip((x_hit / cfg.dx).astype(xp.int64), 0, cfg.nx - 1)
    vn = xp.maximum(xp.abs(velocities[:, 1]), 1.0e-20)
    weights = 1.0 / vn
    wall_accum["sum_w"] += xp.bincount(bins, weights=weights, minlength=cfg.nx)
    wall_accum["sum_wvx"] += xp.bincount(bins, weights=weights * velocities[:, 0], minlength=cfg.nx)
    wall_accum["sum_wvy"] += xp.bincount(bins, weights=weights * velocities[:, 1], minlength=cfg.nx)
    wall_accum["sum_wvz"] += xp.bincount(bins, weights=weights * velocities[:, 2], minlength=cfg.nx)
    wall_accum["sum_wv2"] += xp.bincount(
        bins, weights=weights * xp.sum(velocities * velocities, axis=1), minlength=cfg.nx
    )
    wall_accum["hits"] += xp.bincount(bins, minlength=cfg.nx)


def _diffuse_velocity(n: int, wall: str, cfg: DSMCConfig, xp) -> Any:
    if n <= 0:
        return xp.empty((0, 3), dtype=xp.float64)
    sigma = math.sqrt(cfg.R * cfg.T0)
    normal = xp.sqrt(-2.0 * cfg.R * cfg.T0 * xp.log(xp.maximum(xp.random.random(n), 1.0e-15)))
    t1 = sigma * xp.random.standard_normal(n)
    t2 = sigma * xp.random.standard_normal(n)
    out = xp.empty((n, 3), dtype=xp.float64)
    if wall == "left":
        out[:, 0], out[:, 1], out[:, 2] = normal, t1, t2
    elif wall == "right":
        out[:, 0], out[:, 1], out[:, 2] = -normal, t1, t2
    elif wall == "bottom":
        out[:, 0], out[:, 1], out[:, 2] = t1, normal, t2
    elif wall == "top":
        out[:, 0], out[:, 1], out[:, 2] = cfg.Uwall + t1, -normal, t2
    else:
        raise ValueError(f"Unknown wall {wall!r}")
    return out


def move_and_reflect(
    x: Any,
    v: Any,
    cfg: DSMCConfig,
    xp,
    wall_accum: Dict[str, Any],
    collect_wall: bool,
) -> None:
    x += v[:, :2] * cfg.dt

    left = x[:, 0] < 0.0
    n = int(_scalar(xp.sum(left)))
    if n:
        idx = xp.where(left)[0]
        x[idx, 0] = -x[idx, 0]
        v[idx] = _diffuse_velocity(n, "left", cfg, xp)

    right = x[:, 0] > cfg.L
    n = int(_scalar(xp.sum(right)))
    if n:
        idx = xp.where(right)[0]
        x[idx, 0] = 2.0 * cfg.L - x[idx, 0]
        v[idx] = _diffuse_velocity(n, "right", cfg, xp)

    bottom = x[:, 1] < 0.0
    n = int(_scalar(xp.sum(bottom)))
    if n:
        idx = xp.where(bottom)[0]
        x[idx, 1] = -x[idx, 1]
        v[idx] = _diffuse_velocity(n, "bottom", cfg, xp)

    top = x[:, 1] > cfg.L
    n = int(_scalar(xp.sum(top)))
    if n:
        idx = xp.where(top)[0]
        x_hit = x[idx, 0].copy()
        v_in = v[idx].copy()
        x[idx, 1] = 2.0 * cfg.L - x[idx, 1]
        v_out = _diffuse_velocity(n, "top", cfg, xp)
        v[idx] = v_out
        if collect_wall:
            # Surface-based estimate: count incident and reflected states.
            accumulate_top_surface(x_hit, v_in, cfg, wall_accum, xp)
            accumulate_top_surface(x_hit, v_out, cfg, wall_accum, xp)

    # Guard against extremely rare multi-wall overshoots in a debug/poor-dt case.
    x[:, 0] = xp.clip(x[:, 0], 0.0, cfg.L)
    x[:, 1] = xp.clip(x[:, 1], 0.0, cfg.L)


def cell_keys(x: Any, cfg: DSMCConfig, xp) -> Any:
    ix = xp.clip((x[:, 0] / cfg.dx).astype(xp.int64), 0, cfg.nx - 1)
    iy = xp.clip((x[:, 1] / cfg.dy).astype(xp.int64), 0, cfg.ny - 1)
    return ix + cfg.nx * iy


def collide_ntc_vectorized(
    x: Any,
    v: Any,
    cfg: DSMCConfig,
    xp,
    sg_majorant: float,
) -> Tuple[Any, Any, float, Dict[str, float]]:
    """Vectorized NTC-like collision stage over all cells.

    Accepted pairs that share a particle are filtered with a random-priority
    matching step. This prevents conflicting GPU writes and preserves pairwise
    momentum and kinetic energy. The discarded-conflict fraction is reported;
    it should remain small for a suitable time step.
    """
    ncell = cfg.nx * cfg.ny
    keys = cell_keys(x, cfg, xp)
    order = xp.argsort(keys)
    x = x[order]
    v = v[order]
    keys = keys[order]

    counts = xp.bincount(keys, minlength=ncell).astype(xp.int64)
    offsets = xp.empty(ncell, dtype=xp.int64)
    offsets[0] = 0
    offsets[1:] = xp.cumsum(counts[:-1])

    expected = (
        0.5
        * counts.astype(xp.float64)
        * xp.maximum(counts - 1, 0).astype(xp.float64)
        * cfg.particle_weight
        * sg_majorant
        * cfg.dt
        / cfg.cell_volume
    )
    n_candidates = xp.floor(expected).astype(xp.int64)
    n_candidates += (xp.random.random(ncell) < (expected - n_candidates)).astype(xp.int64)
    n_candidates = xp.where(counts >= 2, n_candidates, 0)
    total_candidates = int(_scalar(xp.sum(n_candidates)))

    stats = {
        "candidates": float(total_candidates),
        "accepted_before_conflict": 0.0,
        "executed": 0.0,
        "conflict_discarded": 0.0,
        "majorant_exceed": 0.0,
        "max_sigma_g": 0.0,
    }
    if total_candidates == 0:
        return x, v, sg_majorant, stats

    candidate_cells = xp.repeat(xp.arange(ncell, dtype=xp.int64), n_candidates)
    local_count = counts[candidate_cells]
    base = offsets[candidate_cells]
    i_local = (xp.random.random(total_candidates) * local_count).astype(xp.int64)
    j_local = (xp.random.random(total_candidates) * local_count).astype(xp.int64)
    j_local = xp.where(j_local == i_local, (j_local + 1) % local_count, j_local)
    p1 = base + i_local
    p2 = base + j_local

    relative = v[p1] - v[p2]
    g = xp.linalg.norm(relative, axis=1)
    sigma_g = vhs_sigma_g(g, cfg, xp)
    majorant_used = sg_majorant
    max_observed = _scalar(xp.max(sigma_g))
    stats["max_sigma_g"] = max_observed
    exceed = sigma_g > majorant_used
    stats["majorant_exceed"] = _scalar(xp.sum(exceed))
    if max_observed > majorant_used:
        sg_majorant = max_observed * cfg.gmax_safety

    probability = xp.minimum(1.0, sigma_g / max(majorant_used, 1.0e-300))
    accepted = xp.random.random(total_candidates) < probability
    n_accepted = int(_scalar(xp.sum(accepted)))
    stats["accepted_before_conflict"] = float(n_accepted)
    if n_accepted == 0:
        return x, v, sg_majorant, stats

    p1a = p1[accepted]
    p2a = p2[accepted]
    ga = g[accepted]

    # Random-priority conflict filter: a particle participates in at most one
    # executed collision during this stage.
    priority = xp.random.random(n_accepted)
    max_priority = xp.full(cfg.Np, -1.0, dtype=xp.float64)
    xp.maximum.at(max_priority, p1a, priority)
    xp.maximum.at(max_priority, p2a, priority)
    keep = (priority >= max_priority[p1a]) & (priority >= max_priority[p2a])
    n_execute = int(_scalar(xp.sum(keep)))
    stats["executed"] = float(n_execute)
    stats["conflict_discarded"] = float(n_accepted - n_execute)
    if n_execute == 0:
        return x, v, sg_majorant, stats

    p1e = p1a[keep]
    p2e = p2a[keep]
    ge = ga[keep]
    center = 0.5 * (v[p1e] + v[p2e])
    mu = 2.0 * xp.random.random(n_execute) - 1.0
    phi = 2.0 * PI * xp.random.random(n_execute)
    sin_theta = xp.sqrt(xp.maximum(0.0, 1.0 - mu * mu))
    new_relative = xp.stack(
        (
            ge * sin_theta * xp.cos(phi),
            ge * sin_theta * xp.sin(phi),
            ge * mu,
        ),
        axis=1,
    )
    v[p1e] = center + 0.5 * new_relative
    v[p2e] = center - 0.5 * new_relative
    return x, v, sg_majorant, stats


def create_field_accumulator(cfg: DSMCConfig, xp) -> Dict[str, Any]:
    ncell = cfg.nx * cfg.ny
    return {
        "count": xp.zeros(ncell, dtype=xp.float64),
        "sum_vx": xp.zeros(ncell, dtype=xp.float64),
        "sum_vy": xp.zeros(ncell, dtype=xp.float64),
        "sum_vz": xp.zeros(ncell, dtype=xp.float64),
        "sum_v2": xp.zeros(ncell, dtype=xp.float64),
        "nsamples": 0,
    }


def sample_fields(x: Any, v: Any, cfg: DSMCConfig, accum: Dict[str, Any], xp) -> None:
    keys = cell_keys(x, cfg, xp)
    ncell = cfg.nx * cfg.ny
    accum["count"] += xp.bincount(keys, minlength=ncell)
    accum["sum_vx"] += xp.bincount(keys, weights=v[:, 0], minlength=ncell)
    accum["sum_vy"] += xp.bincount(keys, weights=v[:, 1], minlength=ncell)
    accum["sum_vz"] += xp.bincount(keys, weights=v[:, 2], minlength=ncell)
    accum["sum_v2"] += xp.bincount(keys, weights=xp.sum(v * v, axis=1), minlength=ncell)
    accum["nsamples"] += 1


def current_u_field(accum: Dict[str, Any], cfg: DSMCConfig, xp) -> Any:
    count = xp.maximum(accum["count"], 1.0)
    return (accum["sum_vx"] / count).reshape(cfg.ny, cfg.nx)


def current_surface_slip(wall_accum: Dict[str, Any], cfg: DSMCConfig, xp) -> Any:
    weight = xp.maximum(wall_accum["sum_w"], 1.0e-300)
    u_wall_gas = wall_accum["sum_wvx"] / weight
    return (cfg.Uwall - u_wall_gas) / cfg.Uwall


def relative_change(current: Any, previous: Optional[np.ndarray]) -> float:
    current_np = _to_numpy(current).astype(float)
    if previous is None:
        return float("nan")
    return float(np.linalg.norm(current_np - previous) / (np.linalg.norm(current_np) + 1.0e-14))


def smooth2d(field: np.ndarray, passes: int) -> np.ndarray:
    out = np.asarray(field, dtype=float).copy()
    for _ in range(max(0, passes)):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 5.0
    return out


def finalize_fields(accum: Dict[str, Any], cfg: DSMCConfig) -> Dict[str, np.ndarray]:
    nsamples = max(int(accum["nsamples"]), 1)
    count_total = _to_numpy(accum["count"]).astype(float)
    safe = np.maximum(count_total, 1.0)
    ux = _to_numpy(accum["sum_vx"]) / safe
    uy = _to_numpy(accum["sum_vy"]) / safe
    uz = _to_numpy(accum["sum_vz"]) / safe
    mean_v2 = _to_numpy(accum["sum_v2"]) / safe
    temperature = np.maximum((mean_v2 - (ux**2 + uy**2 + uz**2)) / (3.0 * cfg.R), 1.0e-12)
    count_mean = count_total / nsamples
    rho = count_mean * cfg.particle_weight * cfg.m / cfg.cell_volume
    pressure = rho * cfg.R * temperature

    def shape(a: np.ndarray) -> np.ndarray:
        return a.reshape(cfg.ny, cfg.nx)

    return {
        "count_mean_raw": shape(count_mean),
        "rho_raw": shape(rho),
        "u_raw": shape(ux),
        "v_raw": shape(uy),
        "w_raw": shape(uz),
        "T_raw": shape(temperature),
        "p_raw": shape(pressure),
    }


def finalize_surface_profiles(
    wall_accum: Dict[str, Any], fields: Dict[str, np.ndarray], cfg: DSMCConfig
) -> Dict[str, np.ndarray]:
    sum_w = np.maximum(_to_numpy(wall_accum["sum_w"]).astype(float), 1.0e-300)
    ux = _to_numpy(wall_accum["sum_wvx"]) / sum_w
    uy = _to_numpy(wall_accum["sum_wvy"]) / sum_w
    uz = _to_numpy(wall_accum["sum_wvz"]) / sum_w
    mean_v2 = _to_numpy(wall_accum["sum_wv2"]) / sum_w
    T_surface = np.maximum((mean_v2 - (ux**2 + uy**2 + uz**2)) / (3.0 * cfg.R), 1.0e-12)
    x_lid = (np.arange(cfg.nx) + 0.5) / cfg.nx
    u_adj = fields["u_raw"][-1, :]
    T_adj = fields["T_raw"][-1, :]
    return {
        "x_over_L": x_lid,
        "u_surface": ux,
        "uslip_surface": cfg.Uwall - ux,
        "T_surface": T_surface,
        "Tjump_surface": T_surface - cfg.T0,
        "u_adjacent_cell": u_adj,
        "uslip_adjacent_cell": cfg.Uwall - u_adj,
        "T_adjacent_cell": T_adj,
        "Tjump_adjacent_cell": T_adj - cfg.T0,
        "surface_samples": _to_numpy(wall_accum["hits"]).astype(float),
    }


def vorticity(u: np.ndarray, v: np.ndarray, cfg: DSMCConfig) -> np.ndarray:
    return np.gradient(v, cfg.dx, axis=1) - np.gradient(u, cfg.dy, axis=0)


def solve_streamfunction(omega: np.ndarray, cfg: DSMCConfig) -> Tuple[np.ndarray, int, float]:
    """Solve Laplacian(psi)=-omega with psi=0 on the cavity boundary."""
    psi = np.zeros_like(omega, dtype=float)
    dx2, dy2 = cfg.dx**2, cfg.dy**2
    denominator = 2.0 * (dx2 + dy2)
    final_change = float("inf")
    for iteration in range(1, cfg.streamfunction_iterations + 1):
        old = psi.copy()
        psi[1:-1, 1:-1] = (
            dy2 * (old[1:-1, 2:] + old[1:-1, :-2])
            + dx2 * (old[2:, 1:-1] + old[:-2, 1:-1])
            + omega[1:-1, 1:-1] * dx2 * dy2
        ) / denominator
        if iteration % 100 == 0 or iteration == cfg.streamfunction_iterations:
            final_change = float(
                np.linalg.norm(psi - old) / (np.linalg.norm(psi) + 1.0e-30)
            )
            if final_change < 1.0e-9:
                return psi, iteration, final_change
    return psi, cfg.streamfunction_iterations, final_change


def find_vortex_center(psi: np.ndarray, cfg: DSMCConfig) -> Tuple[float, float, float]:
    margin_x = max(2, cfg.nx // 20)
    margin_y = max(2, cfg.ny // 20)
    interior = psi[margin_y:-margin_y, margin_x:-margin_x]
    if interior.size == 0:
        j, i = np.unravel_index(np.argmax(np.abs(psi)), psi.shape)
    else:
        jj, ii = np.unravel_index(np.argmax(np.abs(interior)), interior.shape)
        j, i = jj + margin_y, ii + margin_x
    return (i + 0.5) / cfg.nx, (j + 0.5) / cfg.ny, float(psi[j, i])


def write_csv(path: Path, header: List[str], columns: List[np.ndarray]) -> None:
    table = np.column_stack(columns)
    np.savetxt(path, table, delimiter=",", header=",".join(header), comments="")


def save_plots(
    case_dir: Path,
    fields: Dict[str, np.ndarray],
    profiles: Dict[str, np.ndarray],
    cfg: DSMCConfig,
    vortex_center: Tuple[float, float, float],
) -> None:
    import matplotlib.pyplot as plt

    x = (np.arange(cfg.nx) + 0.5) / cfg.nx
    y = (np.arange(cfg.ny) + 0.5) / cfg.ny
    X, Y = np.meshgrid(x, y)

    plot_specs = [
        ("u_raw", fields["u_raw"] / cfg.Uwall, r"$u/U_{wall}$ (raw average)"),
        ("u_smooth", fields["u_smooth"] / cfg.Uwall, r"$u/U_{wall}$ (visual smoothing)"),
        ("v_smooth", fields["v_smooth"] / cfg.Uwall, r"$v/U_{wall}$"),
        ("T_smooth", fields["T_smooth"] / cfg.T0, r"$T/T_0$"),
        ("rho_smooth", fields["rho_smooth"] / np.mean(fields["rho_smooth"]), r"$\rho/\langle\rho\rangle$"),
        ("vorticity", fields["vorticity"], "vorticity [1/s]"),
        ("streamfunction", fields["streamfunction"], r"streamfunction $\psi$ [m$^2$/s]"),
    ]
    for filename, data, label in plot_specs:
        fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=160)
        contour = ax.contourf(X, Y, data, levels=32)
        fig.colorbar(contour, ax=ax, label=label)
        ax.set(xlabel="x/L", ylabel="y/L", title=f"{label}; Kn={cfg.Kn:g}", aspect="equal")
        if filename == "streamfunction":
            ax.plot(vortex_center[0], vortex_center[1], "kx", ms=8, mew=2, label="primary vortex")
            ax.legend()
        fig.tight_layout()
        fig.savefig(case_dir / f"{filename}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    speed = np.hypot(fields["u_smooth"], fields["v_smooth"])
    ax.streamplot(X, Y, fields["u_smooth"], fields["v_smooth"], color=speed, density=1.35)
    ax.plot(vortex_center[0], vortex_center[1], "rx", ms=9, mew=2, label="primary vortex")
    ax.set(xlabel="x/L", ylabel="y/L", title=f"Molecular cavity streamlines; Kn={cfg.Kn:g}", aspect="equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "streamlines_vortex.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
    ax.plot(profiles["x_over_L"], profiles["uslip_surface"] / cfg.Uwall, "o", ms=3, label="surface estimate")
    ax.plot(profiles["x_over_L"], profiles["uslip_adjacent_cell"] / cfg.Uwall, "-", lw=1.5, label="adjacent-cell estimate")
    ax.set(xlabel="x/L", ylabel=r"$u_{slip}/U_{wall}$", title="Moving-lid velocity slip")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "lid_slip.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=160)
    ax.plot(profiles["x_over_L"], profiles["T_surface"], "o", ms=3, label="surface estimate")
    ax.plot(profiles["x_over_L"], profiles["T_adjacent_cell"], "-", lw=1.5, label="adjacent-cell estimate")
    ax.axhline(cfg.T0, ls="--", lw=1.0, label="wall temperature")
    ax.set(xlabel="x/L", ylabel="gas temperature [K]", title="Gas temperature along moving lid")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(case_dir / "lid_temperature.png")
    plt.close(fig)


def run(cfg: DSMCConfig) -> Dict[str, Any]:
    cfg.validate()
    xp, backend_name = get_backend(cfg.debug_cpu)
    case_dir = Path(cfg.output_dir) / cfg.case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Week-3 GPU DSMC molecular cavity")
    print("Backend:", backend_name)
    print(json.dumps(asdict(cfg), indent=2))
    print(
        f"lambda={cfg.lambda0:.4e} m, n0={cfg.n0:.4e} 1/m^3, "
        f"dt={cfg.dt:.4e} s, Np={cfg.Np:,}, Fn={cfg.particle_weight:.4e}"
    )
    print(
        f"dx/lambda={cfg.dx/cfg.lambda0:.4f}, dy/lambda={cfg.dy/cfg.lambda0:.4f}, "
        f"dt/tau_est={cfg.dt/cfg.tau_collision_est:.4f}"
    )
    print("Output:", case_dir)
    print("=" * 78)

    x, v = initialize_particles(cfg, xp)
    field_accum = create_field_accumulator(cfg, xp)
    wall_accum = create_wall_accumulator(cfg, xp)
    sg_majorant = initial_majorant(cfg)

    collision_totals = {
        "candidates": 0.0,
        "accepted_before_conflict": 0.0,
        "executed": 0.0,
        "conflict_discarded": 0.0,
        "majorant_exceed": 0.0,
    }
    history: List[Dict[str, float]] = []
    previous_centerline: Optional[np.ndarray] = None
    previous_slip: Optional[np.ndarray] = None
    start_time = time.perf_counter()

    for step0 in range(cfg.steps):
        step = step0 + 1
        collect_wall = step0 >= cfg.sample_start
        move_and_reflect(x, v, cfg, xp, wall_accum, collect_wall)
        x, v, sg_majorant, collision_stats = collide_ntc_vectorized(x, v, cfg, xp, sg_majorant)
        for key in collision_totals:
            collision_totals[key] += collision_stats[key]

        if step0 >= cfg.sample_start and (step0 - cfg.sample_start) % cfg.sample_stride == 0:
            sample_fields(x, v, cfg, field_accum, xp)

        if step % cfg.monitor_every == 0 and field_accum["nsamples"] > 0:
            u_running = current_u_field(field_accum, cfg, xp)
            centerline = _to_numpy(u_running[:, cfg.nx // 2] / cfg.Uwall)
            slip = _to_numpy(current_surface_slip(wall_accum, cfg, xp))
            elapsed = time.perf_counter() - start_time
            center_change = relative_change(centerline, previous_centerline)
            slip_change = relative_change(slip, previous_slip)
            previous_centerline = centerline.copy()
            previous_slip = slip.copy()
            history.append(
                {
                    "step": float(step),
                    "elapsed_s": elapsed,
                    "field_samples": float(field_accum["nsamples"]),
                    "centerline_relative_change": center_change,
                    "lid_slip_relative_change": slip_change,
                    "cumulative_executed_collisions": collision_totals["executed"],
                    "conflict_discard_fraction": collision_totals["conflict_discarded"]
                    / max(collision_totals["accepted_before_conflict"], 1.0),
                    "majorant_exceed_fraction": collision_totals["majorant_exceed"]
                    / max(collision_totals["candidates"], 1.0),
                }
            )

        if step % cfg.progress_every == 0 or step == cfg.steps:
            elapsed = time.perf_counter() - start_time
            rate = step / max(elapsed, 1.0e-12)
            remaining = (cfg.steps - step) / max(rate, 1.0e-12)
            print(
                f"step {step:7d}/{cfg.steps} | samples={field_accum['nsamples']:5d} | "
                f"elapsed={elapsed/60:7.2f} min | ETA={remaining/60:7.2f} min | "
                f"collisions={collision_totals['executed']:.0f}"
            )

    runtime_s = time.perf_counter() - start_time
    fields = finalize_fields(field_accum, cfg)
    for base in ("rho", "u", "v", "w", "T", "p"):
        fields[f"{base}_smooth"] = smooth2d(fields[f"{base}_raw"], cfg.smoothing_passes)
    fields["vorticity"] = vorticity(fields["u_smooth"], fields["v_smooth"], cfg)
    psi, psi_iterations, psi_change = solve_streamfunction(fields["vorticity"], cfg)
    fields["streamfunction"] = psi
    vortex = find_vortex_center(psi, cfg)
    profiles = finalize_surface_profiles(wall_accum, fields, cfg)

    n_samples = int(field_accum["nsamples"])
    metadata: Dict[str, Any] = asdict(cfg)
    metadata.update(
        {
            "backend_used": backend_name,
            "python_version": platform.python_version(),
            "runtime_s": runtime_s,
            "runtime_h": runtime_s / 3600.0,
            "lambda0_m": cfg.lambda0,
            "n0_per_m3": cfg.n0,
            "dt_s": cfg.dt,
            "particle_weight": cfg.particle_weight,
            "Np": cfg.Np,
            "field_samples": n_samples,
            "dx_over_lambda": cfg.dx / cfg.lambda0,
            "dy_over_lambda": cfg.dy / cfg.lambda0,
            "dt_over_tau_collision_est": cfg.dt / cfg.tau_collision_est,
            "collision_totals": collision_totals,
            "conflict_discard_fraction": collision_totals["conflict_discarded"]
            / max(collision_totals["accepted_before_conflict"], 1.0),
            "majorant_exceed_fraction": collision_totals["majorant_exceed"]
            / max(collision_totals["candidates"], 1.0),
            "final_sigma_g_majorant": sg_majorant,
            "vortex_x_over_L": vortex[0],
            "vortex_y_over_L": vortex[1],
            "vortex_streamfunction": vortex[2],
            "streamfunction_iterations": psi_iterations,
            "streamfunction_relative_change": psi_change,
            "quantitative_data_note": "Use raw time-averaged profiles for metrics; smoothing is for visualization.",
            "source_model": "Week3 GPU VHS-NTC Mohammadzadeh-type cavity",
        }
    )

    np.savez_compressed(case_dir / "fields.npz", **fields)
    with open(case_dir / "metadata.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    x_centers = (np.arange(cfg.nx) + 0.5) / cfg.nx
    y_centers = (np.arange(cfg.ny) + 0.5) / cfg.ny
    X, Y = np.meshgrid(x_centers, y_centers)
    rho_mean = np.mean(fields["rho_raw"])
    dataset_header = [
        "Kn",
        "log10_Kn",
        "Uwall_m_per_s",
        "x_over_L",
        "y_over_L",
        "rho_raw_over_mean",
        "u_raw_over_Uwall",
        "v_raw_over_Uwall",
        "T_raw_over_T0",
        "rho_smooth_over_mean",
        "u_smooth_over_Uwall",
        "v_smooth_over_Uwall",
        "T_smooth_over_T0",
        "vorticity_per_s",
        "streamfunction_m2_per_s",
        "nx",
        "ny",
        "ppc",
        "field_samples",
        "seed",
        "dx_over_lambda",
        "dt_over_tau_est",
    ]
    constants = lambda value: np.full(X.size, value, dtype=float)
    dataset_columns = [
        constants(cfg.Kn),
        constants(math.log10(cfg.Kn)),
        constants(cfg.Uwall),
        X.ravel(),
        Y.ravel(),
        (fields["rho_raw"] / rho_mean).ravel(),
        (fields["u_raw"] / cfg.Uwall).ravel(),
        (fields["v_raw"] / cfg.Uwall).ravel(),
        (fields["T_raw"] / cfg.T0).ravel(),
        (fields["rho_smooth"] / np.mean(fields["rho_smooth"])).ravel(),
        (fields["u_smooth"] / cfg.Uwall).ravel(),
        (fields["v_smooth"] / cfg.Uwall).ravel(),
        (fields["T_smooth"] / cfg.T0).ravel(),
        fields["vorticity"].ravel(),
        fields["streamfunction"].ravel(),
        constants(cfg.nx),
        constants(cfg.ny),
        constants(cfg.ppc),
        constants(n_samples),
        constants(cfg.seed),
        constants(cfg.dx / cfg.lambda0),
        constants(cfg.dt / cfg.tau_collision_est),
    ]
    write_csv(case_dir / "grid_dataset.csv", dataset_header, dataset_columns)

    lid_header = list(profiles.keys())
    write_csv(case_dir / "lid_profiles.csv", lid_header, [profiles[key] for key in lid_header])
    write_csv(
        case_dir / "vortex_center.csv",
        ["x_over_L", "y_over_L", "streamfunction_m2_per_s"],
        [np.asarray([vortex[0]]), np.asarray([vortex[1]]), np.asarray([vortex[2]])],
    )

    history_path = case_dir / "convergence_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "step",
            "elapsed_s",
            "field_samples",
            "centerline_relative_change",
            "lid_slip_relative_change",
            "cumulative_executed_collisions",
            "conflict_discard_fraction",
            "majorant_exceed_fraction",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    save_plots(case_dir, fields, profiles, cfg, vortex)

    print("=" * 78)
    print(f"Completed in {runtime_s/60:.2f} min ({runtime_s/3600:.3f} h)")
    print(f"Field samples: {n_samples}")
    print(f"Primary vortex: x/L={vortex[0]:.4f}, y/L={vortex[1]:.4f}")
    print(
        "Conflict-discard fraction:",
        f"{metadata['conflict_discard_fraction']:.3e}",
        "| majorant-exceed fraction:",
        f"{metadata['majorant_exceed_fraction']:.3e}",
    )
    print("Saved results to:", case_dir)
    print("=" * 78)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--check-gpu", action="store_true", help="Run GPU preflight and exit.")
    parser.add_argument("--debug-cpu", action="store_true", help="Tiny development tests only; not valid for submission.")
    parser.add_argument("--Kn", type=float, default=0.05)
    parser.add_argument("--L", type=float, default=1.0e-6)
    parser.add_argument("--T0", type=float, default=300.0)
    parser.add_argument("--Uwall", type=float, default=100.0)
    parser.add_argument("--m", type=float, default=6.63e-26)
    parser.add_argument("--d-ref", type=float, default=4.17e-10)
    parser.add_argument("--T-ref", type=float, default=300.0)
    parser.add_argument("--omega", type=float, default=0.81)
    parser.add_argument("--depth-factor", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--ny", type=int, default=80)
    parser.add_argument("--ppc", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--sample-start", type=int, default=5_000)
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--dt-factor", type=float, default=0.20)
    parser.add_argument("--gmax-safety", type=float, default=1.35)
    parser.add_argument("--smoothing-passes", type=int, default=3)
    parser.add_argument("--streamfunction-iterations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="week3_gpu_output")
    parser.add_argument("--case-name", default="mohammadzadeh_Kn0p05_long")
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--monitor-every", type=int, default=1_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_gpu:
        check_gpu()
        return
    cfg = DSMCConfig(
        Kn=args.Kn,
        L=args.L,
        T0=args.T0,
        Uwall=args.Uwall,
        m=args.m,
        d_ref=args.d_ref,
        T_ref=args.T_ref,
        omega=args.omega,
        depth_factor=args.depth_factor,
        nx=args.nx,
        ny=args.ny,
        ppc=args.ppc,
        steps=args.steps,
        sample_start=args.sample_start,
        sample_stride=args.sample_stride,
        dt_factor=args.dt_factor,
        gmax_safety=args.gmax_safety,
        smoothing_passes=args.smoothing_passes,
        streamfunction_iterations=args.streamfunction_iterations,
        seed=args.seed,
        output_dir=args.output_dir,
        case_name=args.case_name,
        progress_every=args.progress_every,
        monitor_every=args.monitor_every,
        debug_cpu=args.debug_cpu,
    )
    run(cfg)


if __name__ == "__main__":
    main()
