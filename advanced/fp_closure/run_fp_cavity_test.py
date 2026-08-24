#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a complete held-out cavity condition with exact and neural FP closure.

The script restarts the particle simulation with the same seed for the two
modes, time-averages macroscopic/high-order fields over the same window, and
writes separate NPZ files plus a runtime summary.  The neural weights must have
been exported by train_fp_closure.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from fp_project_utils import FPCavityConfig, configure_reference, load_reference_module

GRID_KEYS = ["U", "T", "rho", "PIJ", "Q", "DM2", "nu"]
HIGH_KEYS = [
    "M3sym", "m3_stf", "Rij_raw", "Rij_dev",
    "Delta4", "Delta4_norm", "DM6", "DM6_norm",
    "sigma_norm", "q_norm", "m3_norm", "Rij_norm",
]


def load_model_gpu(module, path):
    d = np.load(path)
    required = ["X_mean", "X_scale", "y_mean", "y_scale"] + [f"W{i}" for i in range(1, 6)] + [f"b{i}" for i in range(1, 6)]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{path}: missing model arrays {missing}")
    return {k: module.cp.asarray(d[k]) for k in d.files}


def initialize_average(module):
    avg = {}
    for key in GRID_KEYS + HIGH_KEYS:
        if key in ("U", "PIJ", "Q", "M3sym", "m3_stf", "Rij_raw", "Rij_dev"):
            shape = {
                "U": (module.NC, 3),
                "PIJ": (module.NC, 6),
                "Q": (module.NC, 3),
                "M3sym": (module.NC, 10),
                "m3_stf": (module.NC, 10),
                "Rij_raw": (module.NC, 6),
                "Rij_dev": (module.NC, 6),
            }[key]
            avg[key] = np.zeros(shape, dtype=np.float64)
        else:
            avg[key] = np.zeros(module.NC, dtype=np.float64)
    avg["A"] = np.zeros((module.NC, 6), dtype=np.float64)
    avg["B"] = np.zeros((module.NC, 3), dtype=np.float64)
    return avg


def update_average(avg, sample, count):
    alpha = 1.0 / float(count)
    for key, value in sample.items():
        avg[key] += alpha * (value - avg[key])


def snapshot_to_cpu(module, grid, coeffs):
    cp = module.cp
    out = {}
    for key in GRID_KEYS + HIGH_KEYS:
        out[key] = cp.asnumpy(grid[key])
    out["A"] = cp.asnumpy(coeffs["A"])
    out["B"] = cp.asnumpy(coeffs["B"])
    return out


def run_mode(module, cfg, mode, model_gpu, progress_every):
    if mode not in {"physics", "ml"}:
        raise ValueError(mode)
    cp = module.cp
    configure_reference(module, cfg)
    cp.random.seed(int(cfg.seed))

    grid, coeffs, linsys = module.initialize_grid_cupy(
        module.NX, module.NY, module.LX, module.LY
    )
    particles = module.initialize_particles_cupy(
        module.NP, module.LX, module.LY, module.THETA_IN, module.W_PARTICLE
    )

    avg = initialize_average(module)
    n_samples = 0
    t0 = time.perf_counter()

    for step in range(1, cfg.steps + 1):
        particles[9][:] = particles[0]
        particles[10][:] = particles[1]
        particles[0][:] = particles[9] + particles[3] * module.DT
        particles[1][:] = particles[10] + particles[4] * module.DT
        module.apply_boundary_cavity_cupy(particles, module.LX, module.LY, module.DT)

        if mode == "physics":
            module.sort_and_calc_moments_cupy_FULL(
                particles, grid, module.NC, module.NX, module.NY, module.LX, module.LY
            )
            module.build_linear_systems_cupy(grid, linsys)
            module.solve_linear_systems_cupy(linsys, coeffs)
        else:
            module.sort_and_calc_moments_cupy_LITE(
                particles, grid, module.NC, module.NX, module.NY, module.LX, module.LY
            )
            module.predict_coeffs_cupy_native(grid, coeffs, model_gpu)

        if step > cfg.sample_start and step % cfg.sample_stride == 0:
            module.calc_high_moments_R13_R26_cupy(particles, grid, module.NC)
            n_samples += 1
            update_average(avg, snapshot_to_cpu(module, grid, coeffs), n_samples)

        module.evolve_velocities_cupy(
            particles, grid, coeffs, module.DT, module.NC
        )

        if progress_every > 0 and (step % progress_every == 0 or step == cfg.steps):
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"  {mode:7s} step {step}/{cfg.steps}; samples={n_samples}; "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0
    if n_samples == 0:
        raise RuntimeError("No field samples were accumulated")
    return avg, elapsed, n_samples


def save_field_file(path, avg, module, cfg, mode, elapsed, n_samples, model_path):
    metadata = cfg.to_dict()
    metadata.update(
        mode=mode,
        runtime_s=elapsed,
        field_samples=n_samples,
        dt=module.DT,
        dx=module.DX,
        dy=module.DY,
        model_path=model_path if mode == "ml" else None,
        scope=(
            "Reduced educational closed-loop cavity run. Production paper-level claims require "
            "larger particle counts, longer averaging, and independent convergence checks."
        ),
    )
    payload = dict(avg)
    payload.update(
        x_coords=np.linspace(module.DX / 2.0, module.LX - module.DX / 2.0, module.NX),
        y_coords=np.linspace(module.DY / 2.0, module.LY - module.DY / 2.0, module.NY),
        nx=np.asarray(module.NX),
        ny=np.asarray(module.NY),
        U_lid=np.asarray(cfg.u_lid),
        Kn=np.asarray(cfg.nominal_kn),
        T_wall=np.asarray(cfg.wall_temperature),
        runtime_s=np.asarray(elapsed),
        field_samples=np.asarray(n_samples),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    np.savez_compressed(path, **payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="fp_cavity_reference.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="fp_blind_cavity")
    ap.add_argument("--u-lid", type=float, default=800.0)
    ap.add_argument("--density-scale", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=24)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--ppc", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--sample-start", type=int, default=1000)
    ap.add_argument("--sample-stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--progress-every", type=int, default=300)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    module = load_reference_module(args.reference)
    cfg = FPCavityConfig(
        u_lid=args.u_lid,
        density_scale=args.density_scale,
        nx=args.nx,
        ny=args.ny,
        particles_per_cell=args.ppc,
        steps=args.steps,
        sample_start=args.sample_start,
        sample_stride=args.sample_stride,
        seed=args.seed,
    )
    configure_reference(module, cfg)
    model_gpu = load_model_gpu(module, args.model)

    print("Held-out closed-loop cubic-FP cavity test")
    print(json.dumps(cfg.to_dict(), indent=2))

    phys, t_phys, n_phys = run_mode(
        module, cfg, "physics", None, args.progress_every
    )
    ml, t_ml, n_ml = run_mode(
        module, cfg, "ml", model_gpu, args.progress_every
    )

    physics_path = outdir / "fp_cavity_PHYSICS.npz"
    ml_path = outdir / "fp_cavity_ML.npz"
    save_field_file(physics_path, phys, module, cfg, "physics", t_phys, n_phys, None)
    save_field_file(ml_path, ml, module, cfg, "ml", t_ml, n_ml, args.model)

    speedup = t_phys / t_ml if t_ml > 0 else np.nan
    with (outdir / "runtime_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "runtime_s", "field_samples"])
        writer.writerow(["physics", t_phys, n_phys])
        writer.writerow(["ml", t_ml, n_ml])
        writer.writerow(["speedup", speedup, ""])

    print("Saved:", physics_path)
    print("Saved:", ml_path)
    print(f"Runtime physics={t_phys:.3f}s, ML={t_ml:.3f}s, speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()
