#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate exact cubic-FP closure training pairs for the guided cavity project.

Each sampled cell contributes

    X = [rho, T, Ux, Uy, Uz, Pxx, Pxy, Pxz, Pyy, Pyz, Pzz,
         Qx, Qy, Qz, DM2, nu]

and

    Y = [Cxx, Cxy, Cxz, Cyy, Cyz, Czz,
         Gamma_x, Gamma_y, Gamma_z].

The targets are produced by the supplied high-order moment assembly and local
9x9 exact closure solve.  Complete physical conditions are written to one NPZ
file with metadata so that train/validation/test conditions can remain intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fp_project_utils import (
    FEATURE_NAMES,
    TARGET_NAMES,
    FPCavityConfig,
    configure_reference,
    feature_matrix_gpu,
    load_reference_module,
    q_norm_from_features,
    save_metadata_json,
    sigma_norm_from_features,
    target_matrix_gpu,
)


def run_case(module, cfg: FPCavityConfig, sample_cell_stride: int, max_abs_coeff: float,
             min_particles_per_cell: int, progress_every: int):
    cp = module.cp
    metadata = configure_reference(module, cfg)
    cp.random.seed(int(cfg.seed))

    grid, coeffs, linsys = module.initialize_grid_cupy(
        module.NX, module.NY, module.LX, module.LY
    )
    particles = module.initialize_particles_cupy(
        module.NP, module.LX, module.LY, module.THETA_IN, module.W_PARTICLE
    )

    X_blocks = []
    y_blocks = []
    step_blocks = []
    cell_blocks = []
    count_blocks = []

    t0 = time.perf_counter()
    n_sample_events = 0
    n_rows_before_filter = 0
    n_rows_after_filter = 0

    for step in range(1, cfg.steps + 1):
        # Particle transport.
        particles[9][:] = particles[0]
        particles[10][:] = particles[1]
        particles[0][:] = particles[9] + particles[3] * module.DT
        particles[1][:] = particles[10] + particles[4] * module.DT

        # Diffuse wall reflection and exact local closure state.
        module.apply_boundary_cavity_cupy(particles, module.LX, module.LY, module.DT)
        module.sort_and_calc_moments_cupy_FULL(
            particles, grid, module.NC, module.NX, module.NY, module.LX, module.LY
        )
        module.build_linear_systems_cupy(grid, linsys)
        module.solve_linear_systems_cupy(linsys, coeffs)

        if step > cfg.sample_start and step % cfg.sample_stride == 0:
            n_sample_events += 1
            X_gpu = feature_matrix_gpu(module, grid)
            y_gpu = target_matrix_gpu(module, coeffs)
            particle_count_gpu = cp.bincount(
                particles[13], minlength=module.NC
            ).astype(cp.int32)

            X = cp.asnumpy(X_gpu)
            y = cp.asnumpy(y_gpu)
            particle_count = cp.asnumpy(particle_count_gpu)
            cell_id = np.arange(module.NC, dtype=np.int32)

            if sample_cell_stride > 1:
                select = np.arange(0, module.NC, sample_cell_stride, dtype=np.int32)
                X = X[select]
                y = y[select]
                particle_count = particle_count[select]
                cell_id = cell_id[select]

            n_rows_before_filter += len(X)
            finite = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(y), axis=1)
            bounded = np.max(np.abs(y), axis=1) <= float(max_abs_coeff)
            supported = particle_count >= int(min_particles_per_cell)
            keep = finite & bounded & supported

            X = X[keep]
            y = y[keep]
            particle_count = particle_count[keep]
            cell_id = cell_id[keep]
            n_rows_after_filter += len(X)

            X_blocks.append(X.astype(np.float32))
            y_blocks.append(y.astype(np.float32))
            step_blocks.append(np.full(len(X), step, dtype=np.int32))
            cell_blocks.append(cell_id)
            count_blocks.append(particle_count.astype(np.int32))

        module.evolve_velocities_cupy(
            particles, grid, coeffs, module.DT, module.NC
        )

        if progress_every > 0 and (step % progress_every == 0 or step == cfg.steps):
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - t0
            print(
                f"  U_lid={cfg.u_lid:g} m/s, Kn~{cfg.nominal_kn:.4g}: "
                f"step {step}/{cfg.steps}, samples={n_sample_events}, "
                f"rows={n_rows_after_filter}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - t0

    if not X_blocks:
        raise RuntimeError(
            "No training rows were collected. Check sample_start, sample_stride, "
            "particle support, and coefficient filter settings."
        )

    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)
    steps = np.concatenate(step_blocks, axis=0)
    cell_ids = np.concatenate(cell_blocks, axis=0)
    particle_counts = np.concatenate(count_blocks, axis=0)

    qn = q_norm_from_features(X).astype(np.float32)
    sn = sigma_norm_from_features(X).astype(np.float32)

    case_metadata = dict(metadata)
    case_metadata.update(
        elapsed_s=elapsed,
        sample_events=n_sample_events,
        n_rows_before_filter=n_rows_before_filter,
        n_rows_after_filter=n_rows_after_filter,
        retained_fraction=(n_rows_after_filter / max(n_rows_before_filter, 1)),
        max_abs_coeff=float(max_abs_coeff),
        min_particles_per_cell=int(min_particles_per_cell),
        sample_cell_stride=int(sample_cell_stride),
    )

    return {
        "inputs": X,
        "targets": y,
        "step": steps,
        "cell_id": cell_ids,
        "particle_count": particle_counts,
        "q_norm": qn,
        "sigma_norm": sn,
        "case_metadata": case_metadata,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="fp_cavity_reference.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", choices=["train", "validation", "blind", "other"], default="train")
    ap.add_argument("--lid-speeds", nargs="+", type=float, required=True)
    ap.add_argument("--density-scales", nargs="+", type=float, default=[1.0])
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--ppc", type=int, default=80)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--sample-start", type=int, default=600)
    ap.add_argument("--sample-stride", type=int, default=20)
    ap.add_argument("--sample-cell-stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--max-abs-coeff", type=float, default=1.0e8)
    ap.add_argument("--min-particles-per-cell", type=int, default=4)
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    module = load_reference_module(args.reference)

    if len(args.density_scales) not in (1, len(args.lid_speeds)):
        raise ValueError(
            "Provide one density scale for all lid speeds, or one scale per lid speed."
        )
    density_scales = (
        list(args.density_scales) * len(args.lid_speeds)
        if len(args.density_scales) == 1
        else list(args.density_scales)
    )

    all_inputs = []
    all_targets = []
    all_steps = []
    all_cells = []
    all_counts = []
    all_qn = []
    all_sn = []
    all_u = []
    all_kn = []
    all_case_index = []
    case_table = []

    print("Generating exact cubic-FP closure data", flush=True)
    print(f"  role={args.role}; output={out_path}", flush=True)

    for i, (u_lid, density_scale) in enumerate(zip(args.lid_speeds, density_scales)):
        cfg = FPCavityConfig(
            u_lid=float(u_lid),
            density_scale=float(density_scale),
            nx=args.nx,
            ny=args.ny,
            particles_per_cell=args.ppc,
            steps=args.steps,
            sample_start=args.sample_start,
            sample_stride=args.sample_stride,
            seed=args.seed + i,
        )
        print(
            f"Case {i}: U_lid={cfg.u_lid:g} m/s, density_scale={cfg.density_scale:g}, "
            f"nominal Kn~{cfg.nominal_kn:.4g}, particles={cfg.n_particles}",
            flush=True,
        )
        result = run_case(
            module,
            cfg,
            sample_cell_stride=args.sample_cell_stride,
            max_abs_coeff=args.max_abs_coeff,
            min_particles_per_cell=args.min_particles_per_cell,
            progress_every=args.progress_every,
        )
        n = len(result["inputs"])
        all_inputs.append(result["inputs"])
        all_targets.append(result["targets"])
        all_steps.append(result["step"])
        all_cells.append(result["cell_id"])
        all_counts.append(result["particle_count"])
        all_qn.append(result["q_norm"])
        all_sn.append(result["sigma_norm"])
        all_u.append(np.full(n, cfg.u_lid, dtype=np.float32))
        all_kn.append(np.full(n, cfg.nominal_kn, dtype=np.float32))
        all_case_index.append(np.full(n, i, dtype=np.int32))
        case_table.append(result["case_metadata"])

    inputs = np.concatenate(all_inputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metadata = {
        "dataset_role": args.role,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "case_table": case_table,
        "n_rows": int(len(inputs)),
        "generator": "generate_fp_cavity_dataset.py",
        "important_scope": (
            "Reduced educational cubic-FP data. Final conference/manuscript claims require "
            "production particle counts, longer transients, and independent validation."
        ),
    }

    np.savez_compressed(
        out_path,
        inputs=inputs,
        targets=targets,
        step=np.concatenate(all_steps),
        cell_id=np.concatenate(all_cells),
        particle_count=np.concatenate(all_counts),
        q_norm=np.concatenate(all_qn),
        sigma_norm=np.concatenate(all_sn),
        U_lid=np.concatenate(all_u),
        Kn=np.concatenate(all_kn),
        case_index=np.concatenate(all_case_index),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    save_metadata_json(out_path.with_suffix(".metadata.json"), metadata)

    print("Saved:", out_path)
    print("Shape:", inputs.shape, targets.shape)
    print("Cases:", len(case_table))
    print("q_norm mean/p95:", float(np.mean(np.concatenate(all_qn))), float(np.percentile(np.concatenate(all_qn), 95)))


if __name__ == "__main__":
    main()
