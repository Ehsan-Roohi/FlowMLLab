#!/usr/bin/env python3
"""Run one rarefied-cavity DSMC realization and export wall pressure.

The physical kernel is the course VHS--NTC solver.  This wrapper selects the
NumPy vector backend when CUDA is unavailable and exports the same observable
used by Mohammadzadeh et al. (2012): p/p0 along A--B--C--D--A.  Corners are
ordered as A=bottom-left, B=top-left, C=top-right, D=bottom-right.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dsmc_cavity_vhs_ntc_gpu_week3 as solver


REFERENCE_P0 = 101_135.0
REFERENCE_MACH = 0.09


def boundary_profile(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-centred samples around A--B--C--D--A."""
    ny, nx = field.shape
    if nx != ny:
        raise ValueError("The validation case requires a square grid.")
    values = np.concatenate(
        (
            field[:, 0],       # A -> B: left wall, bottom to top
            field[-1, :],      # B -> C: moving lid, left to right
            field[::-1, -1],   # C -> D: right wall, top to bottom
            field[0, ::-1],    # D -> A: bottom wall, right to left
        )
    )
    s_over_L = (np.arange(values.size, dtype=float) + 0.5) / nx
    return s_over_L, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--ppc", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--sample-start", type=int, default=10_000)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--dt-factor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # The solver's array operations are backend-neutral.  Its course CLI marks
    # NumPy as debug-only to discourage slow student production runs; here the
    # identical kernel is deliberately executed on CPU because CUDA is absent.
    solver.get_backend = lambda debug_cpu=False: (np, "numpy-cpu-vectorized")

    # Mizzi et al. specify Ma=0.09, T0=273 K, p0=101135 Pa and a
    # hard-sphere argon diameter of 3.42e-10 m.  L follows from the same
    # hard-sphere mean-free-path convention used by the solver so Kn=0.1 and
    # p0 are imposed simultaneously.  Mohammadzadeh et al. report this
    # validation condition as Re=1.5, Kn=0.1.
    temperature = 273.0
    diameter = 3.42e-10
    number_density = REFERENCE_P0 / (solver.KB * temperature)
    length = 1.0 / (np.sqrt(2.0) * np.pi * diameter**2 * 0.1 * number_density)
    gas_constant = solver.KB / 6.63e-26
    lid_speed = REFERENCE_MACH * np.sqrt(2.0 * gas_constant * temperature)

    cfg = solver.DSMCConfig(
        Kn=0.1,
        L=length,
        T0=temperature,
        Uwall=lid_speed,
        m=6.63e-26,
        d_ref=diameter,
        T_ref=temperature,
        omega=0.5,
        nx=args.grid,
        ny=args.grid,
        ppc=args.ppc,
        steps=args.steps,
        sample_start=args.sample_start,
        sample_stride=args.sample_stride,
        dt_factor=args.dt_factor,
        smoothing_passes=1,
        streamfunction_iterations=800,
        seed=args.seed,
        output_dir=str(args.output_dir),
        case_name=args.case_name,
        progress_every=max(500, args.steps // 20),
        monitor_every=max(500, args.steps // 20),
        debug_cpu=False,
    )
    solver.run(cfg)

    case_dir = args.output_dir / args.case_name
    with np.load(case_dir / "fields.npz", allow_pickle=False) as archive:
        p_raw = np.asarray(archive["p_raw"], dtype=float)
        p_filtered = np.asarray(archive["p_smooth"], dtype=float)

    p0 = cfg.n0 * solver.KB * cfg.T0
    s, raw = boundary_profile(p_raw / p0)
    _, filtered = boundary_profile(p_filtered / p0)
    np.savetxt(
        case_dir / "wall_pressure.csv",
        np.column_stack((s, raw, filtered)),
        delimiter=",",
        header="s_over_L,p_raw_over_p0,p_five_point_filtered_over_p0",
        comments="",
    )

    metadata_path = case_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "reference_case": "Mohammadzadeh et al. PRE 85, 056310 (2012), Fig. 3",
            "reference_reported_Re": 1.5,
            "reference_Ma": REFERENCE_MACH,
            "reference_p0_Pa": REFERENCE_P0,
            "wall_path": "A(bottom-left)-B(top-left)-C(top-right)-D(bottom-right)-A",
            "pressure_normalization_Pa": p0,
            "reported_curve": "one-pass five-neighbour filtered adjacent-cell pressure",
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
