#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities for Track 6: Fokker-Planck cavity closure surrogate.

The project intentionally keeps the research solver in a separate module and
changes only a small, documented set of global configuration values before a
run.  This lets students focus on the machine-learning experiment rather than
rewriting the particle solver.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

FEATURE_NAMES = [
    "rho", "T", "Ux", "Uy", "Uz",
    "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz",
    "Qx", "Qy", "Qz", "DM2", "nu",
]

TARGET_NAMES = [
    "Cxx", "Cxy", "Cxz", "Cyy", "Cyz", "Czz",
    "Gamma_x", "Gamma_y", "Gamma_z",
]

BASE_NOMINAL_KN = 0.15


@dataclass(frozen=True)
class FPCavityConfig:
    """Complete physical/numerical definition of one educational cavity case."""

    u_lid: float = 400.0
    density_scale: float = 1.0
    nx: int = 20
    ny: int = 20
    particles_per_cell: int = 80
    steps: int = 1200
    sample_start: int = 600
    sample_stride: int = 20
    seed: int = 101
    cfl_fraction: float = 0.20
    wall_temperature: float = 273.15

    @property
    def nominal_kn(self) -> float:
        # For the fixed gas/geometry used by the reference code, mean free path
        # is inversely proportional to density.
        return BASE_NOMINAL_KN / float(self.density_scale)

    @property
    def n_cells(self) -> int:
        return int(self.nx * self.ny)

    @property
    def n_particles(self) -> int:
        return int(self.n_cells * self.particles_per_cell)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.update(
            nominal_kn=self.nominal_kn,
            n_cells=self.n_cells,
            n_particles=self.n_particles,
        )
        return out


def load_reference_module(path: str | Path = "fp_cavity_reference.py"):
    """Import the reference solver from a known path without modifying sys.path."""

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference solver not found: {path}")
    spec = importlib.util.spec_from_file_location("fp_cavity_reference_runtime", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            raise RuntimeError(
                "CuPy is not available. Track 6 requires an NVIDIA GPU runtime. "
                "In Google Colab choose Runtime -> Change runtime type -> GPU, "
                "then run the CuPy installation cell before importing the reference solver."
            ) from exc
        raise
    except Exception as exc:
        # CuPy may be installed but unusable when the runtime has no CUDA device.
        message = str(exc).lower()
        if "cuda" in message or "driver" in message or "device" in message:
            raise RuntimeError(
                "The Fokker-Planck reference solver could not initialize CUDA. "
                "Track 6 must be run on an NVIDIA GPU runtime; in Colab choose "
                "Runtime -> Change runtime type -> GPU and restart the runtime."
            ) from exc
        raise
    return module


def configure_reference(module, cfg: FPCavityConfig) -> Dict[str, Any]:
    """Apply one bounded case configuration to the imported research solver.

    Only the case-level physical/numerical controls are changed.  The FP model,
    moment definitions, exact closure system, wall model, and velocity update
    remain the supplied instructor implementation.
    """

    if cfg.nx < 2 or cfg.ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if cfg.particles_per_cell < 4:
        raise ValueError("Use at least 4 particles per cell for the teaching run")
    if cfg.sample_start >= cfg.steps:
        raise ValueError("sample_start must be smaller than steps")
    if cfg.sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    if cfg.density_scale <= 0:
        raise ValueError("density_scale must be positive")

    module.UW_LID = float(cfg.u_lid)
    module.TW_ALL = float(cfg.wall_temperature)
    module.T_IN_BASE = float(cfg.wall_temperature)
    module.THETA_IN = np.sqrt(module.K_B * module.T_IN_BASE / module.MASS_AR)
    module.THETA_W = np.sqrt(module.K_B * module.TW_ALL / module.MASS_AR)

    module.RHO_IN = float(module.RHO_IN_BASE * cfg.density_scale)
    module.NX = int(cfg.nx)
    module.NY = int(cfg.ny)
    module.NC = int(cfg.n_cells)
    module.PARTICLES_PER_CELL_TARGET = int(cfg.particles_per_cell)
    module.NP = int(cfg.n_particles)

    module.DX = module.LX / float(module.NX)
    module.DY = module.LY / float(module.NY)
    module.MIN_DIM = min(module.DX, module.DY)
    module.DT = float(
        cfg.cfl_fraction * module.MIN_DIM / max(module.UW_LID, module.THETA_IN)
    )
    module.NTSS = int(cfg.sample_start)
    module.N_STEPS_PER_RUN = int(cfg.steps)
    module.W_PARTICLE = (
        module.LX * module.LY * module.RHO_IN / float(module.NP)
    )

    # Wrapper scripts control diagnostics explicitly.  Setting a very large
    # interval is safer than zero for legacy code containing modulo operations.
    safe_skip = int(cfg.steps) + 10**9
    if hasattr(module, "ENTROPY_EVERY"):
        module.ENTROPY_EVERY = safe_skip
    if hasattr(module, "HIGH_MOMENTS_EVERY"):
        module.HIGH_MOMENTS_EVERY = safe_skip

    metadata = cfg.to_dict()
    metadata.update(
        dt=module.DT,
        dx=module.DX,
        dy=module.DY,
        rho_initial=module.RHO_IN,
        particle_weight=module.W_PARTICLE,
        reference_solver_version=getattr(module, "FP_REFERENCE_VERSION", "unknown"),
        feature_names=FEATURE_NAMES,
        target_names=TARGET_NAMES,
    )
    return metadata


def feature_matrix_gpu(module, grid):
    """Return the 16 low-order inputs used by the neural closure."""

    cp = module.cp
    return cp.stack(
        [
            grid["rho"],
            grid["T"],
            grid["U"][:, 0], grid["U"][:, 1], grid["U"][:, 2],
            grid["PIJ"][:, 0], grid["PIJ"][:, 1], grid["PIJ"][:, 2],
            grid["PIJ"][:, 3], grid["PIJ"][:, 4], grid["PIJ"][:, 5],
            grid["Q"][:, 0], grid["Q"][:, 1], grid["Q"][:, 2],
            grid["DM2"],
            grid["nu"],
        ],
        axis=1,
    )


def target_matrix_gpu(module, coeffs):
    """Return the six C_ij and three Gamma_i exact closure coefficients."""

    return module.cp.concatenate([coeffs["A"], coeffs["B"]], axis=1)


def q_norm_from_features(X: np.ndarray) -> np.ndarray:
    """Dimensionless heat-flux magnitude used to identify difficult states."""

    X = np.asarray(X, dtype=np.float64)
    dm2 = np.maximum(X[:, 14], 1.0e-300)
    q = X[:, 11:14]
    return np.linalg.norm(q, axis=1) / np.maximum(dm2 ** 1.5, 1.0e-300)


def sigma_norm_from_features(X: np.ndarray) -> np.ndarray:
    """Dimensionless stress-deviator magnitude from the 16 feature vector."""

    X = np.asarray(X, dtype=np.float64)
    dm2 = np.maximum(X[:, 14], 1.0e-300)
    theta = dm2 / 3.0
    P = X[:, 5:11]
    sxx = P[:, 0] - theta
    sxy = P[:, 1]
    sxz = P[:, 2]
    syy = P[:, 3] - theta
    syz = P[:, 4]
    szz = P[:, 5] - theta
    norm2 = sxx**2 + syy**2 + szz**2 + 2.0 * (sxy**2 + sxz**2 + syz**2)
    return np.sqrt(np.maximum(norm2, 0.0)) / dm2


def save_metadata_json(path: str | Path, metadata: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def parse_float_list(values: Iterable[str | float]) -> list[float]:
    return [float(v) for v in values]
