"""Teaching utilities for the Roohi--Mahdavi rarefied-flow case studies.

The micro-nozzle helpers operate on a compact, attributed derivative of the
authors' public DSMC snapshots.  The micro-step field generator is deliberately
manufactured: the article's DSMC fields and trained checkpoint are not public,
so it is used only to teach case-wise splitting and zonal objectives.
"""

from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d


NOZZLE_PRESSURES_KPA = np.array(
    [15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33],
    dtype=float,
)
NOZZLE_HELD_OUT_KPA = np.array([16, 25, 30], dtype=float)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return a dimensionless relative L2 error over finite paired values."""
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(mask):
        raise ValueError("relative_l2 received no finite paired values")
    denominator = np.linalg.norm(reference[mask])
    if denominator <= np.finfo(float).eps:
        raise ValueError("relative_l2 reference norm is zero")
    return float(np.linalg.norm(prediction[mask] - reference[mask]) / denominator)


def manufactured_step_velocity(
    x: np.ndarray,
    y: np.ndarray,
    height_ratio: float,
    *,
    step_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth pedagogical backward-facing-step field.

    This analytic field is not DSMC and is not article evidence.  It creates a
    height-dependent recirculation pocket so students can inspect how a global
    objective underweights a small but important ``u < 0`` region.
    """
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    h = float(height_ratio)
    if not 0.15 <= h <= 0.80:
        raise ValueError("height_ratio must lie in [0.15, 0.80]")

    solid = (x_array < step_x) & (y_array < h)
    lower_wall = np.where(x_array < step_x, h, 0.0)
    eta = np.clip((y_array - lower_wall) / np.maximum(1.0 - lower_wall, 1.0e-12), 0, 1)
    base = 6.0 * eta * (1.0 - eta)

    recirculation_length = 0.65 + 2.1 * h
    xc = step_x + 0.43 * recirculation_length
    yc = 0.30 * h + 0.055
    sx = 0.34 * recirculation_length
    sy = 0.08 + 0.22 * h
    gaussian = np.exp(-((x_array - xc) / sx) ** 2 - ((y_array - yc) / sy) ** 2)
    u = base - (1.25 + 1.25 * h) * gaussian
    v = 0.23 * (x_array - xc) / sx * gaussian
    recovery = 1.0 - 0.08 * np.exp(-np.maximum(x_array - step_x, 0.0) / 1.8)
    u *= recovery
    u[solid] = 0.0
    v[solid] = 0.0
    return u, v, solid


def zonal_velocity_metrics(
    u_reference: np.ndarray,
    v_reference: np.ndarray,
    u_prediction: np.ndarray,
    v_prediction: np.ndarray,
    *,
    alpha: float = 0.7,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate the full field and the true-flow recirculation zone separately.

    The zonal objective is ``alpha * MSE_vortex + (1-alpha) * MSE_main``.
    Each regional MSE is normalized by its own point count, matching the
    scientific intent of the article's physics-guided zonal loss.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    arrays = np.broadcast_arrays(
        np.asarray(u_reference, dtype=float),
        np.asarray(v_reference, dtype=float),
        np.asarray(u_prediction, dtype=float),
        np.asarray(v_prediction, dtype=float),
    )
    u_ref, v_ref, u_pred, v_pred = arrays
    finite = np.isfinite(u_ref) & np.isfinite(v_ref) & np.isfinite(u_pred) & np.isfinite(v_pred)
    if valid_mask is not None:
        finite &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), finite.shape)
    vortex = finite & (u_ref < 0.0)
    main = finite & ~vortex
    if not np.any(vortex) or not np.any(main):
        raise ValueError("both vortex and main-flow points are required")

    squared = (u_pred - u_ref) ** 2 + (v_pred - v_ref) ** 2
    full_mse = float(np.mean(squared[finite]))
    vortex_mse = float(np.mean(squared[vortex]))
    main_mse = float(np.mean(squared[main]))
    return {
        "full_mse": full_mse,
        "vortex_mse": vortex_mse,
        "main_mse": main_mse,
        "zonal_loss": float(alpha * vortex_mse + (1.0 - alpha) * main_mse),
        "full_relative_l2": relative_l2(
            np.stack((u_ref[finite], v_ref[finite])),
            np.stack((u_pred[finite], v_pred[finite])),
        ),
        "vortex_relative_l2": relative_l2(
            np.stack((u_ref[vortex], v_ref[vortex])),
            np.stack((u_pred[vortex], v_pred[vortex])),
        ),
        "vortex_points": int(np.count_nonzero(vortex)),
        "main_points": int(np.count_nonzero(main)),
    }


def detect_density_shock(
    x_m: np.ndarray,
    density: np.ndarray,
    *,
    smooth_sigma: float = 0.7,
) -> dict[str, float]:
    """Detect the strongest interior centerline-density compression and width."""
    x = np.asarray(x_m, dtype=float)
    rho = np.asarray(density, dtype=float)
    if x.ndim != 1 or rho.shape != x.shape or len(x) < 15:
        raise ValueError("x_m and density must be paired 1-D profiles")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_m must be strictly increasing")
    smoothed = gaussian_filter1d(rho, smooth_sigma, mode="nearest")
    gradient = np.gradient(smoothed, x, edge_order=2)
    span = float(np.ptp(x))
    candidates = (x >= x[0] + 0.05 * span) & (x <= x[0] + 0.95 * span)
    index = np.flatnonzero(candidates)[np.argmax(np.abs(gradient[candidates]))]
    xs = float(x[index])
    dx = float(np.median(np.diff(x)))
    left = (x >= xs - 12 * dx) & (x <= xs - 3 * dx)
    right = (x >= xs + 3 * dx) & (x <= xs + 12 * dx)
    rho_left = float(np.mean(rho[left])) if np.any(left) else float(rho[0])
    rho_right = float(np.mean(rho[right])) if np.any(right) else float(rho[-1])
    jump = abs(rho_left - rho_right)
    peak = float(abs(gradient[index]))
    width = jump / peak if peak > 0.0 else 5.0 * dx
    if not np.isfinite(width) or width <= 0.0:
        width = 5.0 * dx
    return {
        "shock_x_m": xs,
        "delta_jump_m": float(width),
        "density_left": rho_left,
        "density_right": rho_right,
        "peak_abs_gradient": peak,
    }


def normalize_density_jump(
    x_m: np.ndarray,
    density: np.ndarray,
    shock_x_m: float,
) -> np.ndarray:
    """Normalize a density profile by local pre/post-shock plateaus."""
    x = np.asarray(x_m, dtype=float)
    rho = np.asarray(density, dtype=float)
    dx = float(np.median(np.diff(x)))
    left = (x >= shock_x_m - 12 * dx) & (x <= shock_x_m - 3 * dx)
    right = (x >= shock_x_m + 3 * dx) & (x <= shock_x_m + 12 * dx)
    rho_left = float(np.mean(rho[left])) if np.any(left) else float(rho[0])
    rho_right = float(np.mean(rho[right])) if np.any(right) else float(rho[-1])
    denominator = rho_left - rho_right
    if abs(denominator) <= np.finfo(float).eps:
        denominator = float(np.ptp(rho)) or 1.0
    return (rho - rho_right) / denominator


def density_snapshot_matrix(
    x_m: np.ndarray,
    density: np.ndarray,
    shock_x_m: np.ndarray,
    delta_jump_m: np.ndarray,
    *,
    coordinate: str,
    points: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate jump-normalized profiles on physical or shock-centered axes."""
    x = np.asarray(x_m, dtype=float)
    fields = np.asarray(density, dtype=float)
    xs = np.asarray(shock_x_m, dtype=float)
    delta = np.asarray(delta_jump_m, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != len(x):
        raise ValueError("density must have shape (cases, len(x_m))")
    if xs.shape != (fields.shape[0],) or delta.shape != xs.shape:
        raise ValueError("shock arrays must contain one value per case")
    if coordinate == "physical":
        grid = np.linspace(0.0, 210.0, points)
    elif coordinate == "shock_centered":
        grid = np.linspace(-8.0, 8.0, points)
    else:
        raise ValueError("coordinate must be 'physical' or 'shock_centered'")

    rows = []
    for profile, location, width in zip(fields, xs, delta, strict=True):
        normalized = normalize_density_jump(x, profile, float(location))
        source_coordinate = x * 1.0e6 if coordinate == "physical" else (x - location) / width
        rows.append(
            np.interp(
                grid,
                source_coordinate,
                normalized,
                left=float(normalized[0]),
                right=float(normalized[-1]),
            )
        )
    return grid, np.vstack(rows)


def pod_spectrum(snapshot_matrix: np.ndarray) -> dict[str, np.ndarray | int | float]:
    """Return a centered snapshot POD and its cumulative energy diagnostics."""
    snapshots = np.asarray(snapshot_matrix, dtype=float)
    if snapshots.ndim != 2 or snapshots.shape[0] < 2:
        raise ValueError("snapshot_matrix must contain at least two row snapshots")
    mean = np.mean(snapshots, axis=0)
    _, singular_values, modes = np.linalg.svd(snapshots - mean, full_matrices=False)
    energy = singular_values**2
    fractions = energy / np.sum(energy)
    cumulative = np.cumsum(fractions)
    n99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {
        "mean": mean,
        "singular_values": singular_values,
        "modes": modes,
        "energy_fraction": fractions,
        "cumulative_energy": cumulative,
        "first_mode_percent": float(100.0 * fractions[0]),
        "first_two_percent": float(100.0 * cumulative[1]),
        "first_three_percent": float(100.0 * cumulative[2]),
        "n99": n99,
    }


def load_nozzle_centerlines(root: str | Path) -> dict[str, np.ndarray]:
    """Load the compact attributed 15-case DSMC centerline archive."""
    root_path = Path(root).resolve()
    path = root_path / "results" / "mahdavi_deeponet" / "nozzle_centerline_15cases.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_week9_evidence(root: str | Path) -> dict[str, Any]:
    """Validate the Week-9 article evidence, derived DSMC data, and POD audit."""
    root_path = Path(root).resolve()
    result_dir = root_path / "results" / "mahdavi_deeponet"
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    archive_path = result_dir / "nozzle_centerline_15cases.npz"
    if _sha256(archive_path) != provenance["derived_files"][archive_path.name]:
        raise ValueError("compact nozzle archive SHA-256 mismatch")
    data = load_nozzle_centerlines(root_path)
    if not np.array_equal(data["pressure_kpa"], NOZZLE_PRESSURES_KPA):
        raise ValueError("unexpected nozzle pressure cases")
    required = (
        "density", "u_ms", "v_ms", "temperature_k", "mach",
        "pressure_tecplot", "knudsen", "shock_x_m", "delta_jump_m",
    )
    for name in required:
        if name not in data or not np.isfinite(data[name]).all():
            raise ValueError(f"invalid compact nozzle field: {name}")
    if data["density"].shape != (15, 101) or data["x_m"].shape != (101,):
        raise ValueError("unexpected compact nozzle data shape")

    pod_rows: dict[str, dict[str, float | int]] = {}
    for coordinate in ("physical", "shock_centered"):
        _, matrix = density_snapshot_matrix(
            data["x_m"], data["density"], data["shock_x_m"],
            data["delta_jump_m"], coordinate=coordinate,
        )
        spectrum = pod_spectrum(matrix)
        pod_rows[coordinate] = {
            "E1_percent": float(spectrum["first_mode_percent"]),
            "E12_percent": float(spectrum["first_two_percent"]),
            "E123_percent": float(spectrum["first_three_percent"]),
            "N99": int(spectrum["n99"]),
        }
    with (result_dir / "nozzle_pod_reference.csv").open(encoding="utf-8") as stream:
        reference = {row["coordinate"]: row for row in csv.DictReader(stream)}
    for coordinate, observed in pod_rows.items():
        expected = reference[coordinate]
        for key in ("E1_percent", "E12_percent", "E123_percent"):
            if abs(float(observed[key]) - float(expected[key])) > 0.03:
                raise ValueError(f"{coordinate} POD audit failed for {key}")
        if int(observed["N99"]) != int(expected["N99"]):
            raise ValueError(f"{coordinate} POD audit failed for N99")

    boundary = provenance["claim_boundary"]
    if "manufactured" not in boundary["microstep_teaching_demo"].lower():
        raise ValueError("micro-step claim boundary is missing")
    if "not" not in boundary["micro_nozzle_teaching_model"].lower():
        raise ValueError("micro-nozzle model claim boundary is missing")
    return {
        "status": "pass",
        "nozzle_cases": int(len(data["pressure_kpa"])),
        "held_out_pressures_kpa": NOZZLE_HELD_OUT_KPA.astype(int).tolist(),
        "physical_density_pod": pod_rows["physical"],
        "shock_centered_density_pod": pod_rows["shock_centered"],
        "archive_sha256": provenance["derived_files"][archive_path.name],
    }
