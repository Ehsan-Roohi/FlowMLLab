"""Executable scientific-software contracts for the Week 1.1 laboratory.

The routines in this module separate four questions that are often conflated
when code is proposed by a person or a coding agent:

1. Does the function execute?
2. Does its discretization converge on a problem with a known answer?
3. Does it preserve the physical contract of a qualified CFD field?
4. Can the exact data and decision thresholds be reconstructed later?

The reference diagnostic uses the FlowMLLab convention ``field[y, x]`` and
defines two-dimensional vorticity as ``dv/dx - du/dy``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .core import EXPECTED_DATA_SHA256


@dataclass(frozen=True)
class DifferentialDiagnostics:
    """Finite-difference divergence and scalar vorticity on a Cartesian grid."""

    divergence: np.ndarray
    vorticity: np.ndarray


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_grid(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional coordinate arrays")
    if u.shape != (y.size, x.size) or v.shape != u.shape:
        raise ValueError("u and v must have shape (len(y), len(x))")
    if min(x.size, y.size) < 5:
        raise ValueError("at least five points per direction are required")
    if not all(np.isfinite(array).all() for array in (x, y, u, v)):
        raise ValueError("coordinates and velocity fields must be finite")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.diff(y) > 0.0):
        raise ValueError("coordinates must be strictly increasing")


def differential_diagnostics(
    x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray
) -> DifferentialDiagnostics:
    """Compute second-order divergence and vorticity with explicit axes.

    Arrays follow the CFD convention ``u[j, i] = u(y[j], x[i])``. Passing the
    coordinate vectors directly to ``numpy.gradient`` also supports a
    stretched grid, while ``edge_order=2`` avoids a silent first-order boundary
    stencil.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    _validated_grid(x, y, u, v)
    du_dx = np.gradient(u, x, axis=1, edge_order=2)
    du_dy = np.gradient(u, y, axis=0, edge_order=2)
    dv_dx = np.gradient(v, x, axis=1, edge_order=2)
    dv_dy = np.gradient(v, y, axis=0, edge_order=2)
    return DifferentialDiagnostics(divergence=du_dx + dv_dy, vorticity=dv_dx - du_dy)


def manufactured_incompressible_field(
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return a smooth no-slip field with exact zero divergence and vorticity.

    The streamfunction is

    ``psi = sin(pi*x)^2 sin(pi*y)^2``,

    with ``u = dpsi/dy`` and ``v = -dpsi/dx``. Both velocity components vanish
    on every wall, making the field useful for testing derivatives without a
    lid-corner singularity.
    """

    if points < 9:
        raise ValueError("the manufactured verification requires at least nine points")
    x = np.linspace(0.0, 1.0, int(points))
    y = np.linspace(0.0, 1.0, int(points))
    xx, yy = np.meshgrid(x, y)
    u = np.pi * np.sin(np.pi * xx) ** 2 * np.sin(2.0 * np.pi * yy)
    v = -np.pi * np.sin(2.0 * np.pi * xx) * np.sin(np.pi * yy) ** 2
    omega = -2.0 * np.pi**2 * (
        np.cos(2.0 * np.pi * xx) * np.sin(np.pi * yy) ** 2
        + np.sin(np.pi * xx) ** 2 * np.cos(2.0 * np.pi * yy)
    )
    return x, y, u, v, omega


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("reference and candidate must have the same shape")
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0.0:
        raise ValueError("relative L2 is undefined for a zero reference")
    return float(np.linalg.norm((candidate - reference).ravel()) / denominator)


def verification_sweep(grids: Iterable[int] = (17, 33, 65, 129)) -> dict[str, object]:
    """Verify the diagnostic against the analytic manufactured field."""

    rows: list[dict[str, float | int]] = []
    for points in grids:
        x, y, u, v, omega_exact = manufactured_incompressible_field(int(points))
        diagnostic = differential_diagnostics(x, y, u, v)
        interior = np.s_[2:-2, 2:-2]
        rows.append(
            {
                "points": int(points),
                "h": float(max(np.diff(x).max(), np.diff(y).max())),
                "vorticity_relative_l2": relative_l2(
                    omega_exact[interior], diagnostic.vorticity[interior]
                ),
                "divergence_rms": float(
                    np.sqrt(np.mean(diagnostic.divergence[interior] ** 2))
                ),
            }
        )
    if len(rows) < 3:
        raise ValueError("at least three grids are required to estimate observed order")
    h = np.asarray([row["h"] for row in rows], dtype=float)
    error = np.asarray([row["vorticity_relative_l2"] for row in rows], dtype=float)
    observed_order = float(np.polyfit(np.log(h), np.log(error), 1)[0])
    return {"rows": rows, "observed_order": observed_order}


def load_cavity_case(
    root: str | Path, reynolds: float = 100.0
) -> tuple[dict[str, np.ndarray | float | str], Path]:
    """Load one accepted complete field from the fixed Week-1 cavity archive."""

    archive_path = Path(root).resolve() / "data" / "cavity_data.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        matches = np.flatnonzero(np.isclose(archive["Re"], float(reynolds)))
        if matches.size != 1:
            raise ValueError(f"expected one Re={reynolds:g} case, found {matches.size}")
        index = int(matches[0])
        if not bool(archive["accepted"][index]):
            raise ValueError("the selected cavity case is not accepted")
        case: dict[str, np.ndarray | float | str] = {
            "x": np.asarray(archive["x"], dtype=float),
            "y": np.asarray(archive["y"], dtype=float),
            "u": np.asarray(archive["u"][index], dtype=float),
            "v": np.asarray(archive["v"][index], dtype=float),
            "omega": np.asarray(archive["omega"][index], dtype=float),
            "reynolds": float(archive["Re"][index]),
            "split": str(archive["split"][index]),
        }
    return case, archive_path


def audit_cavity_case(root: str | Path, reynolds: float = 100.0) -> dict[str, object]:
    """Evaluate the diagnostic on a qualified FlowMLLab cavity field."""

    case, archive_path = load_cavity_case(root, reynolds)
    x = np.asarray(case["x"])
    y = np.asarray(case["y"])
    u = np.asarray(case["u"])
    v = np.asarray(case["v"])
    omega = np.asarray(case["omega"])
    diagnostic = differential_diagnostics(x, y, u, v)
    interior = np.s_[2:-2, 2:-2]
    walls = np.concatenate(
        [
            u[0, 1:-1],
            u[-1, 1:-1] - 1.0,
            u[1:-1, 0],
            u[1:-1, -1],
            v[0, 1:-1],
            v[-1, 1:-1],
            v[1:-1, 0],
            v[1:-1, -1],
        ]
    )
    return {
        "reynolds": float(case["reynolds"]),
        "split": str(case["split"]),
        "grid": [int(y.size), int(x.size)],
        "dataset_sha256": sha256_file(archive_path),
        "interior_divergence_rms": float(
            np.sqrt(np.mean(diagnostic.divergence[interior] ** 2))
        ),
        "interior_divergence_linf": float(np.max(np.abs(diagnostic.divergence[interior]))),
        "archive_vorticity_relative_l2": relative_l2(
            omega[interior], diagnostic.vorticity[interior]
        ),
        "wall_velocity_max_abs_error": float(np.max(np.abs(walls))),
    }


DEFAULT_THRESHOLDS = {
    "observed_order_min": 1.90,
    "fine_vorticity_relative_l2_max": 5.0e-4,
    "manufactured_divergence_rms_max": 1.0e-12,
    "cavity_divergence_rms_max": 1.0e-12,
    "cavity_vorticity_relative_l2_max": 4.0e-2,
    "wall_velocity_max_abs_error_max": 1.0e-12,
}


def evaluate_acceptance(
    verification: dict[str, object],
    cavity: dict[str, object],
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    """Apply predeclared scientific gates and return a machine-readable decision."""

    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    rows = list(verification["rows"])
    fine_row = min(rows, key=lambda row: float(row["h"]))
    gates = {
        "dataset_identity": str(cavity["dataset_sha256"]) == EXPECTED_DATA_SHA256,
        "second_order_verification": float(verification["observed_order"])
        >= limits["observed_order_min"],
        "fine_grid_vorticity": float(fine_row["vorticity_relative_l2"])
        <= limits["fine_vorticity_relative_l2_max"],
        "manufactured_incompressibility": max(float(row["divergence_rms"]) for row in rows)
        <= limits["manufactured_divergence_rms_max"],
        "cavity_incompressibility": float(cavity["interior_divergence_rms"])
        <= limits["cavity_divergence_rms_max"],
        "cavity_vorticity_agreement": float(cavity["archive_vorticity_relative_l2"])
        <= limits["cavity_vorticity_relative_l2_max"],
        "cavity_wall_contract": float(cavity["wall_velocity_max_abs_error"])
        <= limits["wall_velocity_max_abs_error_max"],
    }
    return {
        "decision": "accept" if all(gates.values()) else "reject",
        "gates": gates,
        "thresholds": limits,
        "verification": verification,
        "cavity": cavity,
    }


def write_acceptance_record(record: dict[str, object], path: str | Path) -> Path:
    """Write the deterministic acceptance record without a mutable timestamp."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def validate_week01_1_evidence(root: str | Path) -> dict[str, object]:
    """Recompute Week 1.1 gates and require exact retained evidence agreement."""

    repository = Path(root).resolve()
    record_path = repository / "results" / "week01_1_scientific_software" / "acceptance_record.json"
    if not record_path.is_file():
        raise ValueError("missing Week 1.1 acceptance record")
    retained = json.loads(record_path.read_text(encoding="utf-8"))
    recomputed = evaluate_acceptance(verification_sweep(), audit_cavity_case(repository))
    if retained != recomputed:
        raise ValueError("Week 1.1 retained evidence differs from the recomputed contract")
    if retained["decision"] != "accept" or not all(retained["gates"].values()):
        raise ValueError("Week 1.1 scientific acceptance gate failed")
    return retained


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DifferentialDiagnostics",
    "EXPECTED_DATA_SHA256",
    "audit_cavity_case",
    "differential_diagnostics",
    "evaluate_acceptance",
    "load_cavity_case",
    "manufactured_incompressible_field",
    "relative_l2",
    "sha256_file",
    "verification_sweep",
    "validate_week01_1_evidence",
    "write_acceptance_record",
]
