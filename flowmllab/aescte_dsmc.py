"""Reproducible data-driven DSMC examples from Roohi and Shoja-Sani (2026).

The routines in this module intentionally keep the raw DSMC fields, the
surrogate fit, and the final assessment as separate objects.  They are small
enough for a classroom notebook but operate on the complete 50 by 50 cavity
fields and 300-point shock profiles distributed with FlowMLLab.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np


ARTICLE_DOI = "10.1016/j.ast.2025.110785"
CAVITY_KNUDSEN = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0])
CAVITY_TRAIN_KNUDSEN = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
CAVITY_TEST_KNUDSEN = np.array([0.05, 0.5])
CAVITY_LID_SPEEDS_MS = np.array([10.0, 30.0])
DIATOMIC_MACH = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
MONATOMIC_MACH = np.array([1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
CAVITY_COLUMNS = (
    "x", "y", "temperature_k", "u_ms", "v_ms", "w_ms", "qx",
    "qy", "txy", "density", "mean_free_path_m",
)
DIATOMIC_COLUMNS = (
    "x_over_lambda", "rotational_temperature", "auxiliary_1",
    "auxiliary_2", "translational_temperature", "normalized_velocity",
)
MONATOMIC_COLUMNS = (
    "x_over_lambda", "density", "velocity", "temperature",
    "translational_temperature", "case_parameter",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_root(root: str | Path) -> Path:
    return Path(root).resolve() / "data" / "aescte_dsmc" / "raw"


def read_point_tecplot(path: str | Path) -> dict[str, np.ndarray]:
    """Read the POINT-packed 50 by 50 DSMC2QT cavity table."""
    source = Path(path)
    names: list[str] = []
    rows: list[list[float]] = []
    ni = nj = None
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("VARIABLES"):
            names = [
                item.strip().lower()
                for item in re.split(r"\s*,\s*", stripped.split("=", 1)[1])
            ]
            continue
        if stripped.upper().startswith("ZONE"):
            mi = re.search(r"\bI\s*=\s*(\d+)", stripped, re.IGNORECASE)
            mj = re.search(r"\bJ\s*=\s*(\d+)", stripped, re.IGNORECASE)
            if mi and mj:
                ni, nj = int(mi.group(1)), int(mj.group(1))
            continue
        if not names:
            continue
        try:
            values = [float(item.replace("D", "E")) for item in stripped.split()]
        except ValueError:
            continue
        if len(values) == len(names):
            rows.append(values)
    if (ni, nj) != (50, 50) or len(rows) != 2500:
        raise ValueError(f"{source}: expected a 50 by 50 POINT zone")
    if tuple(names) != (
        "x", "y", "ovtemp", "u", "v", "w", "qx", "qy", "txy", "rho", "lambda"
    ):
        raise ValueError(f"{source}: unexpected Tecplot columns {names}")
    array = np.asarray(rows, dtype=float).reshape(nj, ni, len(names))
    return {
        key: array[:, :, index]
        for index, key in enumerate(CAVITY_COLUMNS)
    }


def load_cavity_raw(root: str | Path) -> dict[str, np.ndarray]:
    """Load all fourteen complete cavity fields from the committed raw data."""
    base = _data_root(root) / "cavity"
    fields = {name: [] for name in CAVITY_COLUMNS}
    case_lid: list[float] = []
    case_kn: list[float] = []
    for lid in CAVITY_LID_SPEEDS_MS:
        for kn in CAVITY_KNUDSEN:
            path = (
                base / f"lid_{int(lid)}_ms" / f"kn_{kn:g}" / "DSMC2QT.PLT"
            )
            values = read_point_tecplot(path)
            case_lid.append(lid)
            case_kn.append(kn)
            for name in fields:
                fields[name].append(values[name])
    result = {name: np.stack(values) for name, values in fields.items()}
    result["lid_speed_ms"] = np.asarray(case_lid)
    result["knudsen"] = np.asarray(case_kn)
    return result


def save_cavity_archive(root: str | Path, output: str | Path) -> Path:
    data = load_cavity_raw(root)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **data)
    return target


def load_cavity_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    if data["temperature_k"].shape != (14, 50, 50):
        raise ValueError("cavity archive must contain fourteen 50 by 50 cases")
    return data


def cavity_case(data: dict[str, np.ndarray], lid_speed_ms: float, knudsen: float) -> int:
    match = np.flatnonzero(
        np.isclose(data["lid_speed_ms"], lid_speed_ms)
        & np.isclose(data["knudsen"], knudsen)
    )
    if len(match) != 1:
        raise ValueError(f"missing/duplicate cavity case U={lid_speed_ms}, Kn={knudsen}")
    return int(match[0])


def logarithmic_kn_prediction(
    data: dict[str, np.ndarray],
    *,
    lid_speed_ms: float,
    test_knudsen: float,
    fields: Iterable[str] = ("u_ms", "v_ms", "temperature_k"),
) -> dict[str, np.ndarray | float]:
    """Reproduce the article's two-specialist interpolation in log10(Kn)."""
    lower_values = CAVITY_TRAIN_KNUDSEN[CAVITY_TRAIN_KNUDSEN < test_knudsen]
    upper_values = CAVITY_TRAIN_KNUDSEN[CAVITY_TRAIN_KNUDSEN > test_knudsen]
    if not len(lower_values) or not len(upper_values):
        raise ValueError("test Kn must be bracketed by training specialists")
    lower, upper = float(lower_values.max()), float(upper_values.min())
    weight = (np.log10(test_knudsen) - np.log10(lower)) / (
        np.log10(upper) - np.log10(lower)
    )
    lower_index = cavity_case(data, lid_speed_ms, lower)
    upper_index = cavity_case(data, lid_speed_ms, upper)
    result: dict[str, np.ndarray | float] = {
        "lower_knudsen": lower,
        "upper_knudsen": upper,
        "weight": float(weight),
    }
    for field in fields:
        result[field] = (
            (1.0 - weight) * data[field][lower_index]
            + weight * data[field][upper_index]
        )
    return result


def normalized_rmse(reference: np.ndarray, prediction: np.ndarray, scale: float) -> float:
    if scale <= 0:
        raise ValueError("normalization scale must be positive")
    return float(100.0 * np.sqrt(np.mean((prediction - reference) ** 2)) / scale)


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        raise ValueError("relative L2 is undefined for a zero reference")
    return float(100.0 * np.linalg.norm(prediction - reference) / denominator)


def _load_shock_directory(
    directory: Path,
    mach: np.ndarray,
    filenames: list[str],
    columns: tuple[str, ...],
) -> dict[str, np.ndarray]:
    raw = [np.loadtxt(directory / filename) for filename in filenames]
    if any(array.shape != (300, len(columns)) for array in raw):
        raise ValueError(f"{directory}: every shock profile must be 300 by {len(columns)}")
    arrays = np.stack(raw)
    if not np.allclose(arrays[:, :, 0], arrays[0, :, 0], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{directory}: shock coordinate grids differ")
    result = {name: arrays[:, :, index] for index, name in enumerate(columns)}
    result["mach"] = mach.copy()
    return result


def load_diatomic_shock_raw(root: str | Path) -> dict[str, np.ndarray]:
    base = _data_root(root) / "diatomic_shock"
    # The source suffix '.5' is the DSMC rotational-collision setting; Mach is
    # M14 -> 1.4, ..., M19 -> 1.9 as defined in the article.
    names = [f"M{int(round(value * 10))}.5.txt" for value in DIATOMIC_MACH]
    return _load_shock_directory(base, DIATOMIC_MACH, names, DIATOMIC_COLUMNS)


def load_monatomic_shock_raw(root: str | Path) -> dict[str, np.ndarray]:
    base = _data_root(root) / "monatomic_shock"
    names = [f"M{int(round(value * 10))}.txt" for value in MONATOMIC_MACH]
    return _load_shock_directory(base, MONATOMIC_MACH, names, MONATOMIC_COLUMNS)


def save_shock_archive(data: dict[str, np.ndarray], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **data)
    return target


def load_shock_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def fit_pod_polynomial_operator(
    mach: np.ndarray,
    snapshots: np.ndarray,
    training_indices: np.ndarray,
    *,
    rank: int,
    degree: int,
) -> dict[str, np.ndarray | float | int]:
    """Fit a compact branch-trunk operator with POD trunk and polynomial branch."""
    x = np.asarray(mach, dtype=float)[training_indices]
    matrix = np.asarray(snapshots, dtype=float)[training_indices]
    mean = matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(matrix - mean, full_matrices=False)
    effective_rank = min(int(rank), len(training_indices) - 1)
    modes = vh[:effective_rank]
    coefficients = (matrix - mean) @ modes.T
    center = float(x.mean())
    scale = float(x.std())
    design = np.vander((x - center) / scale, int(degree) + 1, increasing=True)
    branch = np.linalg.lstsq(design, coefficients, rcond=None)[0]
    return {
        "mean": mean,
        "modes": modes,
        "branch": branch,
        "mach_center": center,
        "mach_scale": scale,
        "degree": int(degree),
        "rank": effective_rank,
    }


def predict_pod_polynomial_operator(
    model: dict[str, np.ndarray | float | int], query_mach: np.ndarray | list[float]
) -> np.ndarray:
    query = np.atleast_1d(np.asarray(query_mach, dtype=float))
    coordinate = (query - float(model["mach_center"])) / float(model["mach_scale"])
    design = np.vander(coordinate, int(model["degree"]) + 1, increasing=True)
    coefficients = design @ np.asarray(model["branch"])
    return np.asarray(model["mean"])[None, :] + coefficients @ np.asarray(model["modes"])


def maxwell_speed_pdf(speed: np.ndarray, temperature_k: float, molecular_mass_kg: float) -> np.ndarray:
    """Three-dimensional Maxwell speed distribution used in the relaxation lesson."""
    boltzmann = 1.380649e-23
    speed = np.asarray(speed, dtype=float)
    factor = 4.0 * np.pi * (molecular_mass_kg / (2.0 * np.pi * boltzmann * temperature_k)) ** 1.5
    return factor * speed**2 * np.exp(-molecular_mass_kg * speed**2 / (2.0 * boltzmann * temperature_k))


def validate_aescte_evidence(root: str | Path) -> dict[str, object]:
    """Validate the checksummed Week-10 raw tables, archives, and result gates."""
    root_path = Path(root).resolve()
    result_dir = root_path / "results" / "aescte_dsmc"
    manifest = json.loads(
        (result_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result_dir / "validation_summary.json").read_text(encoding="utf-8")
    )
    if manifest["article"]["doi"] != ARTICLE_DOI:
        raise ValueError("unexpected Week-10 article DOI")
    for relative, expected in manifest["source_file_sha256"].items():
        if sha256(root_path / relative) != expected:
            raise ValueError(f"raw DSMC hash mismatch: {relative}")
    for relative, expected in manifest["derived_file_sha256"].items():
        if sha256(root_path / relative) != expected:
            raise ValueError(f"derived DSMC hash mismatch: {relative}")
    if sha256(result_dir / "week10_validation_metrics.csv") != summary["metrics_sha256"]:
        raise ValueError("Week-10 metric-table hash mismatch")
    if summary.get("status") != "pass":
        raise ValueError("Week-10 result gates did not pass")
    if summary["cavity_primary_max_nrmse_percent"] >= 2.0:
        raise ValueError("Week-10 cavity gate failed")
    if summary["shock_max_relative_l2_percent"] >= 1.5:
        raise ValueError("Week-10 shock gate failed")
    return {
        **summary,
        "raw_source_file_count": len(manifest["source_file_sha256"]),
        "article_doi": ARTICLE_DOI,
    }
