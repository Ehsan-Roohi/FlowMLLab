#!/usr/bin/env python3
"""Build the compact Week-9 DSMC centerline archive from the public source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.mahdavi_deeponet import (  # noqa: E402
    NOZZLE_PRESSURES_KPA,
    STEP_HEIGHT_ARCHIVE_SHA256,
    STEP_TEACHING_RESULT_SHA256,
    detect_density_shock,
    density_snapshot_matrix,
    pod_spectrum,
)


SOURCE_COMMIT = "e1b234ba499408d3b6224633972f939f3b2301d6"
SOURCE_URL = "https://github.com/Ehsan-Roohi/roohi-nozzle-pod-reproducibility"
VARIABLES = (
    "x_m", "y_m", "density", "qx", "qy", "temperature_k", "u_ms",
    "v_ms", "txy", "mach", "pressure_tecplot", "knudsen",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_main_zone_grid(path: Path) -> dict[str, np.ndarray]:
    """Read the first 101 by 31 POINT-packed Tecplot zone as 2-D arrays."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    zone_start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("ZONE"))
    header = " ".join(lines[zone_start:zone_start + 5])
    i_match = re.search(r"\bI\s*=\s*(\d+)", header, re.IGNORECASE)
    j_match = re.search(r"\bJ\s*=\s*(\d+)", header, re.IGNORECASE)
    if i_match is None or j_match is None:
        raise ValueError(f"cannot parse Tecplot dimensions from {path}")
    ni, nj = int(i_match.group(1)), int(j_match.group(1))
    data_start = next(
        i + 1 for i in range(zone_start, len(lines))
        if "DATAPACKING" in lines[i].upper()
    )
    values: list[float] = []
    expected = ni * nj * len(VARIABLES)
    for line in lines[data_start:]:
        if line.strip().upper().startswith("ZONE"):
            break
        try:
            values.extend(float(token.replace("D", "E")) for token in line.replace(",", " ").split())
        except ValueError:
            continue
        if len(values) >= expected:
            break
    if len(values) < expected:
        raise ValueError(f"{path.name}: expected {expected} values, found {len(values)}")
    block = np.asarray(values[:expected], dtype=float).reshape(nj, ni, len(VARIABLES))
    return {name: block[:, :, index] for index, name in enumerate(VARIABLES)}


def parse_main_zone(path: Path) -> dict[str, np.ndarray]:
    """Return the exported max-y row; raw V there is not symmetry-consistent."""
    grid = parse_main_zone_grid(path)
    centerline_index = int(np.argmax(np.median(grid["y_m"], axis=1)))
    order = np.argsort(grid["x_m"][centerline_index])
    return {
        name: values[centerline_index, order]
        for name, values in grid.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Checkout of Ehsan-Roohi/roohi-nozzle-pod-reproducibility",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    source = args.source.resolve()
    output_dir = args.root.resolve() / "results" / "mahdavi_deeponet"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    grid_rows = []
    source_hashes: dict[str, str] = {}
    for pressure in NOZZLE_PRESSURES_KPA.astype(int):
        path = source / "data" / f"P={pressure}.dat"
        if not path.is_file():
            raise FileNotFoundError(path)
        grid = parse_main_zone_grid(path)
        grid_rows.append(grid)
        centerline_index = int(np.argmax(np.median(grid["y_m"], axis=1)))
        order = np.argsort(grid["x_m"][centerline_index])
        rows.append({
            name: values[centerline_index, order]
            for name, values in grid.items()
        })
        source_hashes[path.name] = digest(path)
    x_m = rows[0]["x_m"]
    if not all(np.allclose(row["x_m"], x_m, rtol=0.0, atol=1.0e-12) for row in rows):
        raise ValueError("the 15 centerline x grids are not identical")

    archive: dict[str, np.ndarray] = {
        "pressure_kpa": NOZZLE_PRESSURES_KPA,
        "x_m": x_m,
        "centerline_y_m": np.asarray([np.median(row["y_m"]) for row in rows]),
    }
    for name in ("density", "temperature_k", "u_ms", "v_ms", "mach", "pressure_tecplot", "knudsen"):
        archive[name] = np.vstack([row[name] for row in rows])
    shock = [detect_density_shock(x_m, row["density"]) for row in rows]
    archive["shock_x_m"] = np.asarray([item["shock_x_m"] for item in shock])
    archive["delta_jump_m"] = np.asarray([item["delta_jump_m"] for item in shock])

    archive_path = output_dir / "nozzle_centerline_15cases.npz"
    np.savez_compressed(archive_path, **archive)

    x_grid = grid_rows[0]["x_m"]
    y_grid = grid_rows[0]["y_m"]
    for case in grid_rows[1:]:
        if not np.allclose(case["x_m"], x_grid, rtol=0.0, atol=1.0e-12):
            raise ValueError("the 15 full-field x grids are not identical")
        if not np.allclose(case["y_m"], y_grid, rtol=0.0, atol=1.0e-12):
            raise ValueError("the 15 full-field y grids are not identical")
    field_archive: dict[str, np.ndarray] = {
        "pressure_kpa": NOZZLE_PRESSURES_KPA,
        "x_m": x_grid,
        "y_m": y_grid,
        "centerline_index": np.asarray(
            int(np.argmax(np.median(y_grid, axis=1))), dtype=int
        ),
    }
    for name in (
        "density", "temperature_k", "u_ms", "v_ms", "mach",
        "pressure_tecplot", "knudsen",
    ):
        field_archive[name] = np.stack([case[name] for case in grid_rows])
    field_archive_path = output_dir / "nozzle_fields_15cases.npz"
    np.savez_compressed(field_archive_path, **field_archive)

    pod_reference = {}
    for coordinate in ("physical", "shock_centered"):
        _, matrix = density_snapshot_matrix(
            archive["x_m"], archive["density"], archive["shock_x_m"],
            archive["delta_jump_m"], coordinate=coordinate,
        )
        spectrum = pod_spectrum(matrix)
        pod_reference[coordinate] = {
            "E1_percent": spectrum["first_mode_percent"],
            "E12_percent": spectrum["first_two_percent"],
            "E123_percent": spectrum["first_three_percent"],
            "N99": spectrum["n99"],
        }

    provenance = {
        "schema_version": 3,
        "source_repository": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_license": "CC BY 4.0 for DSMC data and reference outputs",
        "source_files_sha256": source_hashes,
        "derivation": (
            "First Tecplot POINT zone (101x31); seven full physical fields retained "
            "on the original common grid; max-y exported boundary row "
            "extracted separately; density-shock diagnostics recomputed with "
            "FlowMLLab's documented detector."
        ),
        "boundary_caveat": "The max-y row is at the stated symmetry plane, but exported V is nonzero. Original labels are preserved; this is not a boundary-condition validation dataset.",
        "derived_files": {
            archive_path.name: digest(archive_path),
            field_archive_path.name: digest(field_archive_path),
            **STEP_HEIGHT_ARCHIVE_SHA256,
            **STEP_TEACHING_RESULT_SHA256,
        },
        "paper_sources": {
            "microstep_doi": "https://doi.org/10.1007/s10404-026-02899-8",
            "microstep_arxiv": "https://arxiv.org/abs/2509.17254",
            "microstep_data": "https://github.com/Ehsan-Roohi/roohi-step-dnn-mahdavi",
            "microstep_data_commit": "c3f211376b42b8dc30daad380eaef5e0ab800b5c",
            "micro_nozzle_arxiv": "https://arxiv.org/abs/2605.12723",
            "micro_nozzle_data": SOURCE_URL,
            "micro_nozzle_doi": "https://doi.org/10.1063/5.0343101",
        },
        "claim_boundary": {
            "microstep_data": (
                "The lesson uses two compact derivatives of nine real DSMC height fields "
                "at a pinned source commit, published in FlowMLLab with the corresponding "
                "author's explicit permission; no general upstream-data license is implied."
            ),
            "microstep_teaching_model": (
                "The leakage-free coordinate surrogate receives only h/H, x, y, and known "
                "geometry; it does not receive U, V, pressure, target masks, or target-field patches."
            ),
            "microstep_published_model": (
                "The published-repository C-DeepONet is retained as a privileged-input "
                "reconstruction because its inference patches come from the held-out DSMC U,V field."
            ),
            "microstep_scope": (
                "Kn and h/H generalization are demonstrated separately, not jointly; "
                "the height study is at fixed Kn=0.01."
            ),
            "microstep_dsmc_validation": (
                "The public fields do not establish cell-size, time-step, particles-per-cell, "
                "sampling/replicate uncertainty, or wall accommodation metadata."
            ),
            "micro_nozzle_data": (
                "The compact archives are deterministic full-field and centerline "
                "derivatives of all 15 public DSMC snapshots at the pinned source commit."
            ),
            "micro_nozzle_teaching_model": (
                "The notebook audits shock-centered POD compression and evaluates physical-coordinate "
                "and source-only shock-aligned interpolation baselines. These are not the paper's "
                "trained six-output DeepONet."
            ),
        },
        "builder_pod_audit": pod_reference,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "centerline_archive": str(archive_path),
        "centerline_sha256": digest(archive_path),
        "field_archive": str(field_archive_path),
        "field_sha256": digest(field_archive_path),
        "pod": pod_reference,
    }, indent=2))


if __name__ == "__main__":
    main()
