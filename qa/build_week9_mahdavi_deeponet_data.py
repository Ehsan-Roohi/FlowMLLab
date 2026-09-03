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


def parse_main_zone(path: Path) -> dict[str, np.ndarray]:
    """Read the first POINT-packed Tecplot zone and return its max-y centerline."""
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
    centerline_index = int(np.argmax(np.median(block[:, :, 1], axis=1)))
    centerline = block[centerline_index]
    order = np.argsort(centerline[:, 0])
    return {name: centerline[order, index] for index, name in enumerate(VARIABLES)}


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
    source_hashes: dict[str, str] = {}
    for pressure in NOZZLE_PRESSURES_KPA.astype(int):
        path = source / "data" / f"P={pressure}.dat"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(parse_main_zone(path))
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
        "schema_version": 1,
        "source_repository": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_license": "CC BY 4.0 for DSMC data and reference outputs",
        "source_files_sha256": source_hashes,
        "derivation": (
            "First Tecplot POINT zone (101x31); max-y half-domain symmetry "
            "centerline; seven physical fields retained; density-shock diagnostics "
            "recomputed with FlowMLLab's documented detector."
        ),
        "derived_files": {archive_path.name: digest(archive_path)},
        "paper_sources": {
            "microstep_doi": "https://doi.org/10.1007/s10404-026-02899-8",
            "microstep_arxiv": "https://arxiv.org/abs/2509.17254",
            "micro_nozzle_arxiv": "https://arxiv.org/abs/2605.12723",
            "micro_nozzle_data": SOURCE_URL,
            "micro_nozzle_doi": "https://doi.org/10.1063/5.0343101",
        },
        "claim_boundary": {
            "microstep_teaching_demo": (
                "The generated backward-facing-step fields are manufactured teaching "
                "fields, not the article's DSMC snapshots or a reproduction of its checkpoint."
            ),
            "microstep_retained_evidence": (
                "Reported article metrics are transcribed as immutable evidence and are "
                "never presented as notebook-generated results."
            ),
            "micro_nozzle_data": (
                "The compact archive is a deterministic centerline derivative of all 15 "
                "public DSMC snapshots at the pinned source commit."
            ),
            "micro_nozzle_teaching_model": (
                "The notebook's compact centerline POD/branch model is not the paper's "
                "trained full two-dimensional six-output shock-aligned surrogate."
            ),
        },
        "builder_pod_audit": pod_reference,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"archive": str(archive_path), "sha256": digest(archive_path), "pod": pod_reference}, indent=2))


if __name__ == "__main__":
    main()
