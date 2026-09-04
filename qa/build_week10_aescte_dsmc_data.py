#!/usr/bin/env python3
"""Build compact, checksum-tracked Week-10 archives from committed raw DSMC data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flowmllab.aescte_dsmc import (  # noqa: E402
    ARTICLE_DOI,
    load_diatomic_shock_raw,
    load_monatomic_shock_raw,
    save_cavity_archive,
    save_shock_archive,
    sha256,
)


def main() -> None:
    raw = ROOT / "data" / "aescte_dsmc" / "raw"
    output = ROOT / "results" / "aescte_dsmc"
    output.mkdir(parents=True, exist_ok=True)
    cavity = save_cavity_archive(ROOT, output / "cavity_fields_14cases.npz")
    diatomic = save_shock_archive(
        load_diatomic_shock_raw(ROOT), output / "diatomic_shock_6cases.npz"
    )
    monatomic = save_shock_archive(
        load_monatomic_shock_raw(ROOT), output / "monatomic_shock_7cases.npz"
    )
    source_files = sorted(path for path in raw.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "article": {
            "title": "Data-driven surrogate modeling of DSMC solutions using deep neural networks",
            "authors": "Ehsan Roohi and Ahmad Shoja-Sani",
            "journal": "Aerospace Science and Technology 168 (2026) 110785",
            "doi": ARTICLE_DOI,
        },
        "raw_data_contract": {
            "cavity_cases": 14,
            "cavity_grid": [50, 50],
            "lid_speeds_in_attached_data_ms": [10, 30],
            "knudsen_numbers": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
            "diatomic_shock_cases": 6,
            "diatomic_profile_points": 300,
            "monatomic_shock_cases": 7,
            "monatomic_profile_points": 300,
            "sparta_input_sets": [0.01, 0.05, 0.5, 1.0],
        },
        "source_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files
        },
        "derived_file_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (cavity, diatomic, monatomic)
        },
        "column_interpretation": {
            "diatomic": (
                "The article figures identify source columns 1, 4, and 5 after x/lambda "
                "as normalized rotational temperature, translational temperature, and velocity."
            ),
            "source_filename_suffix": (
                "M14.5 through M19.5 map to Mach 1.4 through 1.9; the final .5 is retained "
                "verbatim from the supplied DSMC filenames and is not parsed as Mach 14.5."
            ),
        },
        "known_limits": {
            "cavity_speed_label": (
                "The supplied full-field archives contain lid speeds 10 and 30 m/s. "
                "The article text also discusses 100 m/s; no 100 m/s raw archive was supplied."
            ),
            "diatomic_mach_2": (
                "The article plots a Mach-2.0 extrapolation, but no Mach-2.0 diatomic DSMC "
                "target or trained checkpoint was present in the supplied archive."
            ),
            "monatomic_mach_25": (
                "The public notebook names M25 and M30 extrapolation targets, but those raw "
                "files were absent; only M1.4 through M2.0 can be independently assessed."
            ),
        },
    }
    manifest_path = output / "data_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest["derived_file_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
