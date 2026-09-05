"""Build the compact Week-7.1 cylinder dataset from the author archive.

The source archive is intentionally not committed.  This builder reads each
400 x 400 Tecplot snapshot directly from the zip file, selects a deterministic
50 x 50 spatial lattice, rejects solid/sentinel values, and writes a compact
NPZ plus a machine-readable provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np


EXPECTED_ARCHIVE_SHA256 = (
    "bda221759a372a2054ca4c76bde923febf0b682404b41f67cc190f9b03c71f7b"
)
GRID_SIZE = 400
SAMPLE_SIZE = 50
COLUMNS = (
    "X", "Y", "ND", "D", "U", "V", "W", "TTR", "TRT", "TVB",
    "TOV", "MA", "MC", "MCT", "MFP", "SOF", "FSP", "ANG", "P",
    "Angular momentum/",
)
TARGET_INDICES = (11, 10, 18)  # local Mach, T/T_inf, p/p_inf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mach_from_entry(name: str) -> float:
    """Decode archive tokens such as M55 -> 5.5 and M825 -> 8.25."""
    match = re.search(r"StructuredGridM(\d+)\.dat$", name)
    if match is None:
        raise ValueError(f"Cannot parse Mach number from {name!r}")
    token = match.group(1)
    if len(token) == 2 and 10 <= int(token) <= 15:
        return float(token)
    if len(token) == 1:
        return float(token)
    return float(f"{token[0]}.{token[1:]}")


def read_snapshot(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_axis = np.linspace(0, GRID_SIZE - 1, SAMPLE_SIZE, dtype=int)
    selected_rows = {
        int(j * GRID_SIZE + i) for j in selected_axis for i in selected_axis
    }
    coordinates: list[tuple[float, float]] = []
    targets: list[tuple[float, float, float]] = []
    source_rows: list[int] = []

    with archive.open(entry) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
        for _ in range(3):
            next(stream)
        for row, line in enumerate(stream):
            if row not in selected_rows:
                continue
            values = np.fromstring(line, sep=" ")
            if values.size != len(COLUMNS):
                raise ValueError(
                    f"{entry.filename}: row {row} has {values.size} columns; "
                    f"expected {len(COLUMNS)}"
                )
            xy = values[:2]
            q = values[list(TARGET_INDICES)]
            valid = np.isfinite(xy).all() and np.isfinite(q).all()
            valid = valid and bool(np.all(np.abs(q) < 1.0e20))
            if valid:
                coordinates.append((float(xy[0]), float(xy[1])))
                targets.append((float(q[0]), float(q[1]), float(q[2])))
                source_rows.append(row)

    if not coordinates:
        raise ValueError(f"No finite teaching points found in {entry.filename}")
    return (
        np.asarray(coordinates, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(source_rows, dtype=np.int32),
    )


def build(archive_path: Path, output_path: Path) -> None:
    archive_hash = sha256(archive_path)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(
            "Archive hash does not match the reviewed AllMachNNCylinder.zip: "
            f"{archive_hash}"
        )

    all_xy: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_mach: list[np.ndarray] = []
    all_case: list[np.ndarray] = []
    all_rows: list[np.ndarray] = []
    cases: list[dict[str, object]] = []

    with zipfile.ZipFile(archive_path) as archive:
        entries = [
            item
            for item in archive.infolist()
            if re.search(r"(^|/)structured/StructuredGridM\d+\.dat$", item.filename)
        ]
        entries.sort(key=lambda item: mach_from_entry(item.filename))
        if len(entries) != 20:
            raise ValueError(f"Expected 20 structured snapshots; found {len(entries)}")

        for case_id, entry in enumerate(entries):
            mach = mach_from_entry(entry.filename)
            xy, targets, rows = read_snapshot(archive, entry)
            all_xy.append(xy)
            all_targets.append(targets)
            all_mach.append(np.full(len(xy), mach, dtype=np.float32))
            all_case.append(np.full(len(xy), case_id, dtype=np.int16))
            all_rows.append(rows)
            cases.append(
                {
                    "case_id": case_id,
                    "mach_inf": mach,
                    "source_entry": entry.filename,
                    "source_grid": [GRID_SIZE, GRID_SIZE],
                    "selected_points_before_filter": SAMPLE_SIZE * SAMPLE_SIZE,
                    "retained_points": int(len(xy)),
                }
            )
            print(f"M_inf={mach:g}: retained {len(xy):,} points")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mach_inf=np.concatenate(all_mach),
        x=np.concatenate([xy[:, 0] for xy in all_xy]),
        y=np.concatenate([xy[:, 1] for xy in all_xy]),
        targets=np.vstack(all_targets),
        case_id=np.concatenate(all_case),
        source_row=np.concatenate(all_rows),
        target_names=np.asarray(["local_mach", "temperature_ratio", "pressure_ratio"]),
    )

    data_hash = sha256(output_path)
    manifest = {
        "schema_version": 1,
        "artifact": output_path.name,
        "artifact_sha256": data_hash,
        "source_archive": archive_path.name,
        "source_archive_sha256": archive_hash,
        "source_description": (
            "Author-supplied structured DSMC fields for the hypersonic "
            "rarefied-cylinder study"
        ),
        "paper_doi": "10.1063/5.0334590",
        "license": "Released by the corresponding author for FlowMLLab teaching use",
        "derivation": {
            "grid_selection": (
                "50 deterministic indices per coordinate from each 400 x 400 grid"
            ),
            "invalid_filter": "finite targets with absolute value below 1e20",
            "committed_columns": [
                "mach_inf", "x", "y", "local_mach", "temperature_ratio",
                "pressure_ratio", "case_id", "source_row",
            ],
        },
        "cases": cases,
        "total_points": int(sum(case["retained_points"] for case in cases)),
    }
    manifest_path = output_path.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KiB)")
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hypersonic_cylinder/cylinder_teaching_subset.npz"),
    )
    args = parser.parse_args()
    build(args.archive.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
