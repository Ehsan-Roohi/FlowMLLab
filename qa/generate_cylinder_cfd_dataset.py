#!/usr/bin/env python3
"""Generate the retained Week-7 cylinder CFD dataset and its manifest.

The split is case-wise: no temporal window from validation or test cases is
used for model fitting.  Each NPZ is independently downloadable and carries
the full velocity/pressure snapshots, force history, geometry, and provenance.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flowmllab.cylinder_lbm import simulate_cylinder  # noqa: E402


SPLIT = {
    60: "development", 80: "development", 90: "development",
    95: "fresh_test", 100: "validation", 105: "retained_test",
    110: "development", 120: "development", 140: "development",
}
SETTINGS = {
    "nx": 240, "ny": 96, "diameter": 12.0, "center": (60.0, 47.5),
    "inflow_velocity": 0.05, "steps": 22000, "history_stride": 4,
    "snapshot_start": 15000, "snapshot_stride": 25,
    "perturbation": 1.0e-2, "seed": 690,
    "collision_model": "trt", "cylinder_boundary": "bouzidi",
}


def filename(reynolds: int) -> str:
    return f"cylinder_cfd_re{reynolds:03d}.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(reynolds: int, output: str) -> dict[str, object]:
    destination = Path(output) / filename(reynolds)
    started = time.perf_counter()
    result = simulate_cylinder(reynolds, **SETTINGS)
    speed = SETTINGS["inflow_velocity"]
    np.savez_compressed(
        destination,
        reynolds=np.asarray(float(reynolds)), split=np.asarray(SPLIT[reynolds]),
        u=np.asarray(result["snapshots"]["u"] / speed, dtype=np.float32),
        v=np.asarray(result["snapshots"]["v"] / speed, dtype=np.float32),
        p=np.asarray(result["snapshots"]["p"] / speed**2, dtype=np.float32),
        solid=np.asarray(result["solid"], dtype=np.uint8),
        x=np.asarray(result["x"], dtype=np.float32),
        y=np.asarray(result["y"], dtype=np.float32),
        snapshot_time=np.asarray(result["snapshot_time"], dtype=np.float32),
        time=np.asarray(result["time"], dtype=np.float32),
        drag_coefficient=np.asarray(result["drag_coefficient"], dtype=np.float32),
        lift_coefficient=np.asarray(result["lift_coefficient"], dtype=np.float32),
        mean_density_ratio=np.asarray(result["mean_density_ratio"], dtype=np.float32),
        strouhal=np.asarray(float(result["strouhal"])),
        metadata_json=np.asarray(json.dumps(result["metadata"], sort_keys=True)),
    )
    strouhal = float(result["strouhal"])
    return {
        "reynolds": reynolds, "split": SPLIT[reynolds], "file": destination.name,
        "bytes": destination.stat().st_size, "sha256": sha256(destination),
        "snapshots": int(result["snapshots"]["u"].shape[0]),
        "strouhal": strouhal if np.isfinite(strouhal) else None,
        "strouhal_resolved": bool(np.isfinite(strouhal)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "cylinder_cfd")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_case, re, str(args.output)): re for re in SPLIT}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    manifest = {
        "schema_version": 1,
        "description": "FlowMLLab circular-cylinder D2Q9 TRT CFD snapshots",
        "split_contract": {
            "development": [60, 80, 90, 110, 120, 140],
            "validation": [100], "fresh_test": [95], "retained_test": [105],
        },
        "settings": SETTINGS, "cases": sorted(rows, key=lambda row: row["reynolds"]),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
