#!/usr/bin/env python3
"""Generate compact, trajectory-split LBM wakes for the Week 7.4 study."""
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
sys.path.insert(0, str(ROOT))
from flowmllab.cylinder_lbm import simulate_cylinder  # noqa: E402

REYNOLDS = tuple(range(60, 136, 5))
VALIDATION = (105,)
TEST = (95, 115, 125, 135)
DEVELOPMENT = tuple(re for re in REYNOLDS if re not in VALIDATION + TEST)
SETTINGS = {
    "nx": 120, "ny": 48, "diameter": 6.0, "center": (30.0, 23.5),
    "inflow_velocity": 0.05, "steps": 10000, "history_stride": 5,
    "snapshot_start": 5000, "snapshot_stride": 20,
    "collision_model": "trt", "cylinder_boundary": "bouzidi",
}
CROP = {"y": [8, 40], "x": [30, 108]}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(reynolds: int, destination: str) -> dict:
    output = Path(destination) / f"re{reynolds:03d}.npz"
    seed = 7400 + reynolds
    amplitude = (0.006, 0.010, 0.014)[(reynolds // 5) % 3]
    started = time.perf_counter()
    result = simulate_cylinder(reynolds, perturbation=amplitude, seed=seed, **SETTINGS)
    sy = slice(*CROP["y"]); sx = slice(*CROP["x"])
    snapshot_time = np.asarray(result["snapshot_time"], float)
    lift = np.interp(snapshot_time, np.asarray(result["time"], float),
                     np.asarray(result["lift_coefficient"], float))
    drag = np.interp(snapshot_time, np.asarray(result["time"], float),
                     np.asarray(result["drag_coefficient"], float))
    role = "validation" if reynolds in VALIDATION else "test" if reynolds in TEST else "development"
    scale = SETTINGS["inflow_velocity"]
    np.savez_compressed(
        output,
        reynolds=np.asarray(float(reynolds)), role=np.asarray(role),
        v=np.asarray(result["snapshots"]["v"][:, sy, sx] / scale, np.float32),
        lift=np.asarray(lift, np.float32), drag=np.asarray(drag, np.float32),
        snapshot_time=np.asarray(snapshot_time, np.float32),
        perturbation=np.asarray(amplitude), seed=np.asarray(seed),
    )
    return {"reynolds": reynolds, "role": role, "file": output.name,
            "frames": len(snapshot_time), "bytes": output.stat().st_size,
            "sha256": digest(output), "elapsed_seconds": time.perf_counter() - started,
            "perturbation": amplitude, "seed": seed,
            "strouhal": float(result["strouhal"]) if np.isfinite(result["strouhal"]) else None}


def existing_case(reynolds: int, destination: Path) -> dict:
    output = destination / f"re{reynolds:03d}.npz"
    with np.load(output, allow_pickle=False) as data:
        return {"reynolds": reynolds, "role": str(data["role"]), "file": output.name,
                "frames": int(len(data["snapshot_time"])), "bytes": output.stat().st_size,
                "sha256": digest(output), "elapsed_seconds": None,
                "perturbation": float(data["perturbation"]), "seed": int(data["seed"]),
                "strouhal": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/week07_4_wakes")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = [existing_case(re, args.output) for re in REYNOLDS
            if (args.output / f"re{re:03d}.npz").is_file()]
    pending = [re for re in REYNOLDS if not (args.output / f"re{re:03d}.npz").is_file()]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_case, re, str(args.output)): re for re in pending}
        for future in as_completed(futures):
            row = future.result(); rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    manifest = {
        "schema_version": 1,
        "description": "Compact FlowMLLab LBM trajectories for diverse-wake pretraining",
        "split_contract": {"development": list(DEVELOPMENT), "validation": list(VALIDATION), "test": list(TEST)},
        "settings": SETTINGS, "crop": CROP, "cases": sorted(rows, key=lambda item: item["reynolds"]),
        "limitations": "Quick educational D2Q9-TRT runs; not grid-independent DNS.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("manifest", args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
