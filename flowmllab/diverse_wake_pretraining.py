"""Utilities for trajectory-split, MAPA-inspired diverse-wake pretraining (Week 7.4)."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np


WEEK07_4_RELEASE_PREFIX = (
    "https://github.com/Ehsan-Roohi/FlowMLLab/releases/download/"
    "week07-4-wakes-v1/"
)


def ensure_release_data(data_dir: str | Path) -> dict:
    """Download missing Week 7.4 assets and verify every frozen SHA-256."""
    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for case in manifest["cases"]:
        target = data_dir / case["file"]
        expected = case["sha256"]
        if target.is_file() and sha256(target) == expected:
            continue
        partial = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(WEEK07_4_RELEASE_PREFIX + case["file"], partial)
        if sha256(partial) != expected:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 mismatch for {case['file']}")
        partial.replace(target)
    return manifest


def load_manifest(data_dir: str | Path) -> dict:
    """Load and validate the frozen trajectory split."""
    path = Path(data_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest["split_contract"]
    groups = [set(split[name]) for name in ("development", "validation", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("development, validation and test trajectories must be disjoint")
    if sum(map(len, groups)) != len(manifest["cases"]):
        raise ValueError("every retained trajectory must have exactly one role")
    return manifest


def load_case(data_dir: str | Path, reynolds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transverse velocity, lift and snapshot times for one trajectory."""
    with np.load(Path(data_dir) / f"re{int(reynolds):03d}.npz", allow_pickle=False) as data:
        fields = np.asarray(data["v"], float)
        lift = np.asarray(data["lift"], float)
        times = np.asarray(data["snapshot_time"], float)
    if fields.ndim != 3 or fields.shape[0] != len(lift) or len(lift) != len(times):
        raise ValueError("trajectory arrays are inconsistent")
    if not np.isfinite(fields).all() or not np.isfinite(lift).all() or not np.isfinite(times).all():
        raise ValueError("trajectory contains non-finite values")
    return fields, lift, times


def normalized_rmse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """RMSE divided by the reference standard deviation, reported in percent."""
    prediction = np.asarray(prediction, float)
    reference = np.asarray(reference, float)
    scale = reference.std(ddof=0)
    if scale <= np.finfo(float).eps:
        return float("nan")
    return float(100.0 * np.sqrt(np.mean((prediction - reference) ** 2)) / scale)


def summarize_trajectory_records(records: list[dict]) -> dict:
    """Aggregate seeds within a trajectory, then report sample SD across test trajectories."""
    curves: dict[str, dict[int, dict[str, float]]] = {}
    methods = sorted({row["method"] for row in records})
    for method in methods:
        curves[method] = {}
        ks = sorted({int(row["k"]) for row in records if row["method"] == method})
        for k in ks:
            target_means = []
            for reynolds in sorted({int(row["reynolds"]) for row in records}):
                values = [row["nrmse"] for row in records
                          if row["method"] == method and int(row["k"]) == k
                          and int(row["reynolds"]) == reynolds]
                target_means.append(float(np.mean(values)))
            values = np.asarray(target_means)
            curves[method][k] = {"mean": float(values.mean()),
                                 "std_across_trajectories": float(values.std(ddof=1)),
                                 "min": float(values.min()), "max": float(values.max()),
                                 "n_trajectories": int(len(values))}
    return curves


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
