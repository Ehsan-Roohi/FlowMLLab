"""Complete the four missing 64-cell pressure-coupling comparisons.

Run one case per process, from rest, with the retained spatial scheme and
stopping criteria. This runner does not alter the teaching solver.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from flowmllab.pressure_velocity_experiments import ExperimentalConfig, run_experiment

CASES = ((400, "piso"), (400, "pimple"), (1000, "piso"), (1000, "pimple"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=range(len(CASES)), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    re, method = CASES[args.case]
    cfg = ExperimentalConfig(method=method, reynolds=re, cells=64,
        convection="central", dt=.05, velocity_relaxation=.5,
        pressure_relaxation=.2, pressure_correctors=2, outer_correctors=3,
        tolerance=1e-6, max_iterations=6000, check_every=50)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{method}-re{re}-n64"
    metadata = args.output / (stem + ".json")
    archive = args.output / (stem + ".npz")
    if metadata.exists() or archive.exists():
        raise FileExistsError(f"Refusing to overwrite retained case {stem}")
    source_paths = ["flowmllab/pressure_velocity.py",
                    "flowmllab/pressure_velocity_experiments.py",
                    "flowmllab/ghia.py", "qa/run_week01_2_refinement.py"]
    provenance = {
        "case_id": stem, "config": asdict(cfg),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("SOURCE_COMMIT", "not supplied"),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in source_paths},
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "host": platform.node(),
        "machine": platform.machine(), "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "threads": {k: os.environ.get(k) for k in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "timing_protocol": "One single-thread CPU solve from rest per case on Unity. Other array cases may run concurrently. Wall times must not be ranked against earlier local-machine runs.",
        "recovery_policy": "Fresh finite-duration solve; no checkpoint writes. A failed case is retained and requires a new explicit attempt directory.",
    }
    start = time.perf_counter()
    print("START", stem, flush=True)
    result = run_experiment(cfg, progress=True)
    elapsed = time.perf_counter() - start
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    finite = all(np.isfinite(value).all() for value in arrays.values())
    gates = {"finite_arrays": finite, "native_converged": bool(result["converged"]),
             "momentum_below_1e-6": bool(result["momentum_residual_linf"] < 1e-6),
             "continuity_below_1e-9": bool(result["continuity_linf"] < 1e-9)}
    record = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(archive, **arrays)
    record.update(provenance, status="converged" if all(gates.values()) else "not_converged",
                  admission_checks=gates, end_to_end_seconds=elapsed,
                  completed_utc=datetime.now(timezone.utc).isoformat(),
                  field_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("DONE", stem, record["status"], elapsed, flush=True)
    if not all(gates.values()):
        raise RuntimeError("Retained last iterate failed at least one unchanged convergence gate")


if __name__ == "__main__":
    main()
