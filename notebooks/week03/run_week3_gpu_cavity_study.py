#!/usr/bin/env python3
"""Staged batch runner for the optional Week 3 GPU cavity study (section 11).

The notebook calls this script with ``--stage smoke|grid|ppc|final``.  Each
stage launches the course VHS--NTC solver (``dsmc_cavity_vhs_ntc_gpu_week3.py``)
once per planned case, in a subprocess, with the parameters of the run plan
below.  The plan is the same table that the notebook displays in section 11.

Completed cases (a case folder that already contains ``metadata.json``) are
skipped, so the four stages can be executed in separate sessions and the
notebook's loader can audit whatever has finished.  Nothing is fabricated for a
case that has not run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

KN_VALIDATION = 0.05

# stage, case name, nx, ny, particles per cell, steps, sampling start, sampling stride
RUN_PLAN = [
    ("smoke", "smoke_40_ppc8", 40, 40, 8, 500, 250, 5),
    ("grid", "grid_40x40_ppc16", 40, 40, 16, 4000, 1500, 5),
    ("grid", "grid_60x60_ppc16", 60, 60, 16, 4000, 1500, 5),
    ("grid", "grid_80x80_ppc16", 80, 80, 16, 4000, 1500, 5),
    ("ppc", "ppc_8_grid80", 80, 80, 8, 4000, 1500, 5),
    ("ppc", "ppc_16_grid80", 80, 80, 16, 4000, 1500, 5),
    ("ppc", "ppc_32_grid80", 80, 80, 32, 4000, 1500, 5),
    ("final", "final_long_80x80_ppc32", 80, 80, 32, 20000, 5000, 5),
]
STAGES = ("smoke", "grid", "ppc", "final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=STAGES, help="which group of cases to run")
    parser.add_argument("--solver", default="dsmc_cavity_vhs_ntc_gpu_week3.py", help="path to the course solver")
    parser.add_argument("--output-root", default="week3_gpu_output", help="folder that receives one sub-folder per case")
    parser.add_argument("--Kn", type=float, default=KN_VALIDATION)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--debug-cpu", action="store_true",
                        help="forward the solver's tiny CPU development mode (not valid for submission)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without launching anything")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    solver = Path(args.solver)
    if not solver.exists():
        raise FileNotFoundError(f"Missing solver: {solver.resolve()}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    planned = [case for case in RUN_PLAN if case[0] == args.stage]
    log = []
    for stage, name, nx, ny, ppc, steps, sample_start, sample_stride in planned:
        case_dir = output_root / name
        if (case_dir / "metadata.json").exists():
            print(f"SKIP {name}: metadata.json already present in {case_dir}")
            log.append({"case": name, "status": "skipped-complete"})
            continue
        command = [
            sys.executable, "-u", str(solver),
            "--Kn", str(args.Kn), "--nx", str(nx), "--ny", str(ny), "--ppc", str(ppc),
            "--steps", str(steps), "--sample-start", str(sample_start), "--sample-stride", str(sample_stride),
            "--seed", str(args.seed), "--output-dir", str(output_root), "--case-name", name,
        ]
        if args.debug_cpu:
            command.append("--debug-cpu")
        print("$", " ".join(command), flush=True)
        if args.dry_run:
            log.append({"case": name, "status": "dry-run"})
            continue
        started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - started
        status = "completed" if completed.returncode == 0 else f"failed (return code {completed.returncode})"
        print(f"{name}: {status} after {elapsed/60:.1f} min", flush=True)
        log.append({"case": name, "status": status, "seconds": elapsed})
        if completed.returncode != 0:
            break
    (output_root / f"stage_{args.stage}_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return 0 if all(entry["status"] in ("completed", "skipped-complete", "dry-run") for entry in log) else 1


if __name__ == "__main__":
    sys.exit(main())
