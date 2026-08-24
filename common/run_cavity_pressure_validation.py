#!/usr/bin/env python3
"""Regenerate the two retained Re=1000 pressure-validation CFD runs."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

import w4utils


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "article_validation"


def save_result(result: dict, path: Path) -> None:
    np.savez_compressed(
        path,
        x=result["x"], y=result["y"], u=result["u"], v=result["v"],
        p=result["p"], psi=result["psi"], omega=result["omega"],
        Re=result["Re"], N=result["N"], dt=result["dt"], U=result["U"],
        steps=result["steps"], final_residual=result["final_residual"],
        residual_steps=result["residual_steps"], residual_values=result["residual_values"],
        pressure_grad_rel_residual=result["pressure_grad_rel_residual"],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    specifications = (
        {"N": 65, "dt": 0.001, "max_steps": 40000, "tol": 1.0e-7},
        {"N": 129, "dt": 0.00075, "max_steps": 54000, "tol": 1.3e-7},
    )
    records = []
    for specification in specifications:
        started = time.perf_counter()
        result = w4utils.run_cavity(
            Re=1000,
            N=specification["N"],
            dt=specification["dt"],
            max_steps=specification["max_steps"],
            min_steps=specification["max_steps"] // 2,
            check_every=500,
            tol=specification["tol"],
            consecutive_required=3,
            verbose=True,
        )
        result_path = OUTPUT / f"re1000_n{specification['N']}.npz"
        save_result(result, result_path)
        records.append(
            {
                **specification,
                "output": str(result_path.relative_to(ROOT)),
                "steps_executed": int(result["steps"]),
                "converged": bool(result["converged"]),
                "final_residual": float(result["final_residual"]),
                "pressure_gradient_relative_mismatch_global": float(result["pressure_grad_rel_residual"]),
                "runtime_s": float(time.perf_counter() - started),
            }
        )
    (OUTPUT / "recompute_log.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
