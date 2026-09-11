"""Compare a completed rectangular-cavity PINN checkpoint against Nektar++ vortex data.

This is a post-only, case-matched extraction audit. It reloads the retained
PINN checkpoint, evaluates its streamfunction on a fixed physical grid and
locates the corresponding alternating extrema near independently extracted
Nektar++ centres. It does not turn a residual-minimised PINN into validated CFD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_week42_deepplasma import load_upstream
from run_week13_rectangular_pinn import fields


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--nektar-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--nx", type=int, default=241)
    p.add_argument("--ny", type=int, default=1201)
    a = p.parse_args()

    reference = json.loads(a.nektar_json.read_text(encoding="utf-8"))
    case = reference["case"]
    if case["Re"] != 500 or case["depth_over_width"] != 5:
        raise ValueError("Expected the retained Re=500, D/W=5 Nektar++ audit")
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    if cfg["reynolds_number"] != 500 or cfg["aspect_ratio"] != 5:
        raise ValueError("Checkpoint and reference cases do not match")

    module, source_sha = load_upstream(a.source.resolve())
    model = module.PINN().to(module.device)
    model.load_state_dict(saved["model"])
    model.eval()

    x = np.linspace(0.0, 1.0, a.nx)
    y = np.linspace(0.0, 5.0, a.ny)
    eta = y / 5.0
    xx, ee = np.meshgrid(x, eta)
    flat = np.column_stack((xx.ravel(), ee.ravel()))
    psi_parts = []
    for first in range(0, len(flat), 32768):
        q = torch.as_tensor(flat[first:first + 32768], dtype=torch.float64,
                            device=module.device).requires_grad_(True)
        _, _, _, psi = fields(module, model, q, 5.0)
        psi_parts.append(psi.detach().cpu().numpy().ravel())
    psi = np.concatenate(psi_parts).reshape(a.ny, a.nx)

    rows = []
    for ref in reference["vortices"]:
        xn, yn, psin, _ = ref["computed"]
        mask_x = (x >= max(0.1, xn - 0.16)) & (x <= min(0.9, xn + 0.16))
        mask_y = (y >= max(0.0, yn - 0.38)) & (y <= min(5.0, yn + 0.38))
        local = psi[np.ix_(mask_y, mask_x)]
        iy, ix = np.unravel_index(np.argmin(local) if psin < 0 else np.argmax(local), local.shape)
        xp, yp = x[mask_x][ix], y[mask_y][iy]
        psip = float(local[iy, ix])
        resolved = bool(psip * psin > 0.0)
        rows.append({
            "vortex": ref["vortex"],
            "nektar": {"x": xn, "y": yn, "psi": psin},
            "pinn": {"x": float(xp), "y": float(yp), "psi": psip},
            "centre_distance_W": float(np.hypot(xp - xn, yp - yn)),
            "status": "compared" if resolved else "not_resolved_with_reference_sign",
            "psi_relative_difference_percent": (
                float(100.0 * abs(psip / psin - 1.0)) if resolved else None
            ),
        })

    result = {
        "claim_status": "case-matched-vortex-comparison-not-validation",
        "case": {"Re": 500, "depth_over_width": 5},
        "pinn_checkpoint": str(a.checkpoint.resolve()),
        "pinn_steps": {"Adam": saved["adam_done"], "SSBroyden2": saved["ssb_done"]},
        "nektar_source": reference["source_directory"],
        "network_source_sha256": source_sha,
        "evaluation_grid": [a.nx, a.ny],
        "vortices": rows,
        "limitations": [
            "PINN extrema are grid-located; Nektar++ centres use element-local root refinement.",
            "This compares vortex centres and streamfunction, not full velocity or pressure fields.",
            "The PINN independent residual remains a separate acceptance gate.",
        ],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
