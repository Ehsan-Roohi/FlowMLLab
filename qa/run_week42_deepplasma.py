#!/usr/bin/env python3
"""Run and independently audit the permitted DeepPlasma cavity PINN.

The upstream source is fetched separately at a pinned commit; it is not vendored.
This driver deliberately records the distinction between a workflow smoke test and
a scientific training run.  Neither mode is a CFD-reference validation by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

UPSTREAM_COMMIT = "fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b"
UPSTREAM_SHA256 = "391a2174cb9f6e8863c14d7b350077e54209134de9f85a17719c398146f91458"


def load_upstream(path: Path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise RuntimeError(f"DeepPlasma source checksum mismatch: {digest}")
    spec = importlib.util.spec_from_file_location("deepplasma_ldc_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def derivative(field, coord, *, create_graph=True):
    return torch.autograd.grad(
        field, coord, torch.ones_like(field), create_graph=create_graph,
        retain_graph=True,
    )[0]


def raw_residuals(module, model, points: torch.Tensor):
    """Momentum and continuity residuals without the upstream corner mask."""
    x = points[:, :1].detach().clone().requires_grad_(True)
    y = points[:, 1:].detach().clone().requires_grad_(True)
    tracked = torch.cat((x, y), dim=1)
    u, v, p = module.output_transform_cavity_flow(tracked, model(tracked))
    ux, uy = derivative(u, x), derivative(u, y)
    vx, vy = derivative(v, x), derivative(v, y)
    uxx, uyy = derivative(ux, x), derivative(uy, y)
    vxx, vyy = derivative(vx, x), derivative(vy, y)
    px, py = derivative(p, x), derivative(p, y)
    rx = u * ux + v * uy - (uxx + uyy) / module.Re + px
    ry = u * vx + v * vy - (vxx + vyy) / module.Re + py
    return rx, ry, ux + vy


def rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(value.detach() ** 2)).cpu())


def wall_audit(module, model, device, n=1001):
    s = torch.linspace(0.0, 1.0, n, device=device).reshape(-1, 1)
    walls = {
        "bottom": torch.cat((s, torch.zeros_like(s)), 1),
        "left": torch.cat((torch.zeros_like(s), s), 1),
        "right": torch.cat((torch.ones_like(s), s), 1),
        "top": torch.cat((s, torch.ones_like(s)), 1),
    }
    result = {}
    for name, xy in walls.items():
        xy.requires_grad_(True)
        u, v, _ = module.output_transform_cavity_flow(xy, model(xy))
        target_u = ((1 - torch.exp(-(s - 1) ** 2 / module.dx**2))
                    * (1 - torch.exp(-s**2 / module.dx**2))) if name == "top" else torch.zeros_like(s)
        result[name] = {"u_max_abs_error": float((u - target_u).abs().max().detach().cpu()),
                        "v_max_abs_error": float(v.abs().max().detach().cpu())}
    return result


def render(module, model, device, output: Path, n=181):
    axis = torch.linspace(0.0, 1.0, n, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx.ravel(), yy.ravel()), 1).requires_grad_(True)
    u, v, p = module.output_transform_cavity_flow(xy, model(xy))
    fields = [u, v, p - p.mean()]
    titles = [r"$u/U$", r"$v/U$", r"$(p-\bar p)/(\rho U^2)$"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)
    for ax, field, title in zip(axes, fields, titles):
        z = field.detach().cpu().numpy().reshape(n, n)
        lim = max(abs(np.nanpercentile(z, 1)), abs(np.nanpercentile(z, 99)), 1e-12)
        image = ax.contourf(xx.detach().cpu(), yy.detach().cpu(), z, 41,
                            cmap="coolwarm", vmin=-lim, vmax=lim, extend="both")
        fig.colorbar(image, ax=ax, shrink=.83)
        ax.set(xlabel="$x/L$", ylabel="$y/L$", title=title, aspect="equal")
    fig.suptitle(f"DeepPlasma cavity PINN, Re={module.Re:g} — unvalidated field")
    fig.savefig(output / "field_contours.png", dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "research"), required=True)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    defaults = {"smoke": (100.0, 2048, 3), "research": (5000.0, 262143, 10000)}
    reynolds, points, steps = defaults[args.mode]
    reynolds = args.re if args.re is not None else reynolds
    points = args.points if args.points is not None else points
    steps = args.steps if args.steps is not None else steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; this job must not run on a login/CPU node")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Unexpected process-wide default dtype before upstream import")
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(args.output)
    module, digest = load_upstream(args.source.resolve())
    if module.device.type != "cuda" or torch.get_default_dtype() != torch.float64:
        raise RuntimeError("Pinned code did not activate CUDA float64")
    module.Re = float(reynolds)
    module.SOAP_STEPS = 0
    module.USE_SSBROYDEN2 = True
    module.SSB_FIXED_PDE_N = int(points)
    module.TEST_PDE_N = min(int(points), 32768)
    module.SSB_STEPS = int(steps)
    module.TEST_EVERY = max(1, steps)
    module.RECORD_EVERY = max(1, min(100, steps))
    torch.manual_seed(module.SEED)
    np.random.seed(module.SEED)
    model = module.PINN().to(module.device)
    started = time.time()
    module.train_pinn(model, module.device, save_prefix=f"{args.mode}_")
    elapsed = time.time() - started
    generator = torch.Generator(device="cpu").manual_seed(78321)
    audit_points = torch.rand((min(8192, points), 2), generator=generator,
                              dtype=torch.float64).to(module.device)
    rx, ry, div = raw_residuals(module, model, audit_points)
    audit = {
        "claim_status": "workflow-only" if args.mode == "smoke" else "pending-reference-validation",
        "mode": args.mode, "reynolds_number": reynolds,
        "collocation_points": points, "optimizer_steps": steps,
        "seed": module.SEED, "elapsed_seconds": elapsed,
        "raw_residual_rms": {"momentum_x": rms(rx), "momentum_y": rms(ry), "continuity": rms(div)},
        "wall_error": wall_audit(module, model, module.device),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "precision": str(next(model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "python": platform.python_version(),
        "upstream_repository": "https://github.com/cmcdevitt2/DeepPlasma",
        "upstream_commit": UPSTREAM_COMMIT, "upstream_sha256": digest,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    render(module, model, module.device, Path.cwd())
    torch.save({"model": model.state_dict(), "audit": audit}, "model/audited_final.pt")
    Path("audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
