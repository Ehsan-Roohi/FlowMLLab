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
import signal
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

UPSTREAM_COMMIT = "fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b"
UPSTREAM_SHA256 = "391a2174cb9f6e8863c14d7b350077e54209134de9f85a17719c398146f91458"
STOP_REQUESTED = False


def request_checkpoint(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after the current optimizer step", flush=True)


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


def compare_cfd(module, model, device, reference_path: Path):
    """Compare against the retained conventional-CFD field at the same Re."""
    data = np.load(reference_path)
    matches = np.flatnonzero(np.isclose(data["Re"], module.Re))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CFD case at Re={module.Re:g}, found {len(matches)}")
    case = int(matches[0])
    axis_x, axis_y = data["x"], data["y"]
    xline = torch.as_tensor(axis_x, device=device).reshape(-1, 1)
    yline = torch.as_tensor(axis_y, device=device).reshape(-1, 1)
    vertical = torch.cat((torch.full_like(yline, 0.5), yline), 1).requires_grad_(True)
    horizontal = torch.cat((xline, torch.full_like(xline, 0.5)), 1).requires_grad_(True)
    up, _, _ = module.output_transform_cavity_flow(vertical, model(vertical))
    _, vp, _ = module.output_transform_cavity_flow(horizontal, model(horizontal))
    u_ref = data["u"][case, :, np.argmin(abs(axis_x - 0.5))]
    v_ref = data["v"][case, np.argmin(abs(axis_y - 0.5)), :]
    up = up.detach().cpu().numpy().ravel()
    vp = vp.detach().cpu().numpy().ravel()
    relative = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    rng = np.random.default_rng(92831)
    xy = rng.uniform(0.05, 0.95, size=(8192, 2))
    xy_t = torch.as_tensor(xy, device=device, dtype=torch.float64).requires_grad_(True)
    ui, vi, _ = module.output_transform_cavity_flow(xy_t, model(xy_t))
    query_yx = xy[:, ::-1]
    ui_ref = RegularGridInterpolator((axis_y, axis_x), data["u"][case])(query_yx)
    vi_ref = RegularGridInterpolator((axis_y, axis_x), data["v"][case])(query_yx)
    pred = np.column_stack((ui.detach().cpu().numpy().ravel(), vi.detach().cpu().numpy().ravel()))
    truth = np.column_stack((ui_ref, vi_ref))
    values = {
        "u_centerline_relative_l2": relative(up, u_ref),
        "v_centerline_relative_l2": relative(vp, v_ref),
        "interior_velocity_relative_l2": relative(pred, truth),
    }
    thresholds = {"u_centerline_relative_l2": 0.10,
                  "v_centerline_relative_l2": 0.15,
                  "interior_velocity_relative_l2": 0.15}
    passes = {key: values[key] <= thresholds[key] for key in thresholds}
    return {"reference": str(reference_path), "reference_case_index": case,
            "note": "PINN smooth lid versus classical-lid CFD: near-matched comparison.",
            "metrics": values, "frozen_thresholds": thresholds,
            "passes": passes, "all_pass": all(passes.values())}


def atomic_checkpoint(path: Path, module, model, optimizer, points, completed_steps, config):
    payload = {
        "model": model.state_dict(),
        "optimizer": {"H": optimizer.state["H"], "x": optimizer.state["x"],
                      "k": optimizer.state["k"]},
        "collocation_points": points,
        "completed_steps": completed_steps,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_restartable(module, model, output, points_n, steps, checkpoint_every, resume):
    checkpoint = output / "checkpoint.pt"
    config = {"reynolds_number": module.Re, "collocation_points": points_n,
              "optimizer_steps": steps, "seed": module.SEED,
              "upstream_commit": UPSTREAM_COMMIT}
    optimizer = module.SSBroyden2(
        model.parameters(), lr=module.SSB_LR, gtol=1e-12,
        line_search=module.SSB_LINE_SEARCH, c1=module.SSB_C1, c2=module.SSB_C2,
        wolfe_maxiter=module.SSB_LS_MAXITER, zoom_maxiter=module.SSB_ZOOM_MAXITER,
        amax=module.SSB_AMAX, dtype=torch.float64, device=module.device,
    )
    start = 0
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=module.device, weights_only=False)
        if saved["config"] != config:
            raise RuntimeError(f"Refusing incompatible checkpoint: {saved['config']} != {config}")
        model.load_state_dict(saved["model"])
        for key in ("H", "x", "k"):
            optimizer.state[key] = saved["optimizer"][key]
        collocation = saved["collocation_points"].to(module.device)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        start = int(saved["completed_steps"])
        print(f"Resumed checkpoint at completed step {start}", flush=True)
    else:
        collocation = module.sample_pde_points(points_n)
    metrics = output / "optimizer-history.jsonl"
    for step in range(start, steps):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            r1, r2 = module.fp_pde(model, collocation)
            loss = torch.mean(r1**2) + torch.mean(r2**2)
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.enable_grad():
            r1, r2 = module.fp_pde(model, collocation)
            values = {"completed_step": step + 1,
                      "masked_momentum_x_mse": float(torch.mean(r1**2).detach().cpu()),
                      "masked_momentum_y_mse": float(torch.mean(r2**2).detach().cpu())}
        with metrics.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(values) + "\n")
        if (step + 1) % checkpoint_every == 0 or step + 1 == steps or STOP_REQUESTED:
            atomic_checkpoint(checkpoint, module, model, optimizer, collocation,
                              step + 1, config)
        if STOP_REQUESTED:
            return step + 1, False
    return steps, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "qualification", "research"), required=True)
    parser.add_argument("--re", type=float, default=None)
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-npz", type=Path)
    args = parser.parse_args()
    defaults = {"smoke": (100.0, 2048, 3),
                "qualification": (100.0, 16384, 300),
                "research": (5000.0, 262143, 10000)}
    reynolds, points, steps = defaults[args.mode]
    reynolds = args.re if args.re is not None else reynolds
    points = args.points if args.points is not None else points
    steps = args.steps if args.steps is not None else steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; this job must not run on a login/CPU node")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("Unexpected process-wide default dtype before upstream import")
    args.output.mkdir(parents=True, exist_ok=args.resume)
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
    signal.signal(signal.SIGUSR1, request_checkpoint)
    signal.signal(signal.SIGTERM, request_checkpoint)
    started = time.time()
    completed_steps, complete = train_restartable(
        module, model, Path.cwd(), points, steps, args.checkpoint_every, args.resume)
    elapsed = time.time() - started
    if not complete:
        print(f"Checkpoint complete at step {completed_steps}; requesting requeue", flush=True)
        return 99
    generator = torch.Generator(device="cpu").manual_seed(78321)
    audit_points = torch.rand((min(8192, points), 2), generator=generator,
                              dtype=torch.float64).to(module.device)
    rx, ry, div = raw_residuals(module, model, audit_points)
    audit = {
        "claim_status": "workflow-only" if args.mode == "smoke" else "pending-reference-validation",
        "mode": args.mode, "reynolds_number": reynolds,
        "collocation_points": points, "optimizer_steps": steps,
        "completed_steps": completed_steps, "restartable": True,
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
    if args.reference_npz:
        audit["cfd_comparison"] = compare_cfd(
            module, model, module.device, args.reference_npz.resolve())
        if args.mode == "qualification" and audit["cfd_comparison"]["all_pass"]:
            audit["claim_status"] = "qualified-near-matched-reference"
    render(module, model, module.device, Path.cwd())
    Path("model").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "audit": audit}, "model/audited_final.pt")
    Path("audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if args.mode == "qualification" and not audit.get("cfd_comparison", {}).get("all_pass", False):
        return 2


if __name__ == "__main__":
    raise SystemExit(main() or 0)
