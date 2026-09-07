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


def predict_grid(module, model, device, n=181):
    """Evaluate velocity and gauge pressure on a Cartesian plotting grid."""
    axis = torch.linspace(0.0, 1.0, n, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((xx.ravel(), yy.ravel()), 1).requires_grad_(True)
    u, v, p = module.output_transform_cavity_flow(xy, model(xy))
    shape = (n, n)
    return (axis.detach().cpu().numpy(),
            u.detach().cpu().numpy().reshape(shape),
            v.detach().cpu().numpy().reshape(shape),
            p.detach().cpu().numpy().reshape(shape))


def render_qualified_validation(module, model, device, reference_path: Path,
                                output: Path, audit: dict, n=181):
    """Render the qualified PINN beside its near-matched CFD reference."""
    data = np.load(reference_path)
    matches = np.flatnonzero(np.isclose(data["Re"], module.Re))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one CFD case at Re={module.Re:g}, found {len(matches)}")
    case = int(matches[0])
    axis, up, vp, _ = predict_grid(module, model, device, n=n)
    xx, yy = np.meshgrid(axis, axis)
    ref_x, ref_y = data["x"], data["y"]
    query = np.column_stack((yy.ravel(), xx.ravel()))
    ur = RegularGridInterpolator((ref_y, ref_x), data["u"][case])(query).reshape(n, n)
    vr = RegularGridInterpolator((ref_y, ref_x), data["v"][case])(query).reshape(n, n)
    speed = np.hypot(up, vp)
    error = np.hypot(up - ur, vp - vr)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "figure.titlesize": 12})
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.15), constrained_layout=True)
    for ax, field, title in zip(axes[0, :2], (up, vp),
                                (r"(a) PINN $u/U$", r"(b) PINN $v/U$")):
        lim = max(abs(np.nanpercentile(field, 1)), abs(np.nanpercentile(field, 99)), 1e-12)
        im = ax.contourf(xx, yy, field, 41, cmap="RdBu_r", vmin=-lim, vmax=lim,
                         extend="both")
        fig.colorbar(im, ax=ax, shrink=.84, pad=.02)
        ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", title=title, aspect="equal")
    ax = axes[0, 2]
    im = ax.contourf(xx, yy, speed, 41, cmap="viridis", extend="max")
    ax.streamplot(axis, axis, up, vp, color="white", density=.75,
                  linewidth=.45, arrowsize=.55)
    fig.colorbar(im, ax=ax, shrink=.84, pad=.02)
    ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", title=r"(c) PINN $|\mathbf{u}|/U$", aspect="equal")

    ax = axes[1, 0]
    vmax = max(float(np.nanpercentile(error, 99)), 1e-12)
    im = ax.contourf(xx, yy, error, 41, cmap="magma", vmin=0, vmax=vmax, extend="max")
    fig.colorbar(im, ax=ax, shrink=.84, pad=.02)
    ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$",
           title=r"(d) $|\mathbf{u}_{PINN}-\mathbf{u}_{CFD}|/U$", aspect="equal")

    mid_x = int(np.argmin(abs(ref_x - .5)))
    mid_y = int(np.argmin(abs(ref_y - .5)))
    model_mid = int(np.argmin(abs(axis - .5)))
    ax = axes[1, 1]
    ax.plot(data["u"][case, :, mid_x], ref_y, color="black", lw=1.8, label="CFD")
    ax.plot(up[:, model_mid], axis, color="#0072B2", lw=1.7, ls="--", label="PINN")
    ax.axvline(0, color="0.75", lw=.7)
    ax.set(xlabel=r"$u/U$", ylabel=r"$y/L$", title=r"(e) $u(0.5,y)$", ylim=(0, 1))
    ax.grid(alpha=.2); ax.legend(frameon=False)
    ax = axes[1, 2]
    ax.plot(ref_x, data["v"][case, mid_y, :], color="black", lw=1.8, label="CFD")
    ax.plot(axis, vp[model_mid, :], color="#D55E00", lw=1.7, ls="--", label="PINN")
    ax.axhline(0, color="0.75", lw=.7)
    ax.set(xlabel=r"$x/L$", ylabel=r"$v/U$", title=r"(f) $v(x,0.5)$", xlim=(0, 1))
    ax.grid(alpha=.2); ax.legend(frameon=False)

    metrics = audit["cfd_comparison"]["metrics"]
    fig.suptitle(
        f"Qualified cavity PINN at Re={module.Re:g} — near-matched CFD validation\n"
        f"centerline errors: u {100*metrics['u_centerline_relative_l2']:.2f}%, "
        f"v {100*metrics['v_centerline_relative_l2']:.2f}%; "
        f"interior velocity {100*metrics['interior_velocity_relative_l2']:.2f}%",
        fontweight="semibold")
    fig.savefig(output / "qualified_validation.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "qualified_validation.pdf", bbox_inches="tight")
    plt.close(fig)


def render_convergence(history_path: Path, output: Path):
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    latest = {int(row["completed_step"]): row for row in records}
    steps = np.array(sorted(latest))
    mx = np.sqrt([latest[int(step)]["masked_momentum_x_mse"] for step in steps])
    my = np.sqrt([latest[int(step)]["masked_momentum_y_mse"] for step in steps])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.semilogy(steps, mx, lw=1.4, color="#0072B2", label=r"masked $r_x$ RMS")
    ax.semilogy(steps, my, lw=1.4, color="#D55E00", label=r"masked $r_y$ RMS")
    if steps.min() <= 300 <= steps.max():
        ax.axvline(300, color="0.35", ls=":", lw=1.1)
        ax.text(306, ax.get_ylim()[1] / 1.8, "initial 300-step gate", fontsize=8,
                color="0.3", va="top")
    ax.set(xlabel="completed SSBroyden2 step", ylabel="collocation residual RMS",
           title="Continuation from the retained optimizer checkpoint")
    ax.grid(which="both", alpha=.22); ax.legend(frameon=False)
    fig.savefig(output / "optimizer_convergence.png", dpi=260, bbox_inches="tight")
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
              "seed": module.SEED,
              "upstream_commit": UPSTREAM_COMMIT}
    optimizer = module.SSBroyden2(
        model.parameters(), lr=module.SSB_LR, gtol=1e-12,
        line_search=module.SSB_LINE_SEARCH, c1=module.SSB_C1, c2=module.SSB_C2,
        wolfe_maxiter=module.SSB_LS_MAXITER, zoom_maxiter=module.SSB_ZOOM_MAXITER,
        amax=module.SSB_AMAX, dtype=torch.float64, device=module.device,
    )
    start = 0
    if resume and checkpoint.exists():
        # Keep serialized CPU RNG bytes on CPU. Move only mathematical GPU state
        # explicitly below; a blanket CUDA map breaks torch.set_rng_state().
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_config = dict(saved["config"])
        # Checkpoints made before extension support stored the former target step
        # count. It is not part of the mathematical state and may be increased.
        saved_config.pop("optimizer_steps", None)
        if saved_config != config:
            raise RuntimeError(f"Refusing incompatible checkpoint: {saved_config} != {config}")
        model.load_state_dict(saved["model"])
        for key in ("H", "x", "k"):
            value = saved["optimizer"][key]
            optimizer.state[key] = value.to(module.device) if torch.is_tensor(value) else value
        collocation = saved["collocation_points"].to(module.device)
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        start = int(saved["completed_steps"])
        if steps < start:
            raise RuntimeError(f"Requested target {steps} precedes checkpoint step {start}")
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
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate evidence figures from the retained checkpoint")
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
    if args.render_only:
        checkpoint = Path.cwd() / "checkpoint.pt"
        audit_path = Path.cwd() / "audit.json"
        if not checkpoint.is_file() or not audit_path.is_file() or not args.reference_npz:
            raise RuntimeError("Render-only mode requires checkpoint, audit and CFD reference")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        model.to(module.device)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("claim_status") != "qualified-near-matched-reference":
            raise RuntimeError("Refusing a qualified figure for an unqualified audit")
        render_qualified_validation(module, model, module.device,
                                    args.reference_npz.resolve(), Path.cwd(), audit)
        render_convergence(Path.cwd() / "optimizer-history.jsonl", Path.cwd())
        return 0
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
    if audit["claim_status"] == "qualified-near-matched-reference":
        render_qualified_validation(module, model, module.device,
                                    args.reference_npz.resolve(), Path.cwd(), audit)
        render_convergence(Path.cwd() / "optimizer-history.jsonl", Path.cwd())
    else:
        render(module, model, module.device, Path.cwd())
    Path("model").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "audit": audit}, "model/audited_final.pt")
    Path("audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if args.mode == "qualification" and not audit.get("cfd_comparison", {}).get("all_pass", False):
        return 2


if __name__ == "__main__":
    raise SystemExit(main() or 0)
