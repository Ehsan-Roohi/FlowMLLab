"""Restartable rectangular-cavity PINN experiment for the Week 13 research lab.

The network and SSBroyden2 implementation are loaded from the pinned DeepPlasma
source.  FlowMLLab supplies a rectangular-coordinate lifting, an Adam warm-up,
independent residual audits, and case-specific validation.  Coordinates are
``(x, eta)`` in the unit square and physical ``y = aspect_ratio * eta``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

from run_week42_deepplasma import UPSTREAM_COMMIT, load_upstream


STOP_REQUESTED = False


def request_checkpoint(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def fields(module, model, xy, aspect):
    """Return physical velocity, gauge pressure and streamfunction."""
    raw = model(xy)
    x, eta = xy[:, :1], xy[:, 1:2]
    corner = ((1 - torch.exp(-(x - 1) ** 2 / module.dx**2))
              * (1 - torch.exp(-x**2 / module.dx**2)))
    top = torch.exp(-(1 - eta) ** 2 / module.dy**2)
    psi_lid = aspect * (eta - 1) * eta**2 * corner * top
    bc = 16 * x * (1 - x) * eta * (1 - eta)
    psi = psi_lid + bc**2 * raw[:, :1]
    grad = torch.autograd.grad(psi, xy, torch.ones_like(psi), create_graph=True)[0]
    u = grad[:, 1:2] / aspect
    v = -grad[:, 0:1]
    return u, v, raw[:, 1:2], psi


def residuals(module, model, points, reynolds, aspect, masked=True):
    xy = points.clone().detach().requires_grad_(True)
    u, v, p, _ = fields(module, model, xy, aspect)
    gu = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, xy, torch.ones_like(v), create_graph=True)[0]
    gp = torch.autograd.grad(p, xy, torch.ones_like(p), create_graph=True)[0]
    uxx = torch.autograd.grad(gu[:, :1], xy, torch.ones_like(u), create_graph=True)[0][:, :1]
    uee = torch.autograd.grad(gu[:, 1:2], xy, torch.ones_like(u), create_graph=True)[0][:, 1:2]
    vxx = torch.autograd.grad(gv[:, :1], xy, torch.ones_like(v), create_graph=True)[0][:, :1]
    vee = torch.autograd.grad(gv[:, 1:2], xy, torch.ones_like(v), create_graph=True)[0][:, 1:2]
    rx = u * gu[:, :1] + (v / aspect) * gu[:, 1:2]
    rx = rx - (uxx + uee / aspect**2) / reynolds + gp[:, :1]
    ry = u * gv[:, :1] + (v / aspect) * gv[:, 1:2]
    ry = ry - (vxx + vee / aspect**2) / reynolds + gp[:, 1:2] / aspect
    div = gu[:, :1] + gv[:, 1:2] / aspect
    if masked:
        x, eta = xy[:, :1], xy[:, 1:2]
        radius = 1e-2
        mask = ((1 - torch.exp(-(x**2 + (1 - eta)**2) / radius**2))
                * (1 - torch.exp(-((1 - x)**2 + (1 - eta)**2) / radius**2)))
        rx, ry = mask * rx, mask * ry
    return rx, ry, div


def sample(module, n, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand((n, 2), generator=generator, dtype=torch.float64).to(module.device)


def mse_pair(module, model, points, reynolds, aspect):
    rx, ry, _ = residuals(module, model, points, reynolds, aspect, masked=True)
    return torch.mean(rx**2), torch.mean(ry**2)


def save_checkpoint(path, model, phase, adam_done, ssb_done, optimizer, train_points, config):
    payload = {"model": model.state_dict(), "phase": phase,
               "adam_done": adam_done, "ssb_done": ssb_done,
               "optimizer": optimizer.state_dict(), "train_points": train_points,
               "config": config, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": np.random.get_state()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_history(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row) + "\n")


def residual_summary(module, model, points, reynolds, aspect, masked):
    rx, ry, div = residuals(module, model, points, reynolds, aspect, masked=masked)
    corner = (points[:, 1] > .9) & ((points[:, 0] < .1) | (points[:, 0] > .9))
    rms = lambda z: float(torch.sqrt(torch.mean(z.detach()**2)).cpu())
    return {"momentum_x_rms": rms(rx), "momentum_y_rms": rms(ry),
            "continuity_rms": rms(div),
            "top_corner_momentum_rms": rms(torch.cat((rx[corner], ry[corner])))}


def record_history(path, phase, phase_step, global_step, l1, l2, module, model,
                   test_points, reynolds, aspect):
    test = residual_summary(module, model, test_points, reynolds, aspect, masked=False)
    append_history(path, {"phase": phase, "phase_step": phase_step,
        "global_step": global_step, "train_rx_mse": float(l1.detach().cpu()),
        "train_ry_mse": float(l2.detach().cpu()),
        "heldout_momentum_rms": float(np.hypot(test["momentum_x_rms"],
                                                test["momentum_y_rms"])),
        "heldout_continuity_rms": test["continuity_rms"],
        "top_corner_momentum_rms": test["top_corner_momentum_rms"]})


def train(module, model, output, reynolds, aspect, points_n, adam_steps, ssb_steps,
          checkpoint_every, resume):
    config = {"reynolds_number": reynolds, "aspect_ratio": aspect,
              "collocation_points": points_n, "adam_steps": adam_steps,
              "ssb_steps": ssb_steps, "seed": module.SEED}
    checkpoint = output / "checkpoint.pt"
    history = output / "optimizer-history.jsonl"
    train_points = sample(module, points_n, module.SEED + 11)
    test_points = sample(module, min(4096, max(1024, points_n // 4)), module.SEED + 29)
    adam_done = ssb_done = 0
    phase = "adam"
    saved = None
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("Refusing an incompatible checkpoint")
        model.load_state_dict(saved["model"])
        train_points = saved["train_points"].to(module.device)
        adam_done, ssb_done, phase = saved["adam_done"], saved["ssb_done"], saved["phase"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        np.random.set_state(saved["numpy_rng"])
        print(f"Resumed {phase}: Adam={adam_done}, SSBroyden2={ssb_done}", flush=True)

    if phase == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        if saved is not None:
            optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(module.device)
        for step in range(adam_done, adam_steps):
            optimizer.zero_grad(set_to_none=True)
            l1, l2 = mse_pair(module, model, train_points, reynolds, aspect)
            loss = l1 + l2
            loss.backward(); optimizer.step()
            adam_done = step + 1
            if adam_done == 1 or adam_done % 100 == 0 or adam_done == adam_steps:
                record_history(history, "Adam", adam_done, adam_done, l1, l2,
                               module, model, test_points, reynolds, aspect)
            if adam_done % checkpoint_every == 0 or adam_done == adam_steps or STOP_REQUESTED:
                next_phase = "ssbroyden2" if adam_done == adam_steps else "adam"
                save_checkpoint(checkpoint, model, next_phase, adam_done, 0,
                                optimizer, train_points, config)
            if STOP_REQUESTED:
                return False
        phase, saved = "ssbroyden2", None
        endpoint = independent_audit(module, model, reynolds, aspect, n=4096)
        (output / "adam-endpoint.json").write_text(
            json.dumps(endpoint, indent=2) + "\n", encoding="utf-8")

    optimizer = module.SSBroyden2(model.parameters(), lr=module.SSB_LR, gtol=1e-12,
        line_search=module.SSB_LINE_SEARCH, c1=module.SSB_C1, c2=module.SSB_C2,
        wolfe_maxiter=module.SSB_LS_MAXITER, zoom_maxiter=module.SSB_ZOOM_MAXITER,
        amax=module.SSB_AMAX, dtype=torch.float64, device=module.device)
    if resume and checkpoint.is_file() and ssb_done:
        optimizer.load_state_dict(saved["optimizer"] if saved is not None else
                                  torch.load(checkpoint, map_location="cpu", weights_only=False)["optimizer"])
        for key in ("H", "x"):
            optimizer.state[key] = optimizer.state[key].to(module.device)
    for step in range(ssb_done, ssb_steps):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            l1c, l2c = mse_pair(module, model, train_points, reynolds, aspect)
            total = l1c + l2c
            total.backward()
            return total
        optimizer.step(closure)
        l1, l2 = mse_pair(module, model, train_points, reynolds, aspect)
        ssb_done = step + 1
        if ssb_done == 1 or ssb_done % 25 == 0 or ssb_done == ssb_steps:
            record_history(history, "SSBroyden2", ssb_done, adam_steps + ssb_done,
                           l1, l2, module, model, test_points, reynolds, aspect)
        if ssb_done % checkpoint_every == 0 or ssb_done == ssb_steps or STOP_REQUESTED:
            save_checkpoint(checkpoint, model, "ssbroyden2", adam_done, ssb_done,
                            optimizer, train_points, config)
        if STOP_REQUESTED:
            return False
    return True


def independent_audit(module, model, reynolds, aspect, n=8192):
    points = sample(module, n, 90401 + int(reynolds) + int(100 * aspect))
    rx, ry, div = residuals(module, model, points, reynolds, aspect, masked=False)
    corner = (points[:, 1] > .9) & ((points[:, 0] < .1) | (points[:, 0] > .9))
    rms = lambda z: float(torch.sqrt(torch.mean(z.detach()**2)).cpu())
    return {"momentum_x_rms": rms(rx), "momentum_y_rms": rms(ry),
            "continuity_rms": rms(div), "top_corner_momentum_x_rms": rms(rx[corner]),
            "top_corner_momentum_y_rms": rms(ry[corner]), "points": n,
            "corner_points": int(corner.sum().cpu())}


def wall_audit(module, model, aspect, n=1001):
    axis = torch.linspace(0.0, 1.0, n, device=module.device, dtype=torch.float64)
    walls = {"bottom": torch.stack((axis, torch.zeros_like(axis)), 1),
             "left": torch.stack((torch.zeros_like(axis), axis), 1),
             "right": torch.stack((torch.ones_like(axis), axis), 1),
             "top": torch.stack((axis, torch.ones_like(axis)), 1)}
    result = {}
    for name, points in walls.items():
        points.requires_grad_(True)
        u, v, _, _ = fields(module, model, points, aspect)
        target_u = torch.zeros_like(u)
        if name == "top":
            x = points[:, :1]
            target_u = ((1 - torch.exp(-(x - 1) ** 2 / module.dx**2))
                        * (1 - torch.exp(-x**2 / module.dx**2)))
        result[name] = {"u_max_abs_error": float(torch.max(abs(u - target_u)).detach().cpu()),
                        "v_max_abs_error": float(torch.max(abs(v)).detach().cpu())}
    return result


def square_cfd_metrics(module, model, reference, reynolds):
    data = np.load(reference)
    matches = np.flatnonzero(np.isclose(data["Re"], reynolds))
    if len(matches) != 1:
        return None
    case = int(matches[0]); x, y = data["x"], data["y"]
    rng = np.random.default_rng(92831)
    xy = rng.uniform(0.05, 0.95, size=(8192, 2))
    points = torch.as_tensor(xy, device=module.device, dtype=torch.float64).requires_grad_(True)
    u, v, _, _ = fields(module, model, points, 1.0)
    pred = np.column_stack((u.detach().cpu().numpy().ravel(), v.detach().cpu().numpy().ravel()))
    query_yx = xy[:, ::-1]
    truth = np.column_stack((RegularGridInterpolator((y, x), data["u"][case])(query_yx),
                             RegularGridInterpolator((y, x), data["v"][case])(query_yx)))
    vector = float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
    xline = torch.as_tensor(x, device=module.device, dtype=torch.float64).reshape(-1, 1)
    yline = torch.as_tensor(y, device=module.device, dtype=torch.float64).reshape(-1, 1)
    vertical = torch.cat((torch.full_like(yline, .5), yline), 1).requires_grad_(True)
    horizontal = torch.cat((xline, torch.full_like(xline, .5)), 1).requires_grad_(True)
    up, _, _, _ = fields(module, model, vertical, 1.0)
    _, vp, _, _ = fields(module, model, horizontal, 1.0)
    relative = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    metrics = {"u_centerline_relative_l2": relative(up.detach().cpu().numpy().ravel(),
                                                      data["u"][case, :, np.argmin(abs(x - .5))]),
               "v_centerline_relative_l2": relative(vp.detach().cpu().numpy().ravel(),
                                                      data["v"][case, np.argmin(abs(y - .5)), :]),
               "interior_velocity_relative_l2": vector}
    thresholds = {"u_centerline_relative_l2": .10,
                  "v_centerline_relative_l2": .15,
                  "interior_velocity_relative_l2": .15}
    passes = {key: metrics[key] <= value for key, value in thresholds.items()}
    return {"reference_case_index": case, "metrics": metrics,
            "frozen_thresholds": thresholds, "passes": passes,
            "all_pass": all(passes.values()),
            "note": "Smooth-lid PINN versus classical-lid CFD; near-matched comparison."}


def render(module, model, output, reynolds, aspect, audit):
    nx, ne = 181, max(181, int(round(181 * aspect)))
    x = np.linspace(0, 1, nx); eta = np.linspace(0, 1, ne)
    xx, ee = np.meshgrid(x, eta)
    xy = torch.as_tensor(np.column_stack((xx.ravel(), ee.ravel())),
                         device=module.device).requires_grad_(True)
    u, v, _, psi = fields(module, model, xy, aspect)
    u = u.detach().cpu().numpy().reshape(ne, nx)
    v = v.detach().cpu().numpy().reshape(ne, nx)
    psi = psi.detach().cpu().numpy().reshape(ne, nx)
    y = eta * aspect; speed = np.hypot(u, v)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
    for ax, z, title, cmap in zip(axes, (speed, psi, np.hypot(u, v)),
                                  (r"$|\mathbf{u}|/U$", r"$\psi/(UL)$", "streamlines"),
                                  ("viridis", "RdBu_r", "viridis")):
        im = ax.contourf(x, y, z, 41, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=.82, pad=.02)
        if title == "streamlines":
            ax.streamplot(x, y, u, v, color="white", density=1.2, linewidth=.45)
        ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", title=title, aspect="equal")
    fig.suptitle(f"Rectangular-cavity PINN: Re={reynolds:g}, D={aspect:g} | independent residual audit")
    fig.savefig(output / "fields.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    raw_rows = [json.loads(line) for line in (output / "optimizer-history.jsonl").read_text().splitlines() if line]
    rows = list({int(row["global_step"]): row for row in raw_rows}.values())
    rows.sort(key=lambda row: row["global_step"])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for phase, color in (("Adam", "#0072B2"), ("SSBroyden2", "#D55E00")):
        selected = [r for r in rows if r["phase"] == phase]
        if selected:
            ax.semilogy([r["global_step"] for r in selected],
                        [np.sqrt(r["train_rx_mse"] + r["train_ry_mse"]) for r in selected],
                        color=color, lw=1.5, label=phase)
    ax.semilogy([r["global_step"] for r in rows],
                [r["heldout_momentum_rms"] for r in rows], color="black", lw=1.35,
                ls="--", label="held-out full residual")
    ax.semilogy([r["global_step"] for r in rows],
                [r["top_corner_momentum_rms"] for r in rows], color="#009E73", lw=1.15,
                ls=":", label="held-out top corners")
    ax.set(xlabel="optimizer step", ylabel="masked momentum residual RMS",
           title=f"Optimizer continuation, Re={reynolds:g}, D={aspect:g}")
    ax.grid(which="both", alpha=.22); ax.legend(frameon=False)
    fig.savefig(output / "loss.png", dpi=240, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path)
    parser.add_argument("--re", type=float, required=True)
    parser.add_argument("--aspect-ratio", type=float, required=True)
    parser.add_argument("--points", type=int, default=16384)
    parser.add_argument("--adam-steps", type=int, default=1000)
    parser.add_argument("--ssb-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.aspect_ratio <= 0:
        raise ValueError("aspect ratio must be positive")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    os.chdir(args.output)
    module, digest = load_upstream(args.source.resolve())
    torch.manual_seed(module.SEED); np.random.seed(module.SEED)
    model = module.PINN().to(module.device)
    signal.signal(signal.SIGUSR1, request_checkpoint)
    signal.signal(signal.SIGTERM, request_checkpoint)
    started = time.time()
    complete = train(module, model, Path.cwd(), args.re, args.aspect_ratio, args.points,
                     args.adam_steps, args.ssb_steps, args.checkpoint_every, args.resume)
    if not complete:
        return 99
    audit = {"claim_status": "residual-qualified-only", "reynolds_number": args.re,
             "aspect_ratio_depth_over_width": args.aspect_ratio,
             "optimizers": ["Adam", "SSBroyden2"], "adam_steps": args.adam_steps,
             "ssbroyden2_steps": args.ssb_steps, "collocation_points": args.points,
             "precision": str(next(model.parameters()).dtype),
             "gpu": torch.cuda.get_device_name(0), "elapsed_seconds": time.time() - started,
             "upstream_commit": UPSTREAM_COMMIT,
             "upstream_sha256": digest, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
             "independent_residual": independent_audit(module, model, args.re, args.aspect_ratio),
             "wall_error": wall_audit(module, model, args.aspect_ratio)}
    if args.reference_npz and np.isclose(args.aspect_ratio, 1.0):
        comparison = square_cfd_metrics(module, model, args.reference_npz.resolve(), args.re)
        if comparison is not None:
            audit["cfd_comparison"] = comparison
            audit["claim_status"] = "field-compared-square-case"
    Path("audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    render(module, model, Path.cwd(), args.re, args.aspect_ratio, audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
