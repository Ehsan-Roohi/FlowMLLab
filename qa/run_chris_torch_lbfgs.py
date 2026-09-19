"""GPU-native L-BFGS refinement for McDevitt's deep-cavity PINN.

The hard streamfunction boundary transform and two momentum residuals follow
the public DeepPlasma LDC formulation.  A retained TensorFlow checkpoint is
converted layer-by-layer, then refined for one declared (Re, D/W, taper) case
with deterministic collocation points.  PyTorch parameters, gradients and
L-BFGS history remain on the GPU; no SciPy optimizer is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def checkpoint_arrays(path: Path, widths: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    result = []
    for index, (fan_in, fan_out) in enumerate(zip(widths[:-1], widths[1:])):
        kernel, bias = data[f"kernel_{index}"], data[f"bias_{index}"]
        if kernel.shape != (fan_in, fan_out) or bias.shape != (fan_out,):
            raise ValueError(f"Layer {index} has incompatible exported shapes")
        result.append((kernel, bias))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--checkpoint-label", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--depth", type=float, default=5.0)
    ap.add_argument("--taper", type=float, default=0.0)
    ap.add_argument("--points", type=int, default=25000)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--history-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda")
    widths = [5] + [args.width] * 6 + [3]

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(a, b, dtype=torch.float64) for a, b in zip(widths[:-1], widths[1:])
            )

        def forward(self, z):
            for layer in self.layers[:-1]:
                z = torch.tanh(layer(z))
            return self.layers[-1](z)

    net = Net().to(device)
    arrays = checkpoint_arrays(args.weights, widths)
    with torch.no_grad():
        for layer, (kernel, bias) in zip(net.layers, arrays):
            layer.weight.copy_(torch.as_tensor(kernel.T, device=device))
            layer.bias.copy_(torch.as_tensor(bias, device=device))

    rng = np.random.default_rng(args.seed)
    count = 2048 if args.preflight else args.points
    xy = rng.random((count, 2))
    # One physical case: midpoint of the adapted Re and depth boxes; zero taper.
    fixed = np.column_stack((xy, np.full(count, .5), np.full(count, .5), np.zeros(count)))
    z = torch.tensor(fixed, device=device, requires_grad=True)
    dx2, dy2 = 1e-3, 1e-2

    def grad(value, coordinate):
        return torch.autograd.grad(value, z, torch.ones_like(value), create_graph=True,
                                   retain_graph=True)[0][:, coordinate:coordinate + 1]

    def transformed():
        raw = net(z)
        x, eta = z[:, 0:1], z[:, 1:2]
        depth = torch.full_like(x, args.depth)
        taper = torch.full_like(x, args.taper)
        left = x - taper * (1 - eta)
        right = 1 - x - taper * (1 - eta)
        bcv = 16 * left * right * eta * (1 - eta)
        exp_l = torch.exp(-left.square() / dx2)
        exp_r = torch.exp(-right.square() / dx2)
        exp_b = torch.exp(-(1 - eta).square() / dy2)
        psi_lid_x = (-depth * (eta - 1) * eta.square() * 2 * right / dx2 * exp_r *
                     (1 - exp_l) * exp_b + depth * (eta - 1) * eta.square() *
                     (1 - exp_r) * 2 * left / dx2 * exp_l * exp_b)
        psi_lid_eta = (depth * (eta.square() + 2 * eta * (eta - 1)) * (1 - exp_r) *
                       (1 - exp_l) * exp_b + depth * (eta - 1) * eta.square() *
                       2 * taper * right / dx2 * exp_r * (1 - exp_l) * exp_b +
                       depth * (eta - 1) * eta.square() * (1 - exp_r) *
                       2 * taper * left / dx2 * exp_l * exp_b +
                       depth * (eta - 1) * eta.square() * (1 - exp_r) *
                       (1 - exp_l) * 2 * (1 - eta) / dy2 * exp_b)
        dbcv_x = 16 * (1 - 2 * x) * eta * (1 - eta)
        dbcv_eta = (16 * left * right * (1 - 2 * eta) +
                    16 * taper * (1 - 2 * taper * (1 - eta)) * eta * (1 - eta))
        phi_x, phi_eta = grad(raw[:, 0:1], 0), grad(raw[:, 0:1], 1)
        u = psi_lid_eta / depth + 2 * bcv * dbcv_eta * raw[:, 0:1] / depth + bcv.square() * phi_eta / depth
        v = -(psi_lid_x + 2 * bcv * dbcv_x * raw[:, 0:1] + bcv.square() * phi_x)
        return u, v, raw[:, 2:3]

    def objective():
        u, v, p = transformed()
        ux, uy, vx, vy = grad(u, 0), grad(u, 1), grad(v, 0), grad(v, 1)
        uxx, uyy, vxx, vyy = grad(ux, 0), grad(uy, 1), grad(vx, 0), grad(vy, 1)
        px, py = grad(p, 0), grad(p, 1)
        x, eta = z[:, 0:1], z[:, 1:2]
        env_l = .5 * (1 + torch.tanh((x - args.taper * (1 - eta)) / np.sqrt(dx2)))
        env_r = .5 * (1 + torch.tanh((1 - x - args.taper * (1 - eta)) / np.sqrt(dx2)))
        r1 = env_l * env_r * (u * ux + v / args.depth * uy - (uxx + uyy / args.depth**2) / args.re + px)
        r2 = env_l * env_r * (u * vx + v / args.depth * vy - (vxx + vyy / args.depth**2) / args.re + py / args.depth)
        return r1.square().mean() + r2.square().mean()

    initial = float(objective().detach().cpu())
    optimizer = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=1,
                                  history_size=args.history_size,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-12, tolerance_change=1e-15)
    rows = []
    steps = 3 if args.preflight else args.iterations
    for step in range(1, steps + 1):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            loss.backward()
            return loss
        value = float(optimizer.step(closure).detach().cpu())
        if step == 1 or step % 10 == 0 or step == steps:
            current = float(objective().detach().cpu())
            rows.append((step, current))
            print(f"step={step} loss={current:.12e} allocated_GiB={torch.cuda.memory_allocated()/2**30:.3f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "widths": widths}, args.output / "torch_lbfgs.pt")
    np.savetxt(args.output / "loss.dat", np.asarray(rows), header="iteration loss")
    report = {"optimizer": "torch.optim.LBFGS", "device": torch.cuda.get_device_name(),
              "dtype": "float64", "points": count, "iterations": steps,
              "initial_loss": initial, "final_loss": rows[-1][1],
              "case": {"Re": args.re, "depth_over_width": args.depth, "taper": args.taper},
              "checkpoint": args.checkpoint_label, "weights": str(args.weights),
              "architecture": widths}
    (args.output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
