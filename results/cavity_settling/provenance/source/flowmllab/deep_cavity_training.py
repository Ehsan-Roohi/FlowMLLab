"""Controlled, restartable optimizer experiment using hash-pinned author D5 code.

The author-supplied source bundle is in data/week13_d5/author_source with its
attribution and scope notice. This adapter verifies exact source hashes; the
published equations and Gaussian lid are not replaced by the earlier teaching
approximation. The inherited source is not implicitly relicensed by this adapter.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

AUTHOR_SHA256 = {
    "cavity.py": "29ff1f7ba495de25c088b4eefcd200202f1a2da020248d4b3e2a43a7e07ba1dd",
    "optimizer.py": "1a69e52dd45f463ad9187434f22f12d98c24e969c9b908c9c094d29d87548d4d",
    "pinn_d5_balanced_experiment.py": "7b7c0b6836f1834644071a54518046a1b437514fd1bc2cd5ba0ee5a570f31010",
}
SOAP_VERSION = "3.3.0"
SCHEMA = "flowmllab-d5-controlled-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(array_digest(value.detach().cpu().numpy()).encode())
    return h.hexdigest()


def model_digest(model):
    return state_digest(model.state_dict())


def load_author_source(directory):
    directory = Path(directory)
    modules = {}
    for filename, expected in AUTHOR_SHA256.items():
        path = directory / filename
        if sha256(path) != expected:
            raise ValueError(f"Author source checksum mismatch: {filename}")
        spec = importlib.util.spec_from_file_location("d5_author_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[path.stem] = module
    return SimpleNamespace(cavity=modules["cavity"], optimizer=modules["optimizer"],
                           sampler=modules["pinn_d5_balanced_experiment"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed; migration admission required")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def atomic_torch_save(payload, path):
    """One complete checkpoint, replaced only after a successful durable write."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def make_problem(source, seed, device="cpu", width=64, depth=6):
    torch.set_default_dtype(torch.float64)
    seed_everything(seed)
    geom = source.cavity.Geometry(W=1., D=5., Re=1000., envelope=True,
                                  kappa=4.212, y0_env=.25, beta=10.)
    mlp = source.cavity.MLP(width=width, depth=depth).to(device)
    return source.cavity.CavityNet(mlp, geom), geom


def frozen_points(source, n, seed):
    """The identical authored regional mixture, frozen for both optimizer arms."""
    a = source.sampler.balanced_points(n, seed)
    return torch.as_tensor(a, dtype=torch.float64)


def audit_points(seed=20264, n=1024):
    """Independent physical samples; no training exclusion or corner mask."""
    rng = np.random.default_rng(seed)
    bounds = {
        "Interior": (.02, .98, -3.98, .98),
        "Upper-left": (.02, .25, .75, .98),
        "Floor_0.4": (.02, .98, -3.98, -3.6),
        "Floor_0.65": (.02, .98, -3.98, -3.35),
        "Lid_corner_left": (.0001, .02, .98, .9999),
        "Lid_corner_right": (.98, .9999, .98, .9999),
    }
    for i in range(5):
        bounds[f"Height_{i}_{i+1}"] = (.02, .98, -4+i+.002, -3+i-.002)
    return {name: torch.as_tensor(rng.random((n, 2)) * [x1-x0, y1-y0] + [x0, y0],
                                 dtype=torch.float64)
            for name, (x0, x1, y0, y1) in bounds.items()}


def _grad(z, x, y):
    return torch.autograd.grad(z, (x, y), torch.ones_like(z), create_graph=True,
                               retain_graph=True)


def raw_residuals(source, net, points, geom):
    """Unmasked, unscaled physical equations, independent of training weighting."""
    x = points[:, :1].detach().clone().requires_grad_(True)
    y = points[:, 1:2].detach().clone().requires_grad_(True)
    u, v, p, psi = source.cavity._uvp(net, x, y, geom)
    ux, uy = _grad(u, x, y)
    vx, vy = _grad(v, x, y)
    uxx, _ = _grad(ux, x, y)
    _, uyy = _grad(uy, x, y)
    vxx, _ = _grad(vx, x, y)
    _, vyy = _grad(vy, x, y)
    px, py = _grad(p, x, y)
    rx = u*ux + v*uy + px - (uxx+uyy)/geom.Re
    ry = u*vx + v*vy + py - (vxx+vyy)/geom.Re
    return {"rx": rx, "ry": ry, "div": ux+vy, "omega": vx-uy,
            "u": u, "v": v, "p": p, "psi": psi}


def loss_backward(source, net, points, geom, chunk_size):
    """Global sum / global N: changing chunks does not reweight samples."""
    total = torch.zeros((), dtype=torch.float64, device=points.device)
    for chunk in points.split(chunk_size):
        rx, ry = source.cavity.residual(net, chunk, geom)
        part = (rx.square().sum() + ry.square().sum()) / len(points)
        if not torch.isfinite(part):
            raise FloatingPointError("Nonfinite weighted momentum objective")
        part.backward()
        total += part.detach()
    return total


def make_optimizer(source, net, method, phase, qn_capacity):
    if phase == "warm":
        if method == "adam_lbfgs":
            return torch.optim.Adam(net.parameters(), lr=.002, betas=(.99, .999))
        version = importlib.metadata.version("pytorch-optimizer")
        if version != SOAP_VERSION:
            raise ValueError(f"New SOAP runtime must be pinned to {SOAP_VERSION}; got {version}")
        from pytorch_optimizer.optimizer.soap import SOAP
        return SOAP(net.parameters(), lr=.002, betas=(.99, .999), weight_decay=0.,
                    precondition_frequency=1)
    if method == "adam_lbfgs":
        return torch.optim.LBFGS(net.parameters(), lr=1., max_iter=1,
                                max_eval=25, tolerance_grad=1e-12,
                                tolerance_change=0., history_size=100,
                                line_search_fn="strong_wolfe")
    return source.optimizer.SSBroydenLM(net.parameters(), m=max(1, qn_capacity),
                                        lr=1., gtol=1e-12, c1=1e-4, c2=.9,
                                        wolfe_maxiter=20, zoom_maxiter=40, amax=10,
                                        dtype=torch.float64,
                                        device=next(net.parameters()).device)


def pack_optimizer(optimizer):
    if hasattr(optimizer, "H"):
        if optimizer.H.dropped:
            raise ValueError("Author SSB generators dropped: comparison invalid")
        store = optimizer.H
        if not np.isfinite([store.S, store.sigma]).all() or not all(
            torch.isfinite(value[:store.n]).all()
            for value in (store.U, store.Sv, store.cuu, store.cus, store.css)):
            raise FloatingPointError("Nonfinite SSB curvature; preserving previous disk recovery")
        return {"kind": "author_SSBroydenLM", "torch": cpu_copy(optimizer.state_dict()),
                "H": optimizer.H.state_dict(), "capacity": optimizer.H.m,
                "n_ls_evals": optimizer.n_ls_evals}
    return {"kind": type(optimizer).__name__, "torch": cpu_copy(optimizer.state_dict())}


def unpack_optimizer(optimizer, state):
    if state["kind"] == "author_SSBroydenLM":
        if not hasattr(optimizer, "H") or optimizer.H.m != state["capacity"]:
            raise ValueError("SSB representation/capacity changed on resume")
        optimizer.load_state_dict(state["torch"])
        optimizer.H.load_state_dict(state["H"])
        optimizer.n_ls_evals = int(state["n_ls_evals"])
        if optimizer.H.dropped:
            raise ValueError("Refusing incomplete SSB curvature history")
    else:
        if type(optimizer).__name__ != state["kind"]:
            raise ValueError("Optimizer changed on resume")
        optimizer.load_state_dict(state["torch"])


class EvaluationBudgetReached(RuntimeError):
    pass


class CountedClosure:
    """Counts every full loss/gradient evaluation, including line-search trials."""
    def __init__(self, source, net, points, geom, chunk_size, optimizer,
                 budget, counters, history, phase, step, clock):
        self.source, self.net, self.points, self.geom = source, net, points, geom
        self.chunk_size, self.optimizer = chunk_size, optimizer
        self.budget, self.counters, self.history = budget, counters, history
        self.phase, self.step, self.clock = phase, step, clock

    def __call__(self):
        key = self.phase + "_evaluations"
        if self.counters[key] >= self.budget:
            raise EvaluationBudgetReached()
        self.counters[key] += 1
        self.counters["function_evaluations"] += 1
        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_backward(self.source, self.net, self.points, self.geom, self.chunk_size)
        self.counters["gradient_evaluations"] += 1
        grads = [p.grad for p in self.net.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Nonfinite parameter gradient")
        gn = torch.sqrt(sum(g.square().sum() for g in grads))
        self.history.append({"phase": self.phase, "step": self.step,
                             "function_evaluations": self.counters["function_evaluations"],
                             "gradient_evaluations": self.counters["gradient_evaluations"],
                             "elapsed_seconds": self.clock(), "loss": float(loss),
                             "evaluation_kind": "pre_update_closure" if self.phase == "warm" else "quasi_Newton_trial",
                             "gradient_norm": float(gn), "accepted": False,
                             "model_sha256": model_digest(self.net)})
        return loss


def controlled_step(closure):
    """Budget interruption restores the last accepted model/optimizer state.

    The authentic SSB implementation changes H only after line search returns,
    so no giant per-step curvature copy is needed.  L-BFGS updates history
    before line search and therefore needs its small rolling-state snapshot.
    """
    net, opt = closure.net, closure.optimizer
    previous_model = cpu_copy(net.state_dict())
    previous_opt = None if hasattr(opt, "H") else pack_optimizer(opt)
    ssb_evals = opt.n_ls_evals if hasattr(opt, "H") else None
    start_calls = closure.counters["function_evaluations"]
    try:
        if closure.phase == "warm":
            closure()
            opt.step()
        else:
            opt.step(closure)
        if not all(torch.isfinite(p).all() for p in net.parameters()):
            raise FloatingPointError("Nonfinite accepted parameters")
    except (EvaluationBudgetReached, FloatingPointError):
        net.load_state_dict(previous_model)
        if previous_opt is not None:
            unpack_optimizer(opt, previous_opt)
        else:
            # Do not count the denied call; all actually computed trials count.
            opt.n_ls_evals = ssb_evals + max(0, closure.counters["function_evaluations"]-start_calls-1)
        opt.zero_grad(set_to_none=True)
        raise
    accepted = model_digest(net)
    for row in reversed(closure.history):
        if row["phase"] != closure.phase or row["step"] != closure.step:
            break
        if row["model_sha256"] == accepted:
            row["accepted"] = True
            break
    if hasattr(opt, "H") and opt.H.dropped:
        raise RuntimeError("SSB generator capacity exhausted; state invalid")


def independent_audit(source, net, geom, points_by_region, chunk_size=256):
    device = next(net.parameters()).device
    report = {}
    for name, points in points_by_region.items():
        sums = dict(rx=0., ry=0., div=0., u=0., v=0.)
        emin, emax = float("inf"), 0.
        for chunk in points.split(chunk_size):
            z = chunk.to(device)
            values = raw_residuals(source, net, z, geom)
            for key in sums:
                sums[key] += float(values[key].detach().square().sum())
            E, _ = source.cavity.envelope(z[:, 1:2], geom)
            emin, emax = min(emin, float(E.min())), max(emax, float(E.max()))
        n = len(points)
        report[name] = {"n": n, "sample_sha256": array_digest(points.numpy()),
                        **{key+"_rms": float(np.sqrt(value/n)) for key,value in sums.items()},
                        "physical_momentum_rms": float(np.sqrt((sums["rx"]+sums["ry"])/n)),
                        "velocity_rms_U": float(np.sqrt((sums["u"]+sums["v"])/n)),
                        "envelope_min": emin, "envelope_max": emax}
    return report


def wall_audit(source, net, geom, n=257):
    device = next(net.parameters()).device
    s = torch.linspace(0., 1., n, dtype=torch.float64, device=device)
    walls = {"top": torch.stack((s, torch.ones_like(s)), 1),
             "bottom": torch.stack((s, torch.full_like(s, -4.)), 1),
             "left": torch.stack((torch.zeros_like(s), -4.+5*s), 1),
             "right": torch.stack((torch.ones_like(s), -4.+5*s), 1)}
    result = {}
    for name, points in walls.items():
        u, v, _, _ = source.cavity.fields(net, points, geom)
        target = ((1-torch.exp(-s.square()/.001)) *
                  (1-torch.exp(-(s-1).square()/.001))) if name == "top" else torch.zeros_like(s)
        result[name] = {"u_max_error": float((u[:,0]-target).abs().max()),
                        "v_max_error": float(v.abs().max())}
    return result


def prediction(source, net, geom, nx=121, nh=605, chunk_size=1024):
    """Height-above-floor grid, raw pressure; omega is labelled native AD."""
    x, h = np.linspace(.002, .998, nx), np.linspace(.002, 4.998, nh)
    X, H = np.meshgrid(x, h)
    points = torch.tensor(np.column_stack((X.ravel(), H.ravel()-4.)),
                          dtype=torch.float64, device=next(net.parameters()).device)
    fields = {q: [] for q in ("u", "v", "p", "psi", "omega")}
    for z in points.split(chunk_size):
        xx = z[:,:1].detach().clone().requires_grad_(True)
        yy = z[:,1:2].detach().clone().requires_grad_(True)
        u,v,p,psi = source.cavity._uvp(net, xx, yy, geom)
        vx,_ = _grad(v, xx, yy)
        _,uy = _grad(u, xx, yy)
        for key,value in zip(fields, (u,v,p,psi,vx-uy)):
            fields[key].append(value.detach().cpu().numpy())
    return {"x": x, "h": h, **{k: np.concatenate(v).reshape(nh,nx) for k,v in fields.items()}}
