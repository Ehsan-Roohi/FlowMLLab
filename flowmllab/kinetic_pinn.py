"""Fresh, bounded CPU characteristic-integral BGK training for Week 10.2.

Independent PyTorch implementation of the mathematical BGK characteristic
formula. No research solver code, DVM fields, checkpoint, or observed moment
targets are imported. The fixed problem is the smoothed tube on [-.5,.5],
tau=1, final time .02, with three molecular velocity components. Transverse
Gaussian moments are integrated analytically; axial and time quadrature remain
finite. This is a new teaching experiment, not a reproduction of a paper run.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

TF = 0.02
TAU = 1.0
LEFT = (1.0, 0.0, 1.0)
RIGHT = (0.125, 0.0, 0.8)
FIELDS = ("rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S")
COMPONENTS = ("density", "momentum", "energy")
# rho_L, rho_L*sqrt(T_L), rho_L*T_L: physical units, never fitted to data.
SCALES = (1.0, 1.0, 1.0)
DTYPE = torch.float64
# Capture the implementation actually imported, even if a file is edited later.
_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Quadrature:
    velocity_order: int = 48
    time_order: int = 12
    vmax: float = 7.0

    def __post_init__(self) -> None:
        for name, lower, upper in (("velocity_order", 8, 128), ("time_order", 2, 32)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower},{upper}]")
        if not math.isfinite(self.vmax) or not 3 <= self.vmax <= 12:
            raise ValueError("vmax must lie in [3,12] for this bounded CPU exercise")


@dataclass(frozen=True)
class PINNConfig:
    seed: int = 1729
    depth: int = 2
    width: int = 24
    optimizer: str = "adam_lbfgs"
    adam_steps: int = 100
    lbfgs_steps: int = 20
    learning_rate: float = 0.003
    lbfgs_learning_rate: float = 1.0
    lbfgs_history_size: int = 20
    collocation_points: int = 64
    velocity_order: int = 32
    time_order: int = 8
    vmax: float = 7.0
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    max_seconds: float = 120.0
    threads: int = 1

    def __post_init__(self) -> None:
        bounds = {"seed": (0, 2**31 - 1), "depth": (1, 5), "width": (4, 64),
                  "adam_steps": (0, 2000), "lbfgs_steps": (0, 200),
                  "lbfgs_history_size": (2, 50), "collocation_points": (16, 256), "threads": (1, 4)}
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo},{hi}]")
        if self.optimizer not in {"adam", "lbfgs", "adam_lbfgs"}:
            raise ValueError("optimizer must be adam, lbfgs, or adam_lbfgs")
        if (len(self.weights) != 3 or any(not math.isfinite(w) or w < 0 for w in self.weights)
                or sum(self.weights) <= 0):
            raise ValueError("weights need three finite nonnegative values, with at least one positive")
        for name in ("learning_rate", "lbfgs_learning_rate", "max_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_seconds > 600:
            raise ValueError("max_seconds must not exceed the 600-second CPU lesson bound")
        Quadrature(self.velocity_order, self.time_order, self.vmax)
        if self.collocation_points * self.velocity_order * self.time_order > 131072:
            raise ValueError("collocation*velocity*time exceeds the bounded CPU graph size")

    def quadrature(self) -> Quadrature:
        return Quadrature(self.velocity_order, self.time_order, self.vmax)


def _tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        if value.device.type != "cpu":
            raise ValueError("this teaching implementation accepts CPU tensors only")
        return value.to(dtype=DTYPE)
    return torch.as_tensor(np.asarray(value), dtype=DTYPE, device="cpu")


def initial_macros(x: torch.Tensor) -> torch.Tensor:
    """Smoothed rho,T followed by a SINGLE local Maxwellian, not a mixture."""
    h = 0.5 * (1 + torch.tanh(20 * x))
    return torch.stack((1 - 0.875 * h, torch.zeros_like(x), 1 - 0.2 * h), dim=-1)


class SourceNetwork(nn.Module):
    """Positive rho,T lift with exact analytic source macros at t=0.

    Only the reconstructed distribution has hard incoming boundary data. The
    source network's entire boundary macro state is not prescribed.
    """
    def __init__(self, config: PINNConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            inputs = 2
            for _ in range(config.depth):
                layers.extend((nn.Linear(inputs, config.width, dtype=DTYPE), nn.Tanh()))
                inputs = config.width
            layers.append(nn.Linear(inputs, 3, dtype=DTYPE))
            self.network = nn.Sequential(*layers)
            with torch.no_grad():
                self.network[-1].weight.mul_(0.05)
                self.network[-1].bias.zero_()

    def forward(self, tx: torch.Tensor) -> torch.Tensor:
        a, x = tx[..., 0] / TF, tx[..., 1]
        raw = self.network(torch.stack((a, 2 * x), dim=-1))
        base = initial_macros(x)
        return torch.stack((base[..., 0] * torch.exp(0.5 * a * raw[..., 0]),
                            a * raw[..., 1],
                            base[..., 2] * torch.exp(0.5 * a * raw[..., 2])), dim=-1)


def velocity_quadrature(quadrature: Quadrature) -> tuple[torch.Tensor, torch.Tensor]:
    v, w = np.polynomial.legendre.leggauss(quadrature.velocity_order)
    return _tensor(v * quadrature.vmax), _tensor(w * quadrature.vmax)


def maxwellian_reductions(v: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
    """Return g,h,k with h=int r^2 f and k=int r^4 f (no factor 1/2)."""
    rho, u, T = macros.unbind(-1)
    g = rho / torch.sqrt(2 * math.pi * T) * torch.exp(-(v - u)**2 / (2 * T))
    return torch.stack((g, 2 * T * g, 8 * T**2 * g), dim=-1)


def characteristic_reductions(
    source: Callable[[torch.Tensor], torch.Tensor], tx: Any, velocity: Any,
    time_order: int = 8, tau: float = TAU,
    initial: Callable[[torch.Tensor], torch.Tensor] = initial_macros,
    left: tuple[float, float, float] = LEFT,
    right: tuple[float, float, float] = RIGHT,
) -> torch.Tensor:
    """Exact initial/inflow plus GL time integration on the backward trajectory.

    z in [0,1] is the elapsed backward-time fraction. The source is evaluated at
    (t-ell*z,x-v*ell*z), with ell limited by the first initial or boundary hit.
    Output shape is (Npoints,Nvelocity,3). No distribution clipping or fitted
    moment correction occurs. Optional analytic states/tau support unit tests;
    training always uses the fixed tube constants.
    """
    tx, v = _tensor(tx), _tensor(velocity)
    if (tx.ndim != 2 or tx.shape[1] != 2 or v.ndim != 1 or len(v) < 1
            or not bool(torch.isfinite(tx).all() and torch.isfinite(v).all())):
        raise ValueError("tx must be finite (N,2), velocity a finite vector")
    if bool((tx[:, 0] < 0).any() or (tx[:, 0] > TF + 1e-14).any()
            or (tx[:, 1].abs() > 0.5 + 1e-14).any()):
        raise ValueError("points must lie in t=[0,.02], x=[-.5,.5]")
    if not math.isfinite(tau) or tau <= 0 or not 2 <= time_order <= 32:
        raise ValueError("positive tau and time_order in [2,32] are required")
    t, x = tx[:, 0, None], tx[:, 1, None]
    vv = v[None, :]
    safe_speed = torch.where(vv != 0, vv.abs(), torch.ones_like(vv))
    distance = torch.where(vv > 0, x + 0.5, 0.5 - x)
    travel = torch.where(vv != 0, distance / safe_speed, torch.full_like(distance, float("inf")))
    ell = torch.minimum(t, travel).clamp_min(0)
    initial_x = x - vv * t
    boundary = torch.where((vv > 0)[..., None], _tensor(left), _tensor(right))
    # At a tie the backward characteristic reaches the initial surface. This
    # also preserves the smoothed initial Maxwellian at the two t=0 corners.
    base_macros = torch.where((travel < t)[..., None], boundary, initial(initial_x))
    base = maxwellian_reductions(vv, base_macros)
    nodes, weights = np.polynomial.legendre.leggauss(time_order)
    z, w = _tensor((nodes + 1) / 2), _tensor(weights / 2)
    lag = ell[..., None] * z
    trajectory = torch.stack(((t[..., None] - lag).expand(-1, len(v), -1),
                               x[..., None] - vv[..., None] * lag), dim=-1)
    macros = source(trajectory.reshape(-1, 2)).reshape(*lag.shape, 3)
    target = maxwellian_reductions(vv[..., None], macros)
    integral_weights = (ell[..., None] / tau) * w * torch.exp(-lag / tau)
    return torch.exp(-ell[..., None] / tau) * base + torch.sum(integral_weights[..., None] * target, dim=2)


def integrated_moments(v: torch.Tensor, w: torch.Tensor, reduced: torch.Tensor) -> dict[str, torch.Tensor]:
    """Integrate reconstructed g/h/k, including separate fourth-order moments."""
    g, h, k = reduced.unbind(-1)
    rho = (g * w).sum(-1)
    u = (g * v * w).sum(-1) / rho
    c = v[None, :] - u[:, None]
    T = ((c**2 * g + h) * w).sum(-1) / (3 * rho)
    sigma = (((2 * c**2 * g - h) / 3) * w).sum(-1)
    q = (0.5 * c * (c**2 * g + h) * w).sum(-1)
    m = ((2 * c**3 * g - 3 * c * h) * w / 5).sum(-1)
    R = ((2 * c**4 * g + c**2 * h - k) * w / 3).sum(-1) - 7 * T * sigma
    delta = ((c**4 * g + 2 * c**2 * h + k) * w).sum(-1) - 15 * rho * T**2
    return dict(rho=rho, u=u, T=T, q=q, sigma=sigma, m=m, R=R, Delta=delta, S=R + delta / 3)


def collocation(config: PINNConfig, *, audit: bool = False) -> torch.Tensor:
    """Fixed sampling uses a separate RNG, independent of network size/optimizer.

    Half the x locations lie in [-.15,.15], half in the whole domain. One
    quarter of times are exactly tf; the rest stratify (0,tf). The audit set
    uses another fixed seed and is never used for updates or state selection.
    """
    rng = torch.Generator(device="cpu").manual_seed(config.seed + (8675309 if audit else 104729))
    n = config.collocation_points
    split = n // 2
    x = torch.cat((0.3 * torch.rand(split, generator=rng, dtype=DTYPE) - 0.15,
                   torch.rand(n - split, generator=rng, dtype=DTYPE) - 0.5))
    nfinal = n // 4
    count = n - nfinal
    t = TF * (torch.arange(count, dtype=DTYPE) + torch.rand(count, generator=rng, dtype=DTYPE)) / count
    t = torch.cat((t, torch.full((nfinal,), TF, dtype=DTYPE)))
    t = t[torch.randperm(n, generator=rng)]
    return torch.stack((t, x), dim=1)


def _residual(model: SourceNetwork, points: torch.Tensor, quadrature: Quadrature) -> torch.Tensor:
    v, w = velocity_quadrature(quadrature)
    reduced = characteristic_reductions(model, points, v, quadrature.time_order)
    g, h = reduced[..., 0], reduced[..., 1]
    source = model(points)
    rho, u, T = source.unbind(-1)
    target = torch.stack(((g * w).sum(-1), (v * g * w).sum(-1),
                          (0.5 * (v**2 * g + h) * w).sum(-1)), dim=-1)
    implied = torch.stack((rho, rho * u, 0.5 * rho * (u**2 + 3 * T)), dim=-1)
    return implied - target


def objective(model: SourceNetwork, points: torch.Tensor, config: PINNConfig,
              quadrature: Quadrature | None = None) -> dict[str, torch.Tensor]:
    residual = _residual(model, points, config.quadrature() if quadrature is None else quadrature)
    raw = residual.square().mean(0)
    scaled = (residual / _tensor(SCALES)).square().mean(0)
    weighted = scaled * _tensor(config.weights)
    return {"loss": weighted.sum(), "raw_mse": raw, "scaled_mse": scaled,
            "weighted_components": weighted, "raw_rms": raw.sqrt()}


def _state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _array_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _numbers(result: dict[str, torch.Tensor]) -> dict[str, Any]:
    out: dict[str, Any] = {"loss": float(result["loss"].detach())}
    for key in ("raw_mse", "scaled_mse", "weighted_components", "raw_rms"):
        out[key] = {name: float(v) for name, v in zip(COMPONENTS, result[key].detach())}
    return out


@dataclass
class TrainingBundle:
    model: SourceNetwork
    initial_model: SourceNetwork
    config: PINNConfig
    history: list[dict[str, Any]]
    closure_history: list[dict[str, Any]]
    training_diagnostics: dict[str, Any]
    work: dict[str, Any]
    provenance: dict[str, Any]
    stop_reason: str


class _BudgetStop(Exception):
    pass


def train(config: PINNConfig | None = None) -> TrainingBundle:
    """Train from a fresh seeded initialization on a fixed physics objective.

    Adam is full-batch. Each LBFGS outer step uses max_iter=1 with strong-Wolfe
    search and the same points in every closure. Closure histories log actual
    objective/gradient calls; history rows are separate post-update reevaluations.
    A time-budget interruption inside LBFGS restores that outer step's entry
    state. The returned state is the final completed update, not a DVM-selected
    best model. Frozen audit points are evaluated only before/after training.
    """
    config = PINNConfig() if config is None else config
    if not isinstance(config, PINNConfig):
        raise TypeError("train expects PINNConfig")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.threads)
    start = time.perf_counter()
    model = SourceNetwork(config)
    initial_model = deepcopy(model)
    for p in initial_model.parameters():
        p.requires_grad_(False)
    points, audit_points = collocation(config), collocation(config, audit=True)
    history: list[dict[str, Any]] = []
    closures: list[dict[str, Any]] = []
    counts = {"gradient_evaluations": 0, "diagnostic_evaluations": 0, "optimizer_steps": 0,
              "adam_steps": 0, "lbfgs_steps": 0, "rolled_back_steps": 0}
    stop_reason = "requested_steps_completed"

    def diagnostic(network=model, locations=points):
        with torch.no_grad():
            result = objective(network, locations, config)
        counts["diagnostic_evaluations"] += 1
        if not bool(torch.isfinite(result["loss"])):
            raise FloatingPointError("nonfinite physics diagnostic")
        return _numbers(result)

    def record(phase, phase_step, change=0.0):
        row = diagnostic()
        if phase != "initial":
            counts["optimizer_steps"] += 1
            counts[f"{phase}_steps"] += 1
        row.update(step=counts["optimizer_steps"], optimizer=phase, phase_step=phase_step,
                   parameter_change_l2=change, gradient_evaluations=counts["gradient_evaluations"],
                   diagnostic_evaluations=counts["diagnostic_evaluations"],
                   function_evaluations=counts["gradient_evaluations"] + counts["diagnostic_evaluations"],
                   elapsed_seconds=time.perf_counter() - start)
        history.append(row)

    try:
        record("initial", 0)
        initial_audit = diagnostic(initial_model, audit_points)
        phases = []
        if config.optimizer in {"adam", "adam_lbfgs"}:
            phases.append(("adam", config.adam_steps, torch.optim.Adam(model.parameters(), lr=config.learning_rate)))
        if config.optimizer in {"lbfgs", "adam_lbfgs"}:
            phases.append(("lbfgs", config.lbfgs_steps,
                           torch.optim.LBFGS(model.parameters(), lr=config.lbfgs_learning_rate,
                                             max_iter=1, max_eval=12, tolerance_grad=1e-12,
                                             tolerance_change=1e-14, history_size=config.lbfgs_history_size,
                                             line_search_fn="strong_wolfe")))
        setup_elapsed = time.perf_counter() - start
        stop = False
        for phase, steps, optimizer in phases:
            for phase_step in range(1, steps + 1):
                if time.perf_counter() - start >= config.max_seconds:
                    stop_reason, stop = "time_budget_before_update", True
                    break
                before = {k: v.detach().clone() for k, v in model.state_dict().items()}

                def closure():
                    if time.perf_counter() - start >= config.max_seconds:
                        raise _BudgetStop()
                    optimizer.zero_grad(set_to_none=True)
                    result = objective(model, points, config)
                    counts["gradient_evaluations"] += 1
                    if not bool(torch.isfinite(result["loss"])):
                        raise FloatingPointError("nonfinite physics objective")
                    result["loss"].backward()
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                        raise FloatingPointError("nonfinite physics gradient")
                    closures.append({**_numbers(result), "optimizer": phase, "phase_step": phase_step,
                                     "next_optimizer_step": counts["optimizer_steps"] + 1,
                                     "gradient_evaluation": counts["gradient_evaluations"],
                                     "elapsed_seconds": time.perf_counter() - start})
                    return result["loss"]

                try:
                    if phase == "adam":
                        closure()
                        optimizer.step()
                    else:
                        optimizer.step(closure)
                    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError("nonfinite updated parameters")
                    change = math.sqrt(sum(float((v.detach() - before[k]).square().sum())
                                           for k, v in model.state_dict().items()))
                    record(phase, phase_step, change)
                except (_BudgetStop, FloatingPointError) as exc:
                    model.load_state_dict(before)
                    counts["rolled_back_steps"] += 1
                    stop_reason = ("time_budget_partial_update_rolled_back" if isinstance(exc, _BudgetStop)
                                   else "nonfinite_update_rolled_back")
                    stop = True
                    break
            if stop:
                break
        update_elapsed = time.perf_counter() - start
        model.eval()
        # Fresh evaluations on frozen points after the final completed update.
        final_training = diagnostic()
        final_audit = diagnostic(model, audit_points)
        diagnostics = {"initial_training": {k: history[0][k] for k in ("loss", "raw_mse", "scaled_mse", "weighted_components", "raw_rms")},
                       "final_training": final_training, "initial_audit": initial_audit,
                       "final_audit": final_audit, "audit_used_for_updates_or_selection": False}
        provenance = {"initial_state_sha256": _state_hash(initial_model), "final_state_sha256": _state_hash(model),
                      "collocation_sha256": _array_hash(points), "audit_points_sha256": _array_hash(audit_points),
                      "source_code_sha256": _SOURCE_SHA256,
                      "torch_version": torch.__version__, "dtype": "float64", "device": "cpu",
                      "case": "BGK_smoothed_tube_tau1_tf0.02", "initial_distribution": "single_local_Maxwellian_of_blended_macros",
                      "transverse_integration": "analytic Gaussian g/h/k", "DVM_data_read": False,
                      "model_selection": "final completed update; no best-reference-score selection",
                      "clipping_applied": False, "consistency_components": COMPONENTS, "physical_scales": SCALES,
                      "same_seed_architecture_initialization_independent_of_optimizer_and_weights": True,
                      "collocation_rng_independent_of_network_size": True,
                      "initialization": "seeded torch.nn.Linear default; final weights times .05 and zero final bias",
                      "optimizer_protocol": {"adam": "full batch on frozen points", "lbfgs_max_iter": 1,
                          "lbfgs_max_eval": 12, "lbfgs_line_search": "strong_wolfe",
                          "lbfgs_tolerance_grad": 1e-12, "lbfgs_tolerance_change": 1e-14},
                      "stopping_rule": "requested update counts or wall-time/nonfinite guard; no data-based selection"}
        counts.update(function_evaluations=counts["gradient_evaluations"] + counts["diagnostic_evaluations"],
                      setup_elapsed_seconds=setup_elapsed,
                      update_elapsed_seconds=update_elapsed - setup_elapsed, elapsed_seconds=time.perf_counter() - start,
                      final_diagnostics_elapsed_seconds=time.perf_counter() - start - update_elapsed,
                      parameter_count=sum(p.numel() for p in model.parameters()))
        return TrainingBundle(model, initial_model, config, history, closures, diagnostics, counts, provenance, stop_reason)
    finally:
        torch.set_num_threads(previous_threads)


def evaluate(bundle: TrainingBundle, x: Any, t: Any = TF,
             quadrature: Quadrature | dict[str, Any] | None = None,
             state: str = "trained", batch_size: int = 32) -> dict[str, Any]:
    """Evaluate integrated fields and separate source consistency after training.

    x is kept exactly as supplied; no alignment/interpolation to any reference
    occurs. Higher-order quadrature is a frozen-model sensitivity check. The
    source macros are returned separately and never substituted for fields.
    """
    if state not in {"trained", "initial"}:
        raise ValueError("state must be trained or initial")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
        raise ValueError("batch_size must be an integer in [1,128]")
    qrule = bundle.config.quadrature() if quadrature is None else quadrature
    if isinstance(qrule, dict):
        qrule = Quadrature(**qrule)
    if not isinstance(qrule, Quadrature):
        raise TypeError("quadrature must be Quadrature, dict, or None")
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or not len(xx) or not np.isfinite(xx).all():
        raise ValueError("x must be a nonempty finite vector")
    tt = np.broadcast_to(np.asarray(t, dtype=np.float64), xx.shape)
    points = _tensor(np.column_stack((tt, xx)))
    model = bundle.model if state == "trained" else bundle.initial_model
    v, w = velocity_quadrature(qrule)
    fields, sources, residuals = [], [], []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(bundle.config.threads)
    try:
        with torch.no_grad():
            for part in points.split(batch_size):
                reduced = characteristic_reductions(model, part, v, qrule.time_order)
                moment = integrated_moments(v, w, reduced)
                source = model(part)
                rho, u, T = source.unbind(-1)
                implied = torch.stack((rho, rho * u, 0.5 * rho * (u**2 + 3 * T)), dim=1)
                g, h = reduced[..., 0], reduced[..., 1]
                actual = torch.stack(((g * w).sum(-1), (v * g * w).sum(-1),
                                      (0.5 * (v**2 * g + h) * w).sum(-1)), dim=1)
                fields.append(torch.stack([moment[k] for k in FIELDS], dim=1).numpy())
                sources.append(source.numpy())
                residuals.append((implied - actual).numpy())
    finally:
        torch.set_num_threads(previous_threads)
    values, src, res = np.concatenate(fields), np.concatenate(sources), np.concatenate(residuals)
    if not np.isfinite(values).all() or not np.isfinite(src).all() or not np.isfinite(res).all():
        raise FloatingPointError("nonfinite frozen-model evaluation")
    return {"x": xx.copy(), "t": tt.copy(), "columns": FIELDS, "fields": values,
            **{name: values[:, i] for i, name in enumerate(FIELDS)}, "source_macros": src,
            **{f"source_{name}": src[:, i] for i, name in enumerate(("rho", "u", "T"))},
            "consistency_raw": res, "consistency_scaled": res / np.asarray(SCALES),
            "consistency_rms": np.sqrt(np.mean(res**2, axis=0)), "quadrature": asdict(qrule),
            "state": state, "transverse_integration": "analytic Gaussian g/h/k", "alignment": "none",
            "clipping_applied": False, "new_training_during_evaluation": False}


def training_report(bundle: TrainingBundle) -> dict[str, Any]:
    """JSON-ready compact report; model/optimizer tensors are excluded."""
    return {"config": asdict(bundle.config), "history": bundle.history,
            "closure_history": bundle.closure_history, "training_diagnostics": bundle.training_diagnostics,
            "work": bundle.work, "provenance": bundle.provenance, "stop_reason": bundle.stop_reason}
