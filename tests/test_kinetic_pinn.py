"""Analytic physics and fair-work contracts for the fresh CPU BGK PINN."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flowmllab import kinetic_pinn as kp


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def constant(macros):
    values = torch.tensor(macros, dtype=torch.float64)
    return lambda tx: values.expand(*tx.shape[:-1], 3)


def constant_initial(macros):
    values = torch.tensor(macros, dtype=torch.float64)
    return lambda x: values.expand(*x.shape, 3)


def test_uniform_equilibrium_characteristics_and_all_nine_moments():
    state = (0.8, 0.3, 0.9)
    rule = kp.Quadrature(64, 12, 10)
    v, w = kp.velocity_quadrature(rule)
    points = torch.tensor([[0, -.5], [.001, -.49], [.02, 0], [.02, .5]], dtype=kp.DTYPE)
    reduced = kp.characteristic_reductions(constant(state), points, v, 12,
                                           initial=constant_initial(state), left=state, right=state)
    expected = kp.maxwellian_reductions(v, torch.tensor(state, dtype=kp.DTYPE))
    torch.testing.assert_close(reduced, expected.expand_as(reduced), rtol=3e-15, atol=3e-15)
    fields = kp.integrated_moments(v, w, reduced)
    for name, target in zip(("rho", "u", "T"), state):
        torch.testing.assert_close(fields[name], torch.full((4,), target, dtype=kp.DTYPE), rtol=2e-13, atol=2e-13)
    for name in ("q", "sigma", "m", "R", "Delta", "S"):
        torch.testing.assert_close(fields[name], torch.zeros(4, dtype=kp.DTYPE), rtol=0, atol=3e-12)


def test_initial_distribution_is_single_local_maxwellian_including_corners():
    model = kp.SourceNetwork(kp.PINNConfig())
    x = torch.tensor([-.5, -.13, 0, .21, .5], dtype=kp.DTYPE)
    v = torch.tensor([-3, -1, 0, 1, 3], dtype=kp.DTYPE)
    points = torch.stack((torch.zeros_like(x), x), -1)
    result = kp.characteristic_reductions(model, points, v)
    expected = kp.maxwellian_reductions(v[None, :], kp.initial_macros(x)[:, None, :])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(model(points), kp.initial_macros(x), rtol=0, atol=0)
    left = kp.maxwellian_reductions(v, torch.tensor(kp.LEFT, dtype=kp.DTYPE))
    right = kp.maxwellian_reductions(v, torch.tensor(kp.RIGHT, dtype=kp.DTYPE))
    # A blend of reservoirs is a different distribution even at the same x.
    assert float((result[2] - .5 * (left + right)).abs().max().detach()) > .01


def test_half_range_inflow_and_zero_velocity_characteristic():
    model = kp.SourceNetwork(kp.PINNConfig())
    v = torch.tensor([-2, 0, 2], dtype=kp.DTYPE)
    points = torch.tensor([[.02, -.5], [.02, .5]], dtype=kp.DTYPE)
    result = kp.characteristic_reductions(model, points, v)
    left = kp.maxwellian_reductions(v, torch.tensor(kp.LEFT, dtype=kp.DTYPE))
    right = kp.maxwellian_reductions(v, torch.tensor(kp.RIGHT, dtype=kp.DTYPE))
    torch.testing.assert_close(result[0, 2], left[2], rtol=0, atol=0)
    torch.testing.assert_close(result[1, 0], right[0], rtol=0, atol=0)
    assert torch.isfinite(result).all()
    assert (result > 0).all()
    assert (result[..., 0] * result[..., 2] - result[..., 1]**2 > 0).all()


def test_free_streaming_and_constant_source_relaxation_limits():
    v = torch.tensor([-2., 0., 1.5], dtype=kp.DTYPE)
    points = torch.tensor([[.02, -.02], [.013, .08]], dtype=kp.DTYPE)
    model = kp.SourceNetwork(kp.PINNConfig())
    ballistic = kp.characteristic_reductions(model, points, v, tau=1e12)
    foot = points[:, 1, None] - v[None, :] * points[:, 0, None]
    exact = kp.maxwellian_reductions(v[None, :], kp.initial_macros(foot))
    torch.testing.assert_close(ballistic, exact, rtol=2e-13, atol=2e-13)
    initial, source, tau = (1., -.2, .8), (.7, .4, 1.1), .005
    result = kp.characteristic_reductions(constant(source), points, v, 16, tau=tau,
                                           initial=constant_initial(initial), left=initial, right=initial)
    a = torch.exp(-points[:, 0, None, None] / tau)
    f0 = kp.maxwellian_reductions(v, torch.tensor(initial, dtype=kp.DTYPE))
    fs = kp.maxwellian_reductions(v, torch.tensor(source, dtype=kp.DTYPE))
    torch.testing.assert_close(result, a * f0 + (1 - a) * fs, rtol=2e-14, atol=2e-14)


def test_fourth_moments_require_k_even_when_consistency_is_identical():
    v, w = kp.velocity_quadrature(kp.Quadrature(64, 12, 10))
    g = torch.exp(-v**2 / 2) / np.sqrt(2 * np.pi)
    a = torch.stack((g, 2 * g, 8 * g), -1)[None, ...]
    b = torch.stack((g, 2 * g, 10 * g), -1)[None, ...]
    ma, mb = kp.integrated_moments(v, w, a), kp.integrated_moments(v, w, b)
    for name in ("rho", "u", "T", "q", "sigma", "m", "S"):
        torch.testing.assert_close(ma[name], mb[name], rtol=0, atol=2e-14)
    assert float(mb["R"][0]) == pytest.approx(-2/3, abs=3e-12)
    assert float(mb["Delta"][0]) == pytest.approx(2, abs=3e-12)


def test_seed_is_local_and_points_do_not_depend_on_architecture_or_optimizer():
    c = kp.PINNConfig()
    before = torch.random.get_rng_state().clone()
    first = kp.SourceNetwork(c)
    assert torch.equal(before, torch.random.get_rng_state())
    other = kp.SourceNetwork(replace(c, optimizer="lbfgs", weights=(.1, 1, 10)))
    for a, b in zip(first.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    points = kp.collocation(c)
    torch.testing.assert_close(points, kp.collocation(replace(c, depth=3, width=32)), rtol=0, atol=0)
    assert not torch.equal(points, kp.collocation(c, audit=True))
    assert (points[:, 0] > 0).all()
    assert int((points[:, 0] == kp.TF).sum()) == c.collocation_points // 4


def test_objective_uses_physical_energy_and_has_finite_correct_gradient():
    c = kp.PINNConfig(collocation_points=16)
    model, points = kp.SourceNetwork(c), kp.collocation(c)
    result = kp.objective(model, points, c)
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    p = model.network[-1].bias
    analytic = float(p.grad[2])
    eps = 1e-6
    with torch.no_grad():
        p[2] += eps
        upper = float(kp.objective(model, points, c)["loss"])
        p[2] -= 2 * eps
        lower = float(kp.objective(model, points, c)["loss"])
        p[2] += eps
    assert analytic == pytest.approx((upper - lower) / (2 * eps), rel=2e-7, abs=1e-10)
    changed = kp.objective(model, points, replace(c, weights=(2., 3., 4.)))
    torch.testing.assert_close(changed["raw_mse"], result["raw_mse"], rtol=1e-12, atol=1e-16)
    torch.testing.assert_close(changed["scaled_mse"], changed["raw_mse"], rtol=0, atol=0)
    torch.testing.assert_close(changed["weighted_components"], changed["raw_mse"] * torch.tensor([2., 3., 4.]))
    # Verify the third invariant is physical total energy, with the half factor.
    v, w = kp.velocity_quadrature(c.quadrature())
    reduced = kp.characteristic_reductions(model, points, v, c.time_order)
    source = model(points)
    energy = .5 * source[:, 0] * (source[:, 1]**2 + 3 * source[:, 2])
    energy -= (.5 * (v**2 * reduced[..., 0] + reduced[..., 1]) * w).sum(-1)
    assert float(changed["raw_mse"][2].detach()) == pytest.approx(float(energy.square().mean().detach()), rel=1e-14)


def test_training_has_no_target_reads_improves_loss_and_logs_fixed_objective(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training attempted a target/checkpoint read")
    for name in ("load", "loadtxt", "genfromtxt"):
        monkeypatch.setattr(np, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    read_bytes = Path.read_bytes
    def own_source_only(path):
        resolved = path.resolve()
        # PyTorch lazily imports optimizer support; pytest reads its .py files.
        # Inside this project the trainer may read only its own source for SHA.
        assert resolved.suffix == ".py"
        if resolved.is_relative_to(Path(kp.__file__).resolve().parents[1]):
            assert resolved == Path(kp.__file__).resolve()
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", own_source_only)
    c = kp.PINNConfig(collocation_points=16, adam_steps=20, lbfgs_steps=5, max_seconds=120)
    objective, seen = kp.objective, []
    def observed(model, points, config, quadrature=None):
        seen.append(points.detach().clone())
        return objective(model, points, config, quadrature)
    monkeypatch.setattr(kp, "objective", observed)
    bundle = kp.train(c)
    d = bundle.training_diagnostics
    assert d["final_training"]["loss"] < d["initial_training"]["loss"] * .6
    assert bundle.stop_reason == "requested_steps_completed"
    assert bundle.work["optimizer_steps"] == 25
    assert bundle.work["gradient_evaluations"] == len(bundle.closure_history)
    assert bundle.work["function_evaluations"] == len(seen)
    assert len(bundle.history) == 26
    assert [h["step"] for h in bundle.history] == list(range(26))
    assert bundle.history[-1]["loss"] == d["final_training"]["loss"]
    assert bundle.provenance["DVM_data_read"] is False
    assert bundle.provenance["initial_state_sha256"] != bundle.provenance["final_state_sha256"]
    assert d["audit_used_for_updates_or_selection"] is False
    training, audit = kp.collocation(c), kp.collocation(c, audit=True)
    assert all(torch.equal(p, training) or torch.equal(p, audit) for p in seen)
    assert sum(torch.equal(p, audit) for p in seen) == 2
    json.dumps(kp.training_report(bundle), allow_nan=False)
    x = np.linspace(-.45, .45, 13)
    evaluated = kp.evaluate(bundle, x, quadrature=kp.Quadrature(48, 12, 7))
    initial = kp.evaluate(bundle, x, state="initial")
    assert evaluated["fields"].shape == (13, 9)
    assert evaluated["columns"] == kp.FIELDS
    np.testing.assert_array_equal(evaluated["x"], x)
    np.testing.assert_allclose(evaluated["S"], evaluated["R"] + evaluated["Delta"] / 3, atol=1e-15)
    assert np.max(np.abs(evaluated["rho"] - evaluated["source_rho"])) > 1e-5
    assert np.max(np.abs(evaluated["fields"] - initial["fields"])) > 1e-5
    assert evaluated["clipping_applied"] is False
    assert evaluated["new_training_during_evaluation"] is False


def test_failed_post_update_diagnostic_rolls_back_state_and_completed_counts(monkeypatch):
    original, calls = kp.objective, 0
    def fail_fourth(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 4:  # initial, initial audit, gradient, then post-update audit
            result["loss"] = torch.tensor(float("nan"), dtype=kp.DTYPE)
        return result
    monkeypatch.setattr(kp, "objective", fail_fourth)
    b = kp.train(kp.PINNConfig(optimizer="adam", adam_steps=2, collocation_points=16))
    assert b.stop_reason == "nonfinite_update_rolled_back"
    assert b.work["optimizer_steps"] == b.work["adam_steps"] == 0
    assert b.work["rolled_back_steps"] == 1
    assert len(b.history) == 1
    assert b.provenance["initial_state_sha256"] == b.provenance["final_state_sha256"]


def test_repeated_short_runs_are_deterministic_and_optimizer_studies_start_equal():
    c = kp.PINNConfig(optimizer="adam", adam_steps=2, collocation_points=16)
    a, b = kp.train(c), kp.train(c)
    assert a.provenance["final_state_sha256"] == b.provenance["final_state_sha256"]
    assert a.training_diagnostics == b.training_diagnostics
    other = kp.train(replace(c, optimizer="lbfgs", lbfgs_steps=1, weights=(1, 1, 10)))
    assert a.provenance["initial_state_sha256"] == other.provenance["initial_state_sha256"]
    assert a.provenance["collocation_sha256"] == other.provenance["collocation_sha256"]


def test_zero_update_budget_is_explicit_and_preserves_initial_state():
    b = kp.train(kp.PINNConfig(collocation_points=16, max_seconds=1e-12))
    assert b.stop_reason == "time_budget_before_update"
    assert b.work["optimizer_steps"] == 0
    assert b.provenance["initial_state_sha256"] == b.provenance["final_state_sha256"]
    assert b.work["elapsed_seconds"] > b.config.max_seconds  # diagnostics are separately counted


@pytest.mark.parametrize("kwargs", [dict(depth=0), dict(width=100), dict(optimizer="random"),
    dict(weights=(0, 0, 0)), dict(weights=(1, -1, 1)), dict(weights=(1, float("nan"), 1)),
    dict(max_seconds=601), dict(collocation_points=256, velocity_order=128, time_order=32)])
def test_invalid_or_unbounded_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        kp.PINNConfig(**kwargs)


def test_characteristics_reject_out_of_case_evaluation():
    model = kp.SourceNetwork(kp.PINNConfig())
    for tx in ([[.021, 0]], [[-.01, 0]], [[.01, .501]], [[.01, float("nan")]]):
        with pytest.raises(ValueError):
            kp.characteristic_reductions(model, tx, [-1, 0, 1])
