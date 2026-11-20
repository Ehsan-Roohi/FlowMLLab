"""Independent polynomial, manufactured-PDE and conservation checks for W1.4."""
import numpy as np
from flowmllab.high_order import (
    chebyshev_grid, poisson_square, dg_operators, dg_rhs,
    dg_diagnostics, solve_dg,
)


def test_chebyshev_polynomials_and_constant():
    x, d = chebyshev_grid(12)
    assert np.max(np.abs(d @ np.ones_like(x))) < 1e-12
    assert np.max(np.abs(d @ x**5 - 5 * x**4)) < 1e-11
    assert np.max(np.abs((d @ d) @ x**5 - 20 * x**3)) < 1e-9


def test_manufactured_poisson_accuracy_and_fd_order():
    spectral = poisson_square(16)
    coarse = poisson_square(16, "finite_difference")
    fine = poisson_square(32, "finite_difference")
    assert spectral["max_error"] < 1e-9
    assert 3.5 < coarse["relative_l2"] / fine["relative_l2"] < 4.5
    assert max(spectral["residual_max"], fine["residual_max"]) < 1e-8
    assert np.all(spectral["p"][[0, -1], :] == 0)
    assert np.all(spectral["p"][:, [0, -1]] == 0)


def test_dg_constant_mass_and_semidiscrete_energy_identity():
    rng = np.random.default_rng(14)
    for speed in (-1.7, 1.0):
        for alpha in (0.0, 1.0):
            ops = dg_operators(9, 4, speed, alpha)
            q = rng.normal(size=(9, 5))
            rhs = dg_rhs(q, ops)
            assert abs(ops["h"] * rhs[:, 0].sum()) < 1e-12
            derivative = np.sum(q * rhs / ops["inverse_mass"])
            jumps = q.sum(axis=1) - np.roll(q @ ops["left"], -1)
            expected = -0.5 * alpha * abs(speed) * np.sum(jumps**2)
            assert abs(derivative - expected) < 1e-10
            constant = np.zeros_like(q)
            constant[:, 0] = 2.3
            assert np.max(np.abs(dg_rhs(constant, ops))) < 1e-11


def test_transport_refinement_and_both_wind_directions():
    for speed in (-1, 1):
        coarse = solve_dg(8, 2, speed=speed, final_time=0.25)
        fine = solve_dg(16, 2, speed=speed, final_time=0.25)
        assert fine["relative_l2"] < coarse["relative_l2"] / 4
        assert fine["relative_l2"] < 0.01
        assert abs(fine["mass_drift"]) < 1e-12
        assert dg_diagnostics(fine["coefficients"], fine["ops"])[1] <= fine["history"][0, 2] + 1e-12


def test_time_refinement_and_discontinuous_projection():
    from flowmllab.high_order import dg_project
    ops = dg_operators(11, 3)
    q = dg_project(ops, "step")
    assert abs(dg_diagnostics(q, ops)[0] - 0.3) < 1e-13
    fast = solve_dg(8, 5, cfl=0.12)
    half = solve_dg(8, 5, cfl=0.06)
    quarter = solve_dg(8, 5, cfl=0.03)
    assert np.linalg.norm(half["coefficients"] - quarter["coefficients"]) < np.linalg.norm(fast["coefficients"] - half["coefficients"]) / 5
