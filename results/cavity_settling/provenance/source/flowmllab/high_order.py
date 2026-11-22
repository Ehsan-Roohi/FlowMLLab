"""Transparent CPU teaching kernels for Week 1.4.

Chebyshev collocation and modal Legendre DG are spatial discretizations.
These scalar model problems are not a full incompressible Navier--Stokes solver.
All coordinates, boundary conditions and error norms are explicit.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander
from scipy.interpolate import BarycentricInterpolator, RegularGridInterpolator
from scipy.linalg import solve_sylvester
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve


def chebyshev_grid(n):
    """n+1 increasing Lobatto nodes on [0,1] and their first derivative matrix."""
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer >= 2")
    x = (1 - np.cos(np.pi * np.arange(n + 1) / n)) / 2
    weights = (-1.0) ** np.arange(n + 1)
    weights[[0, -1]] *= 0.5
    delta = x[:, None] - x[None, :]
    np.fill_diagonal(delta, 1.0)
    d = (weights[None, :] / weights[:, None]) / delta
    np.fill_diagonal(d, 0.0)
    np.fill_diagonal(d, -d.sum(axis=1))
    return x, d


def interpolate_chebyshev(x, values, query, axis=0):
    """Evaluate the nodal polynomial with explicit Lobatto barycentric weights."""
    w = (-1.0) ** np.arange(len(x))
    w[[0, -1]] *= 0.5
    return BarycentricInterpolator(x, values, axis=axis, wi=w)(query)


def interpolate_field(x, field, query, method):
    """All fields use array indexing field[y_index, x_index]."""
    if method == "chebyshev":
        intermediate = interpolate_chebyshev(x, field, query, axis=1)
        return interpolate_chebyshev(x, intermediate, query, axis=0)
    if method == "finite_difference":
        yy, xx = np.meshgrid(query, query, indexing="ij")
        return RegularGridInterpolator((x, x), field)(np.stack((yy, xx), axis=-1))
    raise ValueError("Unknown spatial method")


def exact_pressure(x, y):
    """Manufactured nondimensional pressure, zero on the unit-square boundary."""
    return np.sin(np.pi * x) * np.sin(2 * np.pi * y)


def poisson_square(n=16, method="chebyshev"):
    """Solve Laplacian(p) = -5*pi**2*p_exact, with prescribed p=0 on walls.

    This Dirichlet manufactured problem verifies a spatial Poisson operator.
    It is NOT the physical pressure boundary condition of a driven cavity.
    Timings include assembly + solve; plotting and error interpolation excluded.
    """
    started = perf_counter()
    if method == "chebyshev":
        x, d = chebyshev_grid(n)
        a = (d @ d)[1:-1, 1:-1]
    elif method == "finite_difference":
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("n must be an integer >= 2")
        x = np.linspace(0, 1, n + 1)
        a = diags([np.ones(n - 2), -2 * np.ones(n - 1), np.ones(n - 2)],
                  [-1, 0, 1], format="csr") * n**2
    else:
        raise ValueError("Unknown spatial method")
    yy, xx = np.meshgrid(x, x, indexing="ij")
    rhs = -5 * np.pi**2 * exact_pressure(xx, yy)
    p = np.zeros_like(rhs)
    if method == "chebyshev":
        # Tensor-product Laplacian: A P + P A^T = F, no 2-D dense matrix.
        p[1:-1, 1:-1] = solve_sylvester(a, a.T, rhs[1:-1, 1:-1])
        residual = a @ p[1:-1, 1:-1] + p[1:-1, 1:-1] @ a.T - rhs[1:-1, 1:-1]
    else:
        lap = kron(eye(n - 1), a) + kron(a, eye(n - 1))
        p[1:-1, 1:-1] = spsolve(lap.tocsr(), rhs[1:-1, 1:-1].ravel()).reshape(n - 1, n - 1)
        residual = (lap @ p[1:-1, 1:-1].ravel() - rhs[1:-1, 1:-1].ravel())
    elapsed = perf_counter() - started
    query = np.linspace(0, 1, 151)
    yyq, xxq = np.meshgrid(query, query, indexing="ij")
    exact = exact_pressure(xxq, yyq)
    error = interpolate_field(x, p, query, method) - exact
    return dict(method=method, n=n, x=x, p=p, unknowns=(n - 1)**2,
                seconds=elapsed, residual_max=float(np.max(np.abs(residual))),
                relative_l2=float(np.linalg.norm(error) / np.linalg.norm(exact)),
                max_error=float(np.max(np.abs(error))))


def initial_wave(x, kind="smooth"):
    """Periodic data on a unit interval; smooth, localized, or discontinuous."""
    x = np.asarray(x) % 1.0
    if kind == "smooth":
        return 0.6 * np.sin(2 * np.pi * x) + 0.2 * np.cos(6 * np.pi * x)
    if kind == "localized":
        return np.exp(-60 * np.sin(np.pi * (x - 0.35))**2)
    if kind == "step":
        return ((x >= 0.25) & (x < 0.55)).astype(float)
    raise ValueError("kind must be smooth, localized or step")


def dg_operators(elements=16, degree=3, speed=1.0, dissipation=1.0):
    """Modal Legendre DG on equal periodic cells; 0=central flux, 1=upwind."""
    if not isinstance(elements, (int, np.integer)) or elements < 2:
        raise ValueError("elements must be an integer >= 2")
    if not isinstance(degree, (int, np.integer)) or not 0 <= degree <= 8:
        raise ValueError("degree must be an integer from 0 to 8 for this lab")
    if not np.isfinite(speed) or speed == 0 or not 0 <= dissipation <= 1:
        raise ValueError("Use finite nonzero speed and dissipation in [0,1]")
    h = 1.0 / elements
    k = np.arange(degree + 1)
    # S[m,n] = integral P'_m P_n: exactly 2 for m>n with odd m-n.
    s = 2.0 * ((k[:, None] > k[None, :]) & ((k[:, None] - k[None, :]) % 2 == 1))
    return dict(elements=elements, degree=degree, speed=float(speed),
                dissipation=float(dissipation), h=h, s=s, left=(-1.0)**k,
                inverse_mass=(2 * k + 1) / h)


def dg_project(ops, kind="smooth"):
    """L2 projection using Gauss integration, split at step discontinuities."""
    e, p, h = ops["elements"], ops["degree"], ops["h"]
    z, w = leggauss(max(32, 2 * p + 4))
    coefficients = np.zeros((e, p + 1))
    for cell in range(e):
        lo, hi = cell * h, (cell + 1) * h
        cuts = [lo] + ([c for c in (0.25, 0.55) if lo < c < hi] if kind == "step" else []) + [hi]
        for a, b in zip(cuts[:-1], cuts[1:]):
            x = (a + b) / 2 + (b - a) * z / 2
            xi = 2 * (x - lo) / h - 1
            coefficients[cell] += ((b - a) / 2) * (w * initial_wave(x, kind)) @ legvander(xi, p)
    return coefficients * ops["inverse_mass"][None, :]


def dg_rhs(coefficients, ops):
    """Weak-form volume derivative plus a single conservative interface flux."""
    a, alpha = ops["speed"], ops["dissipation"]
    right_trace = coefficients.sum(axis=1)
    left_trace = coefficients @ ops["left"]
    neighbour = np.roll(left_trace, -1)
    flux_right = 0.5 * a * (right_trace + neighbour) - 0.5 * alpha * abs(a) * (neighbour - right_trace)
    flux_left = np.roll(flux_right, 1)
    volume = a * (coefficients @ ops["s"].T)
    return (volume - flux_right[:, None] + flux_left[:, None] * ops["left"]) * ops["inverse_mass"]


def dg_values(coefficients, ops, x):
    """Evaluate each cell's polynomial, with the right-cell trace at interfaces."""
    x = np.asarray(x) % 1.0
    indices = np.minimum((x / ops["h"]).astype(int), ops["elements"] - 1)
    xi = 2 * (x / ops["h"] - indices) - 1
    return np.sum(coefficients[indices] * legvander(xi, ops["degree"]), axis=-1)


def dg_diagnostics(coefficients, ops):
    mass = ops["h"] * coefficients[:, 0].sum()
    energy = 0.5 * np.sum(coefficients**2 / ops["inverse_mass"][None, :])
    return float(mass), float(energy)


def solve_dg(elements=16, degree=3, speed=1.0, dissipation=1.0,
             kind="smooth", final_time=0.35, cfl=0.12):
    """SSPRK3 with dt <= cfl*h/(|a|*(2p+1)); cfl is a lab control, not a theorem.

    Return a finite-interval transport experiment, never a cavity flow solution.
    A work cap protects interactive Run All from accidental huge settings.
    """
    ops = dg_operators(elements, degree, speed, dissipation)
    if not np.isfinite(final_time) or final_time <= 0 or not np.isfinite(cfl) or cfl <= 0:
        raise ValueError("final_time and cfl must be finite and positive")
    steps = int(np.ceil(final_time * abs(speed) * (2 * degree + 1) / (cfl * ops["h"])))
    if steps > 100000:
        raise ValueError("More than 100000 steps: reduce the interactive workload")
    dt = final_time / steps
    started = perf_counter()
    q = dg_project(ops, kind)
    mass0, energy0 = dg_diagnostics(q, ops)
    history = [(0.0, mass0, energy0)]
    for step in range(steps):
        q1 = q + dt * dg_rhs(q, ops)
        q2 = 0.75 * q + 0.25 * (q1 + dt * dg_rhs(q1, ops))
        q = q / 3 + 2 / 3 * (q2 + dt * dg_rhs(q2, ops))
        if not np.isfinite(q).all() or np.max(np.abs(q)) > 1e8:
            raise RuntimeError("Unstable DG trajectory: decrease CFL; preserve the failed setting")
        if (step + 1) % max(1, steps // 80) == 0 or step + 1 == steps:
            history.append(((step + 1) * dt, *dg_diagnostics(q, ops)))
    elapsed = perf_counter() - started
    x = (np.arange(4096) + 0.5) / 4096
    exact = initial_wave(x - speed * final_time, kind)
    predicted = dg_values(q, ops, x)
    error = predicted - exact
    return dict(ops=ops, coefficients=q, history=np.asarray(history),
                x=x, exact=exact, predicted=predicted, kind=kind, dt=dt,
                steps=steps, rhs_calls=3 * steps, dofs=elements * (degree + 1),
                seconds=elapsed, final_time=final_time, cfl=cfl,
                mass_drift=float(dg_diagnostics(q, ops)[0] - mass0),
                relative_l2=float(np.linalg.norm(error) / np.linalg.norm(exact)),
                max_error=float(np.max(np.abs(error))))
