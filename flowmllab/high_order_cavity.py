"""Small, inspectable steady Navier--Stokes cavity on a CPU.

Velocity: tensor Chebyshev degree N, including prescribed wall nodes.
Pressure: degree N-2 on the interior nodes, with a centre pressure gauge.
Continuity is collocated at interior nodes. Its off-grid defect is also measured:
collocation convergence must not be confused with pointwise incompressibility.
The standard lid is discontinuous at the two upper corners; no smoothing is hidden.
"""
from time import perf_counter
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import solve
from .high_order import chebyshev_grid


def nodal_derivative(nodes):
    """Derivative matrix for the actual supplied nodes, including pressure nodes."""
    difference = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(difference, 1.0)
    weights = 1.0 / np.prod(difference, axis=1)
    derivative = weights[None, :] / weights[:, None] / difference
    np.fill_diagonal(derivative, 0.0)
    np.fill_diagonal(derivative, -derivative.sum(axis=1))
    return derivative


def cavity_operators(n=24, reynolds=100.0, lid="standard"):
    """Dense teaching operators, bounded to at most 2883 coupled unknowns.

    Even N includes the centre node for an unambiguous pressure gauge.
    A higher resolution should use a scalable implementation, not this dense lab.
    """
    if not isinstance(n, (int, np.integer)) or n % 2 or not 8 <= n <= 32:
        raise ValueError("Use even N from 8 to 32 in this CPU teaching implementation")
    if not np.isfinite(reynolds) or reynolds <= 0:
        raise ValueError("Re must be finite and positive")
    if lid not in ("standard", "smooth", "stationary"):
        raise ValueError("lid must be standard, smooth or stationary")
    x, d = chebyshev_grid(n)
    d2, m = d @ d, n - 1
    interior = np.s_[1:-1, 1:-1]
    eye_m = np.eye(m)
    dx = np.kron(eye_m, d[interior])
    dy = np.kron(d[interior], eye_m)
    lap = np.kron(eye_m, d2[interior]) + np.kron(d2[interior], eye_m)
    dp = nodal_derivative(x[1:-1])
    gx, gy = np.kron(eye_m, dp), np.kron(dp, eye_m)
    boundary_u = np.zeros((n + 1, n + 1))
    if lid == "standard":
        boundary_u[-1, 1:-1] = 1.0
    elif lid == "smooth":
        boundary_u[-1, :] = 16 * x**2 * (1-x)**2
    # Both corner velocities are zero; the open top edge moves in +x.
    return dict(n=n, m=m, size=m*m, x=x, d=d, d2=d2, dx=dx, dy=dy,
                lap=lap, gx=gx, gy=gy, reynolds=float(reynolds), lid=lid,
                boundary_u=boundary_u, gauge=(m*m)//2,
                boundary_ux=(boundary_u @ d.T)[interior].ravel(),
                boundary_uy=(d @ boundary_u)[interior].ravel(),
                boundary_lap=(d2 @ boundary_u + boundary_u @ d2.T)[interior].ravel())


def cavity_residual(state, ops, forcing=None):
    """All steady momentum equations plus continuity; one redundant row sets p_c=0."""
    u, v, p = np.split(state, 3)
    dx, dy, lap = ops["dx"], ops["dy"], ops["lap"]
    ux, uy = dx @ u + ops["boundary_ux"], dy @ u + ops["boundary_uy"]
    vx, vy = dx @ v, dy @ v
    ru = -(lap @ u + ops["boundary_lap"]) / ops["reynolds"] + u*ux + v*uy + ops["gx"] @ p
    rv = -(lap @ v) / ops["reynolds"] + u*vx + v*vy + ops["gy"] @ p
    if forcing is not None:
        ru, rv = ru - forcing[0], rv - forcing[1]
    continuity = dx @ u + dy @ v
    continuity[ops["gauge"]] = p[ops["gauge"]]
    return np.concatenate((ru, rv, continuity))


def cavity_jacobian(state, ops):
    """Exact derivative of the discrete advective-form residual for Newton's method."""
    u, v, _ = np.split(state, 3)
    dx, dy = ops["dx"], ops["dy"]
    ux, uy = dx @ u + ops["boundary_ux"], dy @ u + ops["boundary_uy"]
    vx, vy = dx @ v, dy @ v
    a = -ops["lap"] / ops["reynolds"] + u[:, None]*dx + v[:, None]*dy
    bx, by = dx.copy(), dy.copy()
    bx[ops["gauge"], :] = 0.0
    by[ops["gauge"], :] = 0.0
    pp = np.zeros_like(dx)
    pp[ops["gauge"], ops["gauge"]] = 1.0
    return np.block([[a+np.diag(ux), np.diag(uy), ops["gx"]],
                     [np.diag(vx), a+np.diag(vy), ops["gy"]],
                     [bx, by, pp]])


def cavity_sample(result, field, x_query, y_query):
    """Evaluate the saved polynomial; arrays and results use field[y,x].

    The pressure polynomial has its own N-1 nodes, not the velocity's N+1 nodes.
    Evaluation at the walls is a polynomial continuation, not pressure wall data.
    """
    nodes = result["x"][1:-1] if field == "p" else result["x"]
    identity = np.eye(len(nodes))
    interpolation_x = BarycentricInterpolator(nodes, identity, axis=0)(np.atleast_1d(x_query))
    interpolation_y = BarycentricInterpolator(nodes, identity, axis=0)(np.atleast_1d(y_query))
    return interpolation_y @ result[field] @ interpolation_x.T


def solve_spectral_cavity(n=24, reynolds=100.0, lid="standard", tolerance=1e-9,
                          max_iterations=25, forcing=None):
    """Solve the steady incompressible cavity from scratch with damped Newton.

    There is no physical time march and no fitted Ghia data. A converged flag
    certifies the algebraic criterion only. Re=100/400 are the classroom cases;
    other Re require their own resolution/convergence evidence.
    forcing is reserved for manufactured verification, absent in the driven cavity.
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Use positive tolerance and iteration budget")
    started = perf_counter()
    ops = cavity_operators(n, reynolds, lid)
    state = np.zeros(3 * ops["size"])
    history, reason = [], "iteration budget reached"
    for iteration in range(max_iterations + 1):
        residual = cavity_residual(state, ops, forcing)
        norm = float(np.max(np.abs(residual)))
        u, v, _ = np.split(state, 3)
        # Include the continuity equation removed to impose the gauge.
        divergence = float(np.max(np.abs(ops["dx"] @ u + ops["dy"] @ v)))
        history.append([iteration, norm, divergence, 0.0])
        if not np.isfinite(norm):
            reason = "nonfinite residual"
            break
        if max(norm, divergence) <= tolerance:
            reason = "converged discrete steady equations"
            break
        if iteration == max_iterations:
            break
        delta = solve(cavity_jacobian(state, ops), -residual, assume_a="gen", check_finite=False)
        damping, accepted = 1.0, False
        for _ in range(15):
            trial = state + damping * delta
            trial_norm = np.max(np.abs(cavity_residual(trial, ops, forcing)))
            if np.isfinite(trial_norm) and trial_norm <= (1 - 1e-4*damping) * norm:
                state, accepted = trial, True
                break
            damping *= 0.5
        history[-1][3] = damping if accepted else 0.0
        if not accepted:
            reason = "line search failed; no convergence claimed"
            break
    u, v, p = np.split(state, 3)
    uf, vf = ops["boundary_u"].copy(), np.zeros_like(ops["boundary_u"])
    uf[1:-1, 1:-1], vf[1:-1, 1:-1] = u.reshape(ops["m"], ops["m"]), v.reshape(ops["m"], ops["m"])
    result = dict(n=n, reynolds=float(reynolds), lid=lid, x=ops["x"],
                  u=uf, v=vf, p=p.reshape(ops["m"], ops["m"]),
                  omega=vf @ ops["d"].T - ops["d"] @ uf,
                  divergence=uf @ ops["d"].T + ops["d"] @ vf,
                  history=np.asarray(history), iterations=len(history)-1,
                  residual_max=history[-1][1], divergence_nodes_max=history[-1][2],
                  converged=(reason == "converged discrete steady equations"), stop_reason=reason,
                  state=state, tolerance=float(tolerance),
                  solve_seconds=perf_counter()-started)
    # A fixed off-grid core, disclosed separately from the singular corner zones.
    query = np.linspace(0.05, 0.95, 101)
    core_div = cavity_sample(result, "divergence", query, query)
    result["divergence_core_rms"] = float(np.sqrt(np.mean(core_div**2)))
    result["divergence_core_max"] = float(np.max(np.abs(core_div)))
    return result
