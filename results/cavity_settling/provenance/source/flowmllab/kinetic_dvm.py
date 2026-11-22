"""Independent CPU finite-volume reference for the Week 10.2 reduced BGK tube.

This solver has no neural-network dependency. It evolves g, h and k with
MUSCL/minmod transport and SSPRK2. Transverse Gaussian moments are integrated
analytically, while axial velocity uses a finite Gauss-Legendre quadrature.
It is a teaching reference, not the paper's original reference implementation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

FIELDS = ('rho', 'u', 'T', 'q', 'sigma', 'm', 'R', 'Delta', 'S')


@dataclass(frozen=True)
class DVMConfig:
    cells: int = 400
    velocity_order: int = 48
    vmax: float = 7.0
    final_time: float = 0.02
    tau: float = 1.0
    cfl: float = 0.4

    def validate(self):
        for name, low, high in (('cells', 16, 3200), ('velocity_order', 8, 128)):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        if not np.isfinite([self.vmax, self.final_time, self.tau, self.cfl]).all():
            raise ValueError('All controls must be finite')
        if not (0 < self.vmax <= 20 and 0 <= self.final_time <= .1 and self.tau > 0 and 0 < self.cfl <= .45):
            raise ValueError('Require 0<vmax<=20, 0<=final_time<=.1, tau>0, 0<cfl<=.45')


def initial_macros(x):
    x = np.asarray(x, dtype=np.float64)
    blend = .5 * (1 + np.tanh(20 * x))
    return 1 - .875 * blend, np.zeros_like(x), 1 - .2 * blend


def maxwellian_reductions(v, rho, u, temperature):
    """Return [cell, g/h/k, velocity]; h has no factor of one half."""
    rho, u, temperature = np.broadcast_arrays(rho, u, temperature)
    if np.any(rho <= 0) or np.any(temperature <= 0):
        raise FloatingPointError('Nonpositive macroscopic state; no clipping is applied')
    c = np.asarray(v)[None, :] - np.atleast_1d(u)[:, None]
    temp = np.atleast_1d(temperature)[:, None]
    g = np.atleast_1d(rho)[:, None] * np.exp(-c*c/(2*temp))/np.sqrt(2*np.pi*temp)
    return np.stack((g, 2*temp*g, 8*temp*temp*g), axis=1)


def moments(state, v, w):
    state = np.asarray(state, dtype=np.float64)
    g, h, k = state[:, 0], state[:, 1], state[:, 2]
    rho = np.sum(g*w, axis=-1)
    if np.any(rho <= 0) or not np.isfinite(state).all():
        raise FloatingPointError('Invalid reduced distribution; no repair is applied')
    u = np.sum(g*v*w, axis=-1)/rho
    c = v[None, :] - u[:, None]
    temp = np.sum((c*c*g+h)*w, axis=-1)/(3*rho)
    if np.any(temp <= 0):
        raise FloatingPointError('Nonpositive temperature; no repair is applied')
    sigma = np.sum((2*c*c*g-h)*w/3, axis=-1)
    q = .5*np.sum((c**3*g+c*h)*w, axis=-1)
    m = np.sum((2*c**3*g-3*c*h)*w/5, axis=-1)
    r = np.sum((2*c**4*g+c*c*h-k)*w/3, axis=-1)-7*temp*sigma
    delta = np.sum((c**4*g+2*c*c*h+k)*w, axis=-1)-15*rho*temp**2
    return dict(zip(FIELDS, (rho, u, temp, q, sigma, m, r, delta, r+delta/3)))


def _macros(state, v, w):
    g, h = state[:, 0], state[:, 1]
    rho = np.sum(g*w, axis=-1)
    if np.any(rho <= 0) or not np.isfinite(state).all():
        raise FloatingPointError('Nonpositive density or nonfinite DVM stage')
    u = np.sum(g*v*w, axis=-1)/rho
    temp = np.sum(((v[None, :]-u[:, None])**2*g+h)*w, axis=-1)/(3*rho)
    if np.any(temp <= 0):
        raise FloatingPointError('Nonpositive temperature at a DVM stage')
    return rho, u, temp


def conserved(state, v, w):
    g, h = state[:, 0], state[:, 1]
    return np.stack((np.sum(g*w, axis=-1), np.sum(g*v*w, axis=-1),
                     .5*np.sum((v*v*g+h)*w, axis=-1)), axis=-1)


def _minmod(left, right):
    return np.where(left*right > 0, np.sign(left)*np.minimum(np.abs(left), np.abs(right)), 0.)


def _rhs(state, v, w, dx, tau, left, right):
    padded = np.empty((len(state)+4, 3, len(v)), dtype=np.float64)
    padded[2:-2] = state
    padded[:2] = np.where(v > 0, left, state[0])
    padded[-2:] = np.where(v < 0, right, state[-1])
    slopes = np.zeros_like(padded)
    slopes[1:-1] = _minmod(padded[1:-1]-padded[:-2], padded[2:]-padded[1:-1])
    face_left = padded[1:-2] + .5*slopes[1:-2]
    face_right = padded[2:-1] - .5*slopes[2:-1]
    # Exact incoming reservoir value at the boundary face, not a limited ghost extrapolation.
    face_left[0][:, v > 0] = left[:, v > 0]
    face_right[-1][:, v < 0] = right[:, v < 0]
    flux = np.maximum(v, 0)*face_left + np.minimum(v, 0)*face_right
    target = maxwellian_reductions(v, *_macros(state, v, w))
    collision = (target-state)/tau
    rhs = -(flux[1:]-flux[:-1])/dx + collision
    boundary_rate = conserved(flux[[0, -1]], v, w)
    collision_rate = dx*conserved(collision, v, w).sum(axis=0)
    return rhs, boundary_rate[0]-boundary_rate[1], collision_rate


def solve(config: DVMConfig = DVMConfig()):
    config.validate()
    started = time.perf_counter()
    z, weights = leggauss(config.velocity_order)
    v, w = config.vmax*z, config.vmax*weights
    dx = 1./config.cells
    x = -.5 + (np.arange(config.cells)+.5)*dx
    state = maxwellian_reductions(v, *initial_macros(x))
    left = maxwellian_reductions(v, 1., 0., 1.)[0]
    right = maxwellian_reductions(v, .125, 0., .8)[0]
    initial_total = dx*conserved(state, v, w).sum(axis=0)
    boundary_total = np.zeros(3)
    collision_total = np.zeros(3)
    # A combined transport/collision restriction, conservative for positivity.
    dt_limit = config.cfl/(np.max(np.abs(v))/dx + 1/config.tau)
    steps = int(np.ceil(config.final_time/dt_limit)) if config.final_time else 0
    dt = config.final_time/steps if steps else 0.
    minima = state.min(axis=(0, 2))
    initial_rho, _, initial_T = _macros(state, v, w)
    minimum_rho, minimum_T = float(initial_rho.min()), float(initial_T.min())
    for _ in range(steps):
        f0, b0, c0 = _rhs(state, v, w, dx, config.tau, left, right)
        stage = state + dt*f0
        if np.min(stage) < -1e-14:
            raise FloatingPointError('Negative reduced population at RK stage; no clipping')
        stage_rho, _, stage_T = _macros(stage, v, w)
        f1, b1, c1 = _rhs(stage, v, w, dx, config.tau, left, right)
        state = .5*state + .5*(stage+dt*f1)
        if np.min(state) < -1e-14:
            raise FloatingPointError('Negative reduced population at RK endpoint; no clipping')
        end_rho, _, end_T = _macros(state, v, w)
        minimum_rho = min(minimum_rho, float(stage_rho.min()), float(end_rho.min()))
        minimum_T = min(minimum_T, float(stage_T.min()), float(end_T.min()))
        minima = np.minimum(minima, np.minimum(stage.min(axis=(0, 2)), state.min(axis=(0, 2))))
        boundary_total += .5*dt*(b0+b1)
        collision_total += .5*dt*(c0+c1)
    fields = moments(state, v, w)
    totals = dx*conserved(state, v, w).sum(axis=0)
    realizability = state[:, 0]*state[:, 2] - state[:, 1]**2
    scale = max(float(np.max(np.abs(state[:, 0]*state[:, 2]))), np.finfo(float).tiny)
    diagnostics = {
        'steps': steps, 'dt': dt, 'actual_transport_cfl': dt*np.max(np.abs(v))/dx,
        'initial_conserved_totals': initial_total.tolist(), 'final_conserved_totals': totals.tolist(),
        'integrated_boundary_rates': boundary_total.tolist(),
        'integrated_discrete_collision_rates': collision_total.tolist(),
        'mass_momentum_energy_balance_defect': (totals-initial_total-boundary_total).tolist(),
        'algebraic_balance_defect_after_collision_quadrature': (totals-initial_total-boundary_total-collision_total).tolist(),
        'minimum_g_h_k_over_stages': minima.tolist(),
        'minimum_rho': minimum_rho, 'minimum_T': minimum_T,
        'final_transverse_realizability_negative_defect_over_global_gk': float(max(0., -realizability.min())/scale),
        'clipping_applied': False, 'discrete_collision_conservation_correction': False,
        'seconds': time.perf_counter()-started,
        'method': 'MUSCL minmod finite volume; SSPRK2; analytic transverse moments',
        'reference_scope': 'new independent reduced-BGK teaching DVM; no formal convergence certificate',
    }
    return {'x': x, 'v': v, 'weights': w, 'state': state, 'columns': FIELDS,
            'fields': np.column_stack([fields[f] for f in FIELDS]), **fields,
            'config': asdict(config), 'diagnostics': diagnostics}
