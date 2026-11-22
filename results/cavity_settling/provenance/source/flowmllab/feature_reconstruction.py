"""Small, original CPU teaching models for Weeks 11 and 12.

Manufactured fields and Gaussian sampling analogs are NOT CFD/DSMC results.
No research checkpoint, MambaIR model or ShockVortexML implementation is copied.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


def diagnostics(u, v, dx, dy):
    """Planar velocity-gradient diagnostics; not objective under rotating frames."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != v.shape or u.ndim != 2 or min(u.shape) < 3:
        raise ValueError("Use matching 2-D fields with at least three nodes per axis")
    if dx <= 0 or dy <= 0 or not np.isfinite([u, v]).all():
        raise ValueError("Require positive spacing and finite fields")
    uy, ux = np.gradient(u, dy, dx, edge_order=2)
    vy, vx = np.gradient(v, dy, dx, edge_order=2)
    div = ux + vy
    omega = vx - uy
    # Q of the trace-free in-plane gradient; lambda_ci^2=max(Qd,0) in 2-D.
    qd = -0.25 * (ux - vy)**2 - uy * vx
    return dict(div=div, omega=omega, qd=qd,
                swirl=np.sqrt(np.maximum(qd, 0)), compression=np.maximum(-div, 0))


def manufactured_case(seed, size=56):
    """Analytic compression layer + vortex + shear; no governing PDE solution."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, x)
    front = rng.uniform(-.4, .4) + rng.uniform(-.22, .22) * Y
    width = rng.uniform(.055, .10)
    cx, cy = rng.uniform(-.45, .45, 2)
    radius = rng.uniform(.20, .30)
    r2 = (X-cx)**2 + (Y-cy)**2
    envelope = np.exp(-r2/(2*radius**2))
    rotation = rng.choice([-1, 1]) * rng.uniform(2.8, 4.2)
    layer = np.tanh((X-front)/width)
    u = 1 - .35*layer - rotation*(Y-cy)*envelope + .25*Y
    v = rotation*(X-cx)*envelope
    p = 1 + .35*(1+layer)
    d = diagnostics(u, v, x[1]-x[0], x[1]-x[0])
    py, px = np.gradient(p, x[1]-x[0], x[1]-x[0], edge_order=2)
    features = np.stack([u, v, p, d['compression'], d['omega'],
                         d['qd'], d['swirl'], np.hypot(px, py)], axis=-1)
    # Independent construction labels, not outputs from the detector thresholds.
    labels = np.stack([abs(X-front) < width, r2 < (.70*radius)**2], axis=-1)
    return dict(x=x, u=u, v=v, p=p, features=features, labels=labels, **d)


def dice(reference, prediction):
    a, b = np.asarray(reference, bool), np.asarray(prediction, bool)
    if a.shape != b.shape:
        raise ValueError("Mask shapes differ")
    den = a.sum() + b.sum()
    return float(2*np.count_nonzero(a & b)/den) if den else 1.0


def choose_thresholds(cases, scores, grid):
    """Validation-only family-macro Dice; one frozen threshold per label."""
    return np.array([max(grid, key=lambda t: np.mean([
        dice(c['labels'][..., k], p[..., k] >= t)
        for c, p in zip(cases, scores)])) for k in range(2)])


def physical_scores(case):
    """Simple non-neural baseline; thresholds must be selected on development data."""
    return np.stack([case['compression'], case['swirl']], axis=-1)


def sampling_case(seed, size=48, blocks=10, offset=0.):
    """Independent Gaussian noisy-block analog of a scalar heat-flux pattern."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, x)
    truth = ((1+rng.uniform(-.18, .18))*np.sin(np.pi*X)*np.sin(2*np.pi*Y)
             + .2*np.cos(3*np.pi*X)*np.sin(np.pi*Y) + offset)
    sigma = .65*(1+.6*X)
    samples = truth + rng.normal(size=(blocks, size, size))*sigma
    # Separate random draws: this reference shares NO blocks with the observation.
    reference = truth + rng.normal(size=truth.shape)*sigma/np.sqrt(80)
    return dict(truth=truth, blocks=samples, reference=reference, x=x)


def fit_spectral_prior(cases, budget=3):
    """Pedagogical development-only prior/gain, not the paper's MambaIR network."""
    if len(cases) < 2 or budget < 1:
        raise ValueError("Need two development cases and a positive budget")
    prior = np.mean([c['reference'] for c in cases], axis=0)
    z = np.stack([dctn(c['reference']-prior, norm='ortho') for c in cases])
    signal = np.var(z, axis=0, ddof=1)
    # Transform spatial axes only; estimate single-block noise on development draws.
    noise = np.mean([np.var(dctn(c['blocks'], axes=(-2,-1), norm='ortho'),
                                  axis=0, ddof=1) for c in cases], axis=0)
    gain = signal/(signal+noise/budget+1e-15)
    gain[0, 0] = 1.
    return prior, gain


def reconstruct(observation, prior, gain):
    observation, prior, gain = map(np.asarray, (observation, prior, gain))
    if observation.shape != prior.shape or prior.shape != gain.shape:
        raise ValueError("Incompatible observation, prior or gain")
    if not all(np.isfinite(a).all() for a in (observation, prior, gain)):
        raise ValueError("Nonfinite input")
    if np.any((gain < 0) | (gain > 1)):
        raise ValueError("Gain must be bounded between zero and one")
    result = prior + idctn(gain*dctn(observation-prior, norm='ortho'), norm='ortho')
    # Equal-area classroom grid. Native nonuniform cells require area weights.
    return result + observation.mean() - result.mean()


def nrmse(prediction, reference, area=None):
    a, b = np.asarray(prediction), np.asarray(reference)
    w = np.ones_like(b) if area is None else np.asarray(area)
    if a.shape != b.shape or w.shape != b.shape or np.any(w <= 0):
        raise ValueError("Invalid fields or positive cell areas")
    if not all(np.isfinite(v).all() for v in (a, b, w)):
        raise ValueError("Nonfinite fields")
    den = np.sum(w*b*b)
    if den <= 0:
        raise ValueError("Zero reference norm")
    return float(np.sqrt(np.sum(w*(a-b)**2)/den))


def additive_moments(velocity):
    """Unit-weight molecular velocities, three velocity components."""
    a = np.asarray(velocity, float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Expected finite nonempty N x 3 molecular velocities")
    speed2 = np.sum(a*a, axis=1)
    return dict(count=float(len(a)), first=a.sum(axis=0), second=a.T@a,
                energy=speed2.sum(), flux=(speed2[:, None]*a).sum(axis=0))


def central_heat_numerator(raw):
    """J_i before common molecular-mass/volume/time normalization."""
    u = raw['first']/raw['count']
    return (raw['flux'] - u*raw['energy'] - 2*raw['second']@u
            + 2*raw['count']*u*np.dot(u, u))


def support_score(observation, prior, noise_variance):
    """One-field residual/noise heuristic, NOT the paper's 27-zone monitor."""
    if noise_variance <= 0:
        raise ValueError("Positive development noise variance required")
    return float(np.mean((np.asarray(observation)-prior)**2)/noise_variance)
