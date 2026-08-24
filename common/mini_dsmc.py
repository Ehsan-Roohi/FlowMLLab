"""Lightweight teaching DSMC-style cavity for Week-5 Track 5.

This is deliberately a pedagogical, CPU-friendly model. It is useful for
studying statistical noise, case-wise splitting, label averaging, and ML
workflow design. It is NOT a validation-quality production DSMC solver.

Version 1.2 pools particle counts, first moments, and second moments over the
entire sampling window before computing temperature.  This avoids the large
small-sample temperature bias that results from computing a cell variance at
each instant with only a few particles and then averaging those biased
instantaneous estimates.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np

MINI_DSMC_VERSION = "1.2"


@dataclass
class MiniDSMCConfig:
    Kn: float = 0.1
    Uwall: float = 0.6
    nx: int = 12
    ny: int = 12
    ppc: int = 6
    # The original short teaching run began sampling before a reasonably
    # developed lid-driven signal emerged. Version 1.2 retains the same number
    # of stored field samples but moves the window later in time, reducing the
    # Kn-dependent transient contamination that matters especially in Variant 5B.
    steps: int = 3200
    sample_start: int = 2400
    sample_stride: int = 5
    dt: float = 0.003
    T_wall: float = 1.0
    seed: int = 11

    @property
    def n_particles(self):
        return self.nx * self.ny * self.ppc

    @property
    def mean_thermal_speed(self):
        return np.sqrt(8 * self.T_wall / np.pi)

    @property
    def total_time(self):
        return self.steps * self.dt

    @property
    def sampling_start_time(self):
        return self.sample_start * self.dt


def apply_diffuse_walls(pos, vel, cfg, rng):
    T = cfg.T_wall
    sigma = np.sqrt(T)
    left = pos[:, 0] < 0
    if np.any(left):
        pos[left, 0] *= -1
        n = int(left.sum())
        vel[left, 0] = np.sqrt(-2 * T * np.log(np.maximum(rng.random(n), 1e-15)))
        vel[left, 1:] = sigma * rng.standard_normal((n, 2))
    right = pos[:, 0] > 1
    if np.any(right):
        pos[right, 0] = 2 - pos[right, 0]
        n = int(right.sum())
        vel[right, 0] = -np.sqrt(-2 * T * np.log(np.maximum(rng.random(n), 1e-15)))
        vel[right, 1:] = sigma * rng.standard_normal((n, 2))
    bottom = pos[:, 1] < 0
    if np.any(bottom):
        pos[bottom, 1] *= -1
        n = int(bottom.sum())
        vel[bottom, 0] = sigma * rng.standard_normal(n)
        vel[bottom, 1] = np.sqrt(-2 * T * np.log(np.maximum(rng.random(n), 1e-15)))
        vel[bottom, 2] = sigma * rng.standard_normal(n)
    top = pos[:, 1] > 1
    if np.any(top):
        pos[top, 1] = 2 - pos[top, 1]
        n = int(top.sum())
        vel[top, 0] = cfg.Uwall + sigma * rng.standard_normal(n)
        vel[top, 1] = -np.sqrt(-2 * T * np.log(np.maximum(rng.random(n), 1e-15)))
        vel[top, 2] = sigma * rng.standard_normal(n)


def collide_pair(v1, v2, rng):
    cm = 0.5 * (v1 + v2)
    g = np.linalg.norm(v1 - v2)
    mu = 2 * rng.random() - 1
    phi = 2 * np.pi * rng.random()
    st = np.sqrt(max(0.0, 1 - mu * mu))
    d = np.array([st * np.cos(phi), st * np.sin(phi), mu])
    return cm + 0.5 * g * d, cm - 0.5 * g * d


def collide_cells(pos, vel, cfg, rng):
    ix = np.minimum((pos[:, 0] * cfg.nx).astype(int), cfg.nx - 1)
    iy = np.minimum((pos[:, 1] * cfg.ny).astype(int), cfg.ny - 1)
    cid = ix + cfg.nx * iy
    order = np.argsort(cid)
    sid = cid[order]
    _, starts, counts = np.unique(sid, return_index=True, return_counts=True)
    for start, count in zip(starts, counts):
        if count < 2:
            continue
        expected = 0.5 * count * cfg.mean_thermal_speed * cfg.dt / max(cfg.Kn, 1e-12)
        npairs = int(expected + rng.random())
        ids = order[start:start + count]
        for _ in range(npairs):
            a, b = rng.integers(0, count, size=2)
            if a == b:
                b = (b + 1) % count
            i, j = ids[a], ids[b]
            vel[i], vel[j] = collide_pair(vel[i], vel[j], rng)


def cell_moment_sums(pos, vel, cfg):
    """Return raw cell moments for one stored particle realization."""
    ix = np.minimum((pos[:, 0] * cfg.nx).astype(int), cfg.nx - 1)
    iy = np.minimum((pos[:, 1] * cfg.ny).astype(int), cfg.ny - 1)
    cid = ix + cfg.nx * iy
    nc = cfg.nx * cfg.ny
    count = np.bincount(cid, minlength=nc).astype(float)
    sx = np.bincount(cid, weights=vel[:, 0], minlength=nc)
    sy = np.bincount(cid, weights=vel[:, 1], minlength=nc)
    sz = np.bincount(cid, weights=vel[:, 2], minlength=nc)
    s2 = np.bincount(cid, weights=np.sum(vel * vel, axis=1), minlength=nc)
    return count, sx, sy, sz, s2


def fields_from_pooled_moments(moment_sums, cfg):
    """Convert sampling-window raw moments to fields.

    Temperature is computed from all particle observations accumulated in a
    cell during the sampling window.  A Bessel correction is applied to the
    pooled central second moment.  With a few hundred effective observations
    per cell, the residual finite-sample bias is below the percent level in an
    equilibrium sanity test.  Time correlation can still reduce the effective
    sample size; this teaching solver does not estimate an autocorrelation time.
    """
    count, sx, sy, sz, s2 = [np.asarray(a, dtype=float) for a in moment_sums]
    safe = np.maximum(count, 1.0)
    u, v, w = sx / safe, sy / safe, sz / safe
    centered_s2 = s2 - (sx * sx + sy * sy + sz * sz) / safe
    dof = np.maximum(count - 1.0, 1.0)
    T = np.maximum(centered_s2 / (3.0 * dof), 1e-10)

    # Density is proportional to the pooled particle count. Any common
    # sampling-window factor cancels when normalizing by the domain mean.
    rho = count / np.mean(count)
    shape = (cfg.ny, cfg.nx)
    return u.reshape(shape), v.reshape(shape), T.reshape(shape), rho.reshape(shape)


def instantaneous_fields(pos, vel, cfg):
    """Return one-snapshot fields using a Bessel-corrected cell variance.

    This helper is retained for demonstrations.  Production labels in
    :func:`run_case` use pooled moments over the whole sampling window.
    """
    return fields_from_pooled_moments(cell_moment_sums(pos, vel, cfg), cfg)


def run_case(cfg, progress=False):
    if cfg.sample_start >= cfg.steps:
        raise ValueError("sample_start must be smaller than steps")
    if cfg.sample_stride <= 0 or cfg.ppc <= 0:
        raise ValueError("sample_stride and ppc must be positive")

    rng = np.random.default_rng(cfg.seed)
    pos = rng.random((cfg.n_particles, 2))
    vel = np.sqrt(cfg.T_wall) * rng.standard_normal((cfg.n_particles, 3))
    vel -= vel.mean(axis=0)

    nc = cfg.nx * cfg.ny
    pooled = [np.zeros(nc, dtype=float) for _ in range(5)]
    ns = 0
    t0 = time.time()
    for step in range(cfg.steps):
        pos += vel[:, :2] * cfg.dt
        apply_diffuse_walls(pos, vel, cfg, rng)
        collide_cells(pos, vel, cfg, rng)
        if step >= cfg.sample_start and (step - cfg.sample_start) % cfg.sample_stride == 0:
            for acc, value in zip(pooled, cell_moment_sums(pos, vel, cfg)):
                acc += value
            ns += 1
        if progress and (step + 1) % max(1, cfg.steps // 4) == 0:
            print(step + 1, cfg.steps, ns)

    fields = fields_from_pooled_moments(pooled, cfg)
    x = (np.arange(cfg.nx) + 0.5) / cfg.nx
    y = (np.arange(cfg.ny) + 0.5) / cfg.ny
    counts = pooled[0].reshape(cfg.ny, cfg.nx)
    return {
        "x": x,
        "y": y,
        "u": fields[0],
        "v": fields[1],
        "T": fields[2],
        "rho": fields[3],
        "field_samples": ns,
        "mean_particle_observations_per_cell": float(np.mean(counts)),
        "temperature_estimator": "pooled central second moment with Bessel correction",
        "sampling_start_time": cfg.sampling_start_time,
        "sampling_end_time": cfg.total_time,
        "sampling_window_time": cfg.total_time - cfg.sampling_start_time,
        "transient_note": "finite-window teaching data; not an asymptotic validation reference",
        "runtime_s": time.time() - t0,
        "config": asdict(cfg),
        "mini_dsmc_version": MINI_DSMC_VERSION,
    }
