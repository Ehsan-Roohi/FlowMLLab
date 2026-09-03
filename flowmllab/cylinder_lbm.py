"""Educational D2Q9 lattice--Boltzmann solver for flow past a cylinder.

The implementation is intentionally small enough to read in a notebook while
retaining the safeguards needed for a reproducible teaching benchmark:

* D2Q9 equilibrium with stable two-relaxation-time (TRT) collision by default,
  plus BGK as an explicit comparison option;
* a low-Mach Zou--He velocity inlet and a non-reflecting-style convective outlet;
* Bouzidi interpolated bounce-back on the analytical circular cylinder, with
  halfway bounce-back retained as an explicit comparison option; and
* a periodic transverse boundary, representing an array with a deliberately
  small blockage ratio (rather than adding artificial channel-wall layers).
* a three-grid Richardson/GCI diagnostic that refuses non-monotone sequences.

All lengths and times are in lattice units.  With cylinder diameter ``D`` and
inlet speed ``U``, ``nu = U*D/Re`` and ``tau = 1/2 + 3*nu``.  Pressure is the
weakly-compressible LBM gauge pressure ``c_s**2 * (rho-rho0)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


CYLINDER_LBM_VERSION = "0.4.0"
CS2 = 1.0 / 3.0

# Direction order: rest, E, N, W, S, NE, NW, SW, SE.
LATTICE_VELOCITIES = np.asarray(
    ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
     (1, 1), (-1, 1), (-1, -1), (1, -1)),
    dtype=np.int8,
)
LATTICE_WEIGHTS = np.asarray(
    (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0),
    dtype=float,
)
OPPOSITE = np.asarray((0, 3, 4, 1, 2, 7, 8, 5, 6), dtype=np.int8)


@dataclass(frozen=True)
class CylinderLBMConfig:
    """Configuration for :func:`simulate_cylinder` in lattice units.

    The default transverse boundary is periodic.  Keep ``diameter/ny`` small
    (ideally at most 0.15) when approximating an unconfined cylinder wake.
    ``outlet_speed`` is the advection speed in the first-order convective
    distribution boundary; the inlet speed is a sensible default.
    """

    reynolds: float
    nx: int = 240
    ny: int = 80
    diameter: float = 10.0
    center_x: float | None = None
    center_y: float | None = None
    inflow_velocity: float = 0.05
    rho0: float = 1.0
    steps: int = 5000
    history_stride: int = 5
    snapshot_stride: int | None = None
    snapshot_start: int = 0
    statistics_start: int | None = None
    perturbation: float = 1.0e-3
    seed: int = 0
    startup_ramp_steps: int = 0
    transverse_boundary: str = "periodic"
    outlet_boundary: str = "convective"
    outlet_speed: float | None = None
    collision_model: str = "trt"
    cylinder_boundary: str = "bouzidi"
    trt_magic_parameter: float = 3.0 / 16.0

    @property
    def cylinder_center(self) -> tuple[float, float]:
        return (
            0.25 * self.nx if self.center_x is None else float(self.center_x),
            0.5 * (self.ny - 1) if self.center_y is None else float(self.center_y),
        )

    @property
    def viscosity(self) -> float:
        return float(self.inflow_velocity * self.diameter / self.reynolds)

    @property
    def relaxation_time(self) -> float:
        return 0.5 + self.viscosity / CS2

    @property
    def mach(self) -> float:
        return float(self.inflow_velocity / np.sqrt(CS2))

    @property
    def blockage_ratio(self) -> float:
        return float(self.diameter / self.ny)


def _validate_config(config: CylinderLBMConfig) -> None:
    if config.reynolds <= 0:
        raise ValueError("reynolds must be positive")
    if config.nx < 32 or config.ny < 20:
        raise ValueError("require nx >= 32 and ny >= 20")
    if config.diameter < 4.0:
        raise ValueError("diameter must span at least four lattice nodes")
    if config.inflow_velocity <= 0:
        raise ValueError("inflow_velocity must be positive")
    if config.rho0 <= 0:
        raise ValueError("rho0 must be positive")
    if config.steps < 1 or config.history_stride < 1:
        raise ValueError("require steps >= 1 and history_stride >= 1")
    if config.snapshot_stride is not None and config.snapshot_stride < 1:
        raise ValueError("snapshot_stride must be positive or None")
    if not 0 <= config.snapshot_start <= config.steps:
        raise ValueError("snapshot_start must lie between zero and steps")
    if config.statistics_start is not None and not 0 <= config.statistics_start <= config.steps:
        raise ValueError("statistics_start must lie between zero and steps")
    if config.perturbation < 0:
        raise ValueError("perturbation must be nonnegative")
    if config.startup_ramp_steps < 0 or config.startup_ramp_steps > config.steps:
        raise ValueError("startup_ramp_steps must lie between zero and steps")
    if config.transverse_boundary != "periodic":
        raise ValueError("this educational solver supports only periodic transverse BC")
    if config.outlet_boundary != "convective":
        raise ValueError("this educational solver supports only the convective outlet")
    if config.collision_model not in {"trt", "bgk"}:
        raise ValueError("collision_model must be 'trt' or 'bgk'")
    if config.cylinder_boundary not in {"halfway", "bouzidi"}:
        raise ValueError("cylinder_boundary must be 'halfway' or 'bouzidi'")
    if config.trt_magic_parameter <= 0:
        raise ValueError("trt_magic_parameter must be positive")
    if config.relaxation_time <= 0.5005:
        raise ValueError(
            f"BGK relaxation time tau={config.relaxation_time:.6g} is too close "
            "to 0.5; increase diameter or inflow_velocity, or reduce Reynolds"
        )
    if config.collision_model == "bgk" and config.relaxation_time < 0.53:
        raise ValueError(
            "BGK cylinder runs require tau >= 0.53 in this teaching solver; "
            "increase resolution/diameter or use the default TRT collision"
        )
    if config.mach > 0.10:
        raise ValueError(
            f"lattice Mach number {config.mach:.3f} exceeds the low-Mach gate 0.10"
        )
    cx, cy = config.cylinder_center
    radius = 0.5 * config.diameter
    if cx - radius < 4 or cx + radius > config.nx - 8:
        raise ValueError("cylinder must be separated from inlet and outlet")
    if cy - radius < 2 or cy + radius > config.ny - 3:
        raise ValueError("cylinder must be separated from its transverse periodic image")


def cylinder_mask(
    nx: int, ny: int, diameter: float, center: tuple[float, float]
) -> np.ndarray:
    """Return the node-centered circular solid mask with shape ``(ny, nx)``."""
    x = np.arange(nx, dtype=float)[None, :]
    y = np.arange(ny, dtype=float)[:, None]
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= (0.5 * diameter) ** 2


def curved_link_fractions(
    solid: np.ndarray,
    center: tuple[float, float],
    diameter: float,
) -> tuple[np.ndarray, ...]:
    """Return exact circle-intersection fractions for fluid-to-solid links.

    Each returned array has the same shape as ``solid``.  For lattice direction
    ``i``, finite entries identify a fluid node whose neighbor in direction
    ``i`` is solid.  The stored value ``q`` is the fraction of that lattice
    link between the fluid-node center and the analytical circular wall.  It
    lies in ``(0, 1]`` and is used by Bouzidi interpolated bounce-back.
    """
    mask = np.asarray(solid, dtype=bool)
    if mask.ndim != 2 or not mask.any() or mask.all():
        raise ValueError("solid must be a two-dimensional partial-domain mask")
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    cx0, cy0 = map(float, center)
    yy, xx = np.indices(mask.shape, dtype=float)
    x0 = xx - cx0
    y0 = yy - cy0
    radius2 = (0.5 * float(diameter)) ** 2
    fluid = ~mask
    fractions: list[np.ndarray] = []
    for direction, (cx, cy) in enumerate(LATTICE_VELOCITIES):
        values = np.full(mask.shape, np.nan, dtype=float)
        if direction:
            neighbor_solid = np.roll(
                np.roll(mask, -int(cy), axis=0), -int(cx), axis=1
            )
            link = fluid & neighbor_solid
            a = float(cx * cx + cy * cy)
            b = 2.0 * (x0 * float(cx) + y0 * float(cy))
            discriminant = b * b - 4.0 * a * (x0 * x0 + y0 * y0 - radius2)
            root = (-b - np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * a)
            valid = link & (discriminant >= -1.0e-12) & (root > 0.0) & (root <= 1.0 + 1.0e-12)
            if not np.array_equal(valid, link):
                raise FloatingPointError(
                    "failed to locate an analytical circle intersection on a boundary link"
                )
            values[link] = np.clip(root[link], np.finfo(float).eps, 1.0)
        fractions.append(values)
    return tuple(fractions)


def equilibrium(rho: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Return the second-order isothermal D2Q9 equilibrium distribution."""
    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if rho.shape != u.shape or rho.shape != v.shape:
        raise ValueError("rho, u, and v must have identical shapes")
    cu = (
        u[..., None] * LATTICE_VELOCITIES[:, 0]
        + v[..., None] * LATTICE_VELOCITIES[:, 1]
    )
    speed2 = u**2 + v**2
    return rho[..., None] * LATTICE_WEIGHTS * (
        1.0 + cu / CS2 + 0.5 * cu**2 / CS2**2 - 0.5 * speed2[..., None] / CS2
    )


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover density and velocity from a D2Q9 population array."""
    populations = np.asarray(f, dtype=float)
    if populations.ndim != 3 or populations.shape[-1] != 9:
        raise ValueError("f must have shape (ny, nx, 9)")
    rho = populations.sum(axis=-1)
    if np.any(rho <= 0) or not np.isfinite(rho).all():
        raise FloatingPointError("LBM density is non-positive or non-finite")
    momentum = np.einsum("...q,qd->...d", populations, LATTICE_VELOCITIES)
    return rho, momentum[..., 0] / rho, momentum[..., 1] / rho


def compute_vorticity(u: np.ndarray, v: np.ndarray, solid: np.ndarray | None = None) -> np.ndarray:
    """Compute ``omega_z = dv/dx-du/dy`` using second-order differences.

    The transverse derivative is explicitly periodic; streamwise boundary
    points use second-order one-sided differences through :func:`numpy.gradient`.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be two-dimensional arrays of equal shape")
    dvdx = np.gradient(v, axis=1, edge_order=2)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) * 0.5
    omega = dvdx - dudy
    if solid is not None:
        if solid.shape != u.shape:
            raise ValueError("solid mask must match velocity fields")
        omega = omega.copy()
        omega[solid] = 0.0
    return omega


def strouhal_diagnostics(
    time: np.ndarray,
    lift_coefficient: np.ndarray,
    diameter: float,
    inflow_velocity: float,
    transient_fraction: float = 0.5,
    strouhal_band: tuple[float, float] = (0.05, 0.5),
    minimum_cycles: float = 8.0,
    maximum_rms_change: float = 0.25,
    minimum_peak_to_median: float = 5.0,
    minimum_band_power_fraction: float = 0.01,
) -> dict[str, Any]:
    """Diagnose a physically admissible post-transient shedding frequency.

    The dominant peak is searched only inside ``strouhal_band``.  A result is
    accepted only when the retained interval spans ``minimum_cycles``, the lift
    RMS is stationary between its two halves, and the spectral peak is distinct
    from the band background.  Invalid signals return ``strouhal=nan`` and a
    machine-readable reason instead of reporting an acoustic or startup mode.
    """
    time = np.asarray(time, dtype=float).reshape(-1)
    lift = np.asarray(lift_coefficient, dtype=float).reshape(-1)
    if time.size != lift.size:
        raise ValueError("time and lift_coefficient must have equal length")
    if diameter <= 0 or inflow_velocity <= 0:
        raise ValueError("diameter and inflow_velocity must be positive")
    if not 0 <= transient_fraction < 1:
        raise ValueError("transient_fraction must be in [0, 1)")
    lower_st, upper_st = map(float, strouhal_band)
    if not 0 < lower_st < upper_st:
        raise ValueError("strouhal_band must contain two increasing positive values")
    if (
        minimum_cycles <= 0
        or maximum_rms_change < 0
        or minimum_peak_to_median <= 1
        or not 0 < minimum_band_power_fraction <= 1
    ):
        raise ValueError("invalid Strouhal quality-gate settings")
    diagnostics: dict[str, Any] = {
        "valid": False,
        "reason": "insufficient_samples",
        "strouhal": float("nan"),
        "band": [lower_st, upper_st],
        "minimum_cycles": float(minimum_cycles),
        "cycle_count": 0.0,
        "rms_first_half": float("nan"),
        "rms_second_half": float("nan"),
        "relative_rms_change": float("nan"),
        "peak_to_median": float("nan"),
        "band_power_fraction": float("nan"),
    }
    start = int(np.floor(transient_fraction * time.size))
    t = time[start:]
    signal = lift[start:]
    if t.size < 16 or not np.isfinite(t).all() or not np.isfinite(signal).all():
        return diagnostics
    dt = np.diff(t)
    if np.any(dt <= 0) or not np.allclose(dt, dt.mean(), rtol=1.0e-5, atol=1.0e-12):
        raise ValueError("time must be strictly increasing and uniformly sampled")
    signal = signal - signal.mean()
    split = signal.size // 2
    rms_first = float(np.sqrt(np.mean((signal[:split] - signal[:split].mean()) ** 2)))
    rms_second = float(np.sqrt(np.mean((signal[split:] - signal[split:].mean()) ** 2)))
    mean_rms = 0.5 * (rms_first + rms_second)
    relative_rms_change = abs(rms_second - rms_first) / max(
        mean_rms, np.finfo(float).eps
    )
    diagnostics.update(
        rms_first_half=rms_first,
        rms_second_half=rms_second,
        relative_rms_change=float(relative_rms_change),
    )
    if mean_rms < 1.0e-10:
        diagnostics["reason"] = "lift_signal_is_constant"
        return diagnostics
    # Remove residual linear drift before the FFT. Amplitude growth remains
    # visible to the independent RMS stationarity gate above.
    coordinate = np.linspace(-1.0, 1.0, signal.size)
    slope = float(np.dot(coordinate, signal) / np.dot(coordinate, coordinate))
    signal = signal - slope * coordinate
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    frequencies = np.fft.rfftfreq(signal.size, d=float(dt.mean()))
    strouhal_values = frequencies * float(diameter) / float(inflow_velocity)
    admissible = (strouhal_values >= lower_st) & (strouhal_values <= upper_st)
    admissible[0] = False
    if not np.any(admissible):
        diagnostics["reason"] = "no_fft_bin_inside_physical_band"
        return diagnostics
    indices = np.flatnonzero(admissible)
    peak = int(indices[np.argmax(spectrum[indices])])
    frequency = float(frequencies[peak])
    if 0 < peak < spectrum.size - 1:
        left, middle, right = np.log(np.maximum(spectrum[peak - 1:peak + 2], 1.0e-300))
        denominator = left - 2.0 * middle + right
        if denominator != 0.0:
            offset = 0.5 * (left - right) / denominator
            frequency += float(np.clip(offset, -0.5, 0.5)) * (frequencies[1] - frequencies[0])
    estimate = frequency * float(diameter) / float(inflow_velocity)
    duration = float(t[-1] - t[0])
    cycle_count = frequency * duration
    background = float(np.median(spectrum[indices]))
    peak_to_median = float(
        spectrum[peak] / max(background, np.finfo(float).tiny)
    )
    band_power_fraction = float(
        spectrum[indices].sum() / max(float(spectrum[1:].sum()), np.finfo(float).tiny)
    )
    diagnostics.update(
        raw_peak_strouhal=float(estimate),
        cycle_count=float(cycle_count),
        peak_to_median=peak_to_median,
        band_power_fraction=band_power_fraction,
        retained_duration=duration,
        retained_dimensionless_duration=float(
            duration * float(inflow_velocity) / float(diameter)
        ),
    )
    if band_power_fraction < minimum_band_power_fraction:
        diagnostics["reason"] = "dominant_energy_outside_physical_band"
    elif cycle_count < minimum_cycles:
        diagnostics["reason"] = "fewer_than_minimum_cycles"
    elif relative_rms_change > maximum_rms_change:
        diagnostics["reason"] = "lift_amplitude_not_stationary"
    elif peak_to_median < minimum_peak_to_median:
        diagnostics["reason"] = "spectral_peak_not_distinct"
    else:
        diagnostics.update(valid=True, reason="accepted", strouhal=float(estimate))
    return diagnostics


def estimate_strouhal(
    time: np.ndarray,
    lift_coefficient: np.ndarray,
    diameter: float,
    inflow_velocity: float,
    transient_fraction: float = 0.5,
    strouhal_band: tuple[float, float] = (0.05, 0.5),
    minimum_cycles: float = 8.0,
) -> float:
    """Return a quality-gated ``St=fD/U`` estimate or ``nan``."""
    return float(
        strouhal_diagnostics(
            time,
            lift_coefficient,
            diameter,
            inflow_velocity,
            transient_fraction=transient_fraction,
            strouhal_band=strouhal_band,
            minimum_cycles=minimum_cycles,
        )["strouhal"]
    )


def recirculation_length(
    u: np.ndarray,
    solid: np.ndarray,
    center: tuple[float, float],
    diameter: float,
) -> tuple[float, float]:
    """Return wake-centerline recirculation length in lattice units and ``D``.

    The length starts at the rear cylinder surface.  ``nan`` is returned when a
    closed negative-velocity interval cannot be identified inside the domain.
    """
    u = np.asarray(u, dtype=float)
    if u.shape != solid.shape:
        raise ValueError("u and solid mask must have equal shapes")
    cx, cy = center
    row = int(np.clip(round(cy), 0, u.shape[0] - 1))
    rear = cx + 0.5 * diameter
    start = max(int(np.floor(rear)) + 1, 1)
    line = u[row]
    candidates = np.flatnonzero((np.arange(u.shape[1]) >= start) & (~solid[row]))
    if candidates.size == 0 or line[candidates[0]] >= 0:
        return float("nan"), float("nan")
    downstream = candidates[candidates > candidates[0]]
    crossings = downstream[line[downstream] >= 0]
    if crossings.size == 0:
        return float("nan"), float("nan")
    right = int(crossings[0])
    left = right - 1
    denominator = line[right] - line[left]
    zero = float(right) if denominator == 0 else left - float(line[left]) / denominator
    length = max(0.0, zero - rear)
    return length, length / float(diameter)


def _apply_velocity_inlet(f: np.ndarray, speed: float) -> None:
    """Apply a uniform ``(speed, 0)`` Zou--He inlet at ``x=0`` in place."""
    boundary = f[:, 0, :]
    rho = (
        boundary[:, 0] + boundary[:, 2] + boundary[:, 4]
        + 2.0 * (boundary[:, 3] + boundary[:, 6] + boundary[:, 7])
    ) / (1.0 - speed)
    boundary[:, 1] = boundary[:, 3] + (2.0 / 3.0) * rho * speed
    boundary[:, 5] = (
        boundary[:, 7] + 0.5 * (boundary[:, 4] - boundary[:, 2])
        + (1.0 / 6.0) * rho * speed
    )
    boundary[:, 8] = (
        boundary[:, 6] + 0.5 * (boundary[:, 2] - boundary[:, 4])
        + (1.0 / 6.0) * rho * speed
    )


def _stream_and_bounce(
    post_collision: np.ndarray,
    solid: np.ndarray,
    curved_fractions: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, float, float]:
    """Stream populations and apply halfway or Bouzidi bounce-back."""
    streamed = np.empty_like(post_collision)
    fluid = ~solid
    force_x = 0.0
    force_y = 0.0
    link_masks: list[np.ndarray | None] = [None] * 9
    for q, (cx, cy) in enumerate(LATTICE_VELOCITIES):
        streamed[..., q] = np.roll(
            np.roll(post_collision[..., q], int(cy), axis=0), int(cx), axis=1
        )
        if q:
            neighbor_solid = np.roll(
                np.roll(solid, -int(cy), axis=0), -int(cx), axis=1
            )
            link = fluid & neighbor_solid
            link_masks[q] = link
            outgoing = post_collision[..., q][link]
            force_x += 2.0 * float(cx) * float(outgoing.sum())
            force_y += 2.0 * float(cy) * float(outgoing.sum())
    for q in range(1, 9):
        link = link_masks[q]
        outgoing = post_collision[..., q]
        if curved_fractions is None:
            reflected = outgoing[link]
        else:
            fraction = curved_fractions[q]
            if fraction.shape != solid.shape or not np.array_equal(np.isfinite(fraction), link):
                raise ValueError("curved boundary fractions do not match fluid-solid links")
            q_link = fraction[link]
            reflected = np.empty_like(q_link)
            near = q_link < 0.5
            if np.any(near):
                cx, cy = LATTICE_VELOCITIES[q]
                behind = np.roll(
                    np.roll(outgoing, int(cy), axis=0), int(cx), axis=1
                )[link]
                reflected[near] = (
                    2.0 * q_link[near] * outgoing[link][near]
                    + (1.0 - 2.0 * q_link[near]) * behind[near]
                )
            if np.any(~near):
                opposite = post_collision[..., int(OPPOSITE[q])][link]
                reflected[~near] = (
                    outgoing[link][~near] / (2.0 * q_link[~near])
                    + (2.0 * q_link[~near] - 1.0)
                    * opposite[~near]
                    / (2.0 * q_link[~near])
                )
        streamed[..., int(OPPOSITE[q])][link] = reflected
        if curved_fractions is not None:
            cx, cy = LATTICE_VELOCITIES[q]
            # Momentum exchange equals outgoing plus reflected link momentum.
            force_x += float(cx) * float((reflected - outgoing[link]).sum())
            force_y += float(cy) * float((reflected - outgoing[link]).sum())
    return streamed, force_x, force_y


def _collide(
    f: np.ndarray,
    feq: np.ndarray,
    relaxation_time: float,
    model: str,
    magic_parameter: float,
) -> np.ndarray:
    """Apply BGK or TRT collision.

    TRT retains the same kinematic viscosity as BGK in its symmetric moments,
    while relaxing antisymmetric non-equilibrium moments separately.  The
    conventional magic parameter ``(tau+-.5)*(tau--.5)=3/16`` substantially
    improves bounce-back stability at the small viscosities used for cylinder
    shedding, without changing the Navier--Stokes limit.
    """
    omega_plus = 1.0 / relaxation_time
    if model == "bgk":
        return f - omega_plus * (f - feq)
    opposite_f = f[..., OPPOSITE]
    opposite_eq = feq[..., OPPOSITE]
    non_eq_plus = 0.5 * ((f + opposite_f) - (feq + opposite_eq))
    non_eq_minus = 0.5 * ((f - opposite_f) - (feq - opposite_eq))
    tau_minus = 0.5 + magic_parameter / (relaxation_time - 0.5)
    omega_minus = 1.0 / tau_minus
    return f - omega_plus * non_eq_plus - omega_minus * non_eq_minus


def _snapshot_fields(
    f: np.ndarray, solid: np.ndarray, rho0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho, u, v = macroscopic(f)
    rho = rho.copy()
    u = u.copy()
    v = v.copy()
    rho[solid] = rho0
    u[solid] = 0.0
    v[solid] = 0.0
    pressure = CS2 * (rho - rho0)
    omega = compute_vorticity(u, v, solid)
    return rho, u, v, pressure, omega


def grid_convergence_diagnostics(
    nodes_per_diameter: Sequence[float],
    values: Sequence[float],
    *,
    safety_factor: float = 1.25,
    ratio_tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Return a three-grid Richardson/GCI solution-verification diagnostic.

    ``nodes_per_diameter`` must be ordered from coarse to fine.  The calculation
    follows the generalized asymptotic three-grid form with
    ``h/D = 1 / nodes_per_diameter`` and supports unequal refinement ratios.
    A non-monotone sequence, or a sequence inconsistent with positive-order
    convergence, is reported explicitly and does not receive a fabricated
    order or GCI value.

    The Grid Convergence Index is a discretization-uncertainty estimate, not a
    comparison with physical truth.  Statistical convergence, domain
    sensitivity, and validation against independent reference data remain
    separate requirements.
    """
    resolution = np.asarray(nodes_per_diameter, dtype=float).reshape(-1)
    quantity = np.asarray(values, dtype=float).reshape(-1)
    if resolution.size != 3 or quantity.size != 3:
        raise ValueError("three coarse-to-fine resolutions and values are required")
    if (
        not np.isfinite(resolution).all()
        or not np.isfinite(quantity).all()
        or np.any(resolution <= 0)
        or np.any(np.diff(resolution) <= 0)
    ):
        raise ValueError("resolutions and values must be finite and resolutions increasing")
    if safety_factor <= 1.0:
        raise ValueError("GCI safety_factor must exceed one")

    ratios = resolution[1:] / resolution[:-1]
    refinement_ratio = float(ratios[-1])
    constant_ratio = bool(
        np.allclose(ratios, ratios[0], rtol=ratio_tolerance, atol=0.0)
    )

    coarse, medium, fine = map(float, quantity)
    coarse_to_medium = coarse - medium
    medium_to_fine = medium - fine
    scale = max(abs(coarse), abs(medium), abs(fine), np.finfo(float).eps)
    monotone = bool(
        coarse_to_medium * medium_to_fine > 0.0
        and abs(coarse_to_medium) > 100.0 * np.finfo(float).eps * scale
        and abs(medium_to_fine) > 100.0 * np.finfo(float).eps * scale
    )
    fine_pair_change = 100.0 * abs(fine - medium) / max(abs(fine), np.finfo(float).eps)
    diagnostic: dict[str, Any] = {
        "valid_asymptotic_sequence": False,
        "reason": "non_monotone_or_roundoff_limited",
        "refinement_ratio": refinement_ratio,
        "refinement_ratios": ratios.tolist(),
        "constant_refinement_ratio": constant_ratio,
        "observed_order": float("nan"),
        "richardson_extrapolated": float("nan"),
        "fine_pair_relative_change_percent": float(fine_pair_change),
        "fine_grid_gci_percent": float("nan"),
        "safety_factor": float(safety_factor),
    }
    if not monotone:
        return diagnostic

    spacing = 1.0 / resolution
    target_ratio = abs(coarse_to_medium / medium_to_fine)

    def residual(order: float) -> float:
        numerator = spacing[0] ** order - spacing[1] ** order
        denominator_local = spacing[1] ** order - spacing[2] ** order
        return numerator / denominator_local - target_ratio

    lower_order, upper_order = 1.0e-3, 12.0
    lower_value, upper_value = residual(lower_order), residual(upper_order)
    if (
        not np.isfinite(lower_value)
        or not np.isfinite(upper_value)
        or lower_value * upper_value > 0.0
    ):
        diagnostic["reason"] = "non_positive_observed_order"
        return diagnostic
    for _ in range(100):
        midpoint = 0.5 * (lower_order + upper_order)
        midpoint_value = residual(midpoint)
        if lower_value * midpoint_value <= 0.0:
            upper_order, upper_value = midpoint, midpoint_value
        else:
            lower_order, lower_value = midpoint, midpoint_value
    observed_order = float(0.5 * (lower_order + upper_order))
    denominator = refinement_ratio**observed_order - 1.0
    if denominator <= np.finfo(float).eps:
        diagnostic["reason"] = "ill_conditioned_richardson_extrapolation"
        return diagnostic
    extrapolated = fine + (fine - medium) / denominator
    gci = safety_factor * abs((fine - medium) / fine) / denominator * 100.0
    diagnostic.update(
        valid_asymptotic_sequence=True,
        reason="accepted",
        observed_order=observed_order,
        richardson_extrapolated=float(extrapolated),
        fine_grid_gci_percent=float(gci),
    )
    return diagnostic


def recommended_parameters(reynolds: float, fidelity: str = "quick") -> dict[str, Any]:
    """Return transparent parameters for a qualitative or validation-grade run.

    ``quick`` is intended for notebook exploration and regime visualization; it
    is *not* a grid-converged external-cylinder reference.  ``validation`` uses
    at least 24 nodes per diameter, ``tau >= 0.53``, a cylinder centre eight
    diameters from the inlet, twenty-two centre-to-outlet diameters, blockage
    at most 0.05, and roughly 150
    convective time units.  Those conservative settings are intentionally
    expensive and should still be accompanied by grid/domain/time refinement.
    """
    reynolds = float(reynolds)
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")
    fidelity = str(fidelity).lower()
    if fidelity == "quick":
        diameter = max(12, int(np.ceil(reynolds / 12.0)))
        ny = 8 * diameter
        nx = 20 * diameter
        return {
            "nx": nx,
            "ny": ny,
            "diameter": float(diameter),
            "center": (5.0 * diameter, 0.5 * (ny - 1)),
            "inflow_velocity": 0.05,
            "startup_ramp_steps": 25 * diameter,
            "steps": (2000 if reynolds >= 60 else 800) * diameter,
            "history_stride": max(2, diameter // 3),
        }
    if fidelity == "validation":
        # tau=.5+3UD/Re >= .53 at U=.05 requires D >= .2 Re.
        diameter = max(32, int(np.ceil(0.2 * reynolds)))
        ny = 20 * diameter
        nx = 30 * diameter
        return {
            "nx": nx,
            "ny": ny,
            "diameter": float(diameter),
            "center": (8.0 * diameter, 0.5 * (ny - 1)),
            "inflow_velocity": 0.05,
            "startup_ramp_steps": 25 * diameter,
            "steps": 3000 * diameter,
            "history_stride": max(2, diameter // 3),
        }
    raise ValueError("fidelity must be 'quick' or 'validation'")


def simulate_cylinder(
    reynolds: float,
    *,
    nx: int = 240,
    ny: int = 80,
    diameter: float = 10.0,
    center: tuple[float, float] | None = None,
    inflow_velocity: float = 0.05,
    rho0: float = 1.0,
    steps: int = 5000,
    history_stride: int = 5,
    snapshot_stride: int | None = None,
    snapshot_start: int = 0,
    statistics_start: int | None = None,
    perturbation: float = 1.0e-3,
    seed: int = 0,
    startup_ramp_steps: int = 0,
    collision_model: str = "trt",
    cylinder_boundary: str = "bouzidi",
    restart_state: Mapping[str, Any] | None = None,
    return_restart_state: bool = False,
) -> dict[str, Any]:
    """Run deterministic low-Mach flow past a cylinder using D2Q9 TRT or BGK.

    Parameters are in lattice units.  Setting ``snapshot_stride`` retains
    time-resolved ``u``, ``v``, ``p``, and ``vorticity`` arrays for ROM/ML
    exercises; otherwise only the final fields and inexpensive force history are
    returned.  ``seed`` controls a tiny localized initial wake perturbation that
    lets the supercritical Hopf mode grow without forcing it continuously.
    ``startup_ramp_steps`` applies a half-cosine, density-preserving far-field
    acceleration together with the inlet and outlet memory.
    ``restart_state`` continues from saved populations and outlet memory; set
    ``return_restart_state`` to retain those arrays for a later quick run.
    """
    center_x, center_y = (None, None) if center is None else center
    config = CylinderLBMConfig(
        reynolds=float(reynolds), nx=int(nx), ny=int(ny), diameter=float(diameter),
        center_x=center_x, center_y=center_y, inflow_velocity=float(inflow_velocity),
        rho0=float(rho0), steps=int(steps), history_stride=int(history_stride),
        snapshot_stride=snapshot_stride, snapshot_start=int(snapshot_start),
        statistics_start=(None if statistics_start is None else int(statistics_start)),
        perturbation=float(perturbation), seed=int(seed),
        startup_ramp_steps=int(startup_ramp_steps),
        collision_model=str(collision_model).lower(),
        cylinder_boundary=str(cylinder_boundary).lower(),
    )
    _validate_config(config)
    center_xy = config.cylinder_center
    solid = cylinder_mask(config.nx, config.ny, config.diameter, center_xy)
    curved_fractions = (
        curved_link_fractions(solid, center_xy, config.diameter)
        if config.cylinder_boundary == "bouzidi"
        else None
    )

    completed_steps = 0
    if restart_state is not None:
        if config.startup_ramp_steps:
            raise ValueError("startup_ramp_steps cannot be used with restart_state")
        f = np.asarray(restart_state["populations"], dtype=float).copy()
        outlet_previous = np.asarray(
            restart_state["outlet_previous"], dtype=float
        ).copy()
        completed_steps = int(restart_state.get("completed_steps", 0))
        if f.shape != (config.ny, config.nx, 9):
            raise ValueError("restart populations do not match the requested grid")
        if outlet_previous.shape != (config.ny, 9):
            raise ValueError("restart outlet_previous does not match the requested grid")
        if not np.isfinite(f).all() or not np.isfinite(outlet_previous).all():
            raise ValueError("restart state contains non-finite values")
    else:
        x = np.arange(config.nx, dtype=float)[None, :]
        y = np.arange(config.ny, dtype=float)[:, None]
        rho = np.full((config.ny, config.nx), config.rho0, dtype=float)
        initial_speed = 0.0 if config.startup_ramp_steps else config.inflow_velocity
        u = np.full_like(rho, initial_speed)
        v = np.zeros_like(rho)
        if config.perturbation:
            rng = np.random.default_rng(config.seed)
            # A smooth localized wake disturbance avoids grid-scale noise.
            sign = -1.0 if rng.integers(0, 2) == 0 else 1.0
            cx, cy = center_xy
            v += sign * config.perturbation * config.inflow_velocity * np.exp(
                -((x - (cx + 2.0 * config.diameter)) / (1.5 * config.diameter)) ** 2
                -((y - cy) / config.diameter) ** 2
            )
        u[solid] = 0.0
        v[solid] = 0.0
        f = equilibrium(rho, u, v)
        outlet_previous = f[:, -1, :].copy()

    times: list[float] = []
    drag: list[float] = []
    lift: list[float] = []
    mass: list[float] = []
    snapshot_times: list[float] = []
    snapshots: dict[str, list[np.ndarray]] = {
        "u": [], "v": [], "p": [], "vorticity": []
    }
    q_scale = 0.5 * config.rho0 * config.inflow_velocity**2 * config.diameter
    outlet_speed = config.inflow_velocity if config.outlet_speed is None else config.outlet_speed
    force_x_accumulator = 0.0
    force_y_accumulator = 0.0
    force_samples = 0
    statistics_start_local = (
        config.steps // 2 if config.statistics_start is None else config.statistics_start
    )
    mean_u_accumulator = np.zeros((config.ny, config.nx), dtype=float)
    mean_v_accumulator = np.zeros_like(mean_u_accumulator)
    mean_sample_count = 0
    previous_ramp_fraction = 0.0

    for local_step in range(1, config.steps + 1):
        step = completed_steps + local_step
        ramp_fraction = 1.0
        if config.startup_ramp_steps and local_step <= config.startup_ramp_steps:
            phase = np.pi * local_step / config.startup_ramp_steps
            ramp_fraction = 0.5 * (1.0 - np.cos(phase))

        rho, u, v = macroscopic(f)
        u[solid] = 0.0
        v[solid] = 0.0
        if config.startup_ramp_steps and local_step <= config.startup_ramp_steps:
            # Smoothly accelerate the complete far field together with the
            # inlet.  Adding an equilibrium difference changes momentum but
            # preserves density exactly, avoiding the mass accumulation that a
            # rest-filled domain develops with a convective outlet.
            accelerated_u = u.copy()
            accelerated_u[~solid] += config.inflow_velocity * (
                ramp_fraction - previous_ramp_fraction
            )
            equilibrium_shift = equilibrium(rho, accelerated_u, v) - equilibrium(
                rho, u, v
            )
            f += equilibrium_shift
            outlet_previous += equilibrium_shift[:, -1, :]
            u = accelerated_u
            previous_ramp_fraction = ramp_fraction
        feq = equilibrium(rho, u, v)
        post = _collide(
            f, feq, config.relaxation_time, config.collision_model,
            config.trt_magic_parameter,
        )
        streamed, force_x, force_y = _stream_and_bounce(
            post, solid, curved_fractions
        )
        force_x_accumulator += force_x
        force_y_accumulator += force_y
        force_samples += 1

        # First-order convective outlet: df/dt + c_out*df/dx = 0.  The implicit
        # update below is notably calmer than direct distribution copying.
        interior = streamed[:, -2, :]
        local_outlet_speed = outlet_speed * ramp_fraction
        new_outlet = (
            outlet_previous + local_outlet_speed * interior
        ) / (1.0 + local_outlet_speed)
        streamed[:, -1, :] = new_outlet
        outlet_previous = new_outlet.copy()
        inlet_speed = config.inflow_velocity * ramp_fraction
        _apply_velocity_inlet(streamed, inlet_speed)
        f = streamed

        if not np.isfinite(f).all() or np.min(f.sum(axis=-1)) <= 0:
            raise FloatingPointError(
                f"LBM became non-physical at step {step}; use a larger diameter "
                "or lower Reynolds/inflow speed"
            )

        if local_step % config.history_stride == 0 or local_step == config.steps:
            times.append(float(step))
            # Block averaging removes harmless one-lattice-step momentum-exchange
            # jitter without filtering the much slower physical shedding signal.
            drag.append(force_x_accumulator / (force_samples * q_scale))
            lift.append(force_y_accumulator / (force_samples * q_scale))
            mass.append(float(f.sum(axis=-1)[~solid].mean() / config.rho0))
            if local_step >= statistics_start_local:
                # ``u`` and ``v`` are the state entering this update; the
                # one-step sampling offset is negligible and avoids a second
                # full macroscopic reconstruction at every history sample.
                mean_u_accumulator += u
                mean_v_accumulator += v
                mean_sample_count += 1
            force_x_accumulator = 0.0
            force_y_accumulator = 0.0
            force_samples = 0
        if (
            config.snapshot_stride is not None
            and local_step >= config.snapshot_start
            and (local_step - config.snapshot_start) % config.snapshot_stride == 0
        ):
            _, su, sv, sp, sw = _snapshot_fields(f, solid, config.rho0)
            snapshot_times.append(float(step))
            snapshots["u"].append(su)
            snapshots["v"].append(sv)
            snapshots["p"].append(sp)
            snapshots["vorticity"].append(sw)

    final_rho, final_u, final_v, final_p, final_omega = _snapshot_fields(
        f, solid, config.rho0
    )
    history_time = np.asarray(times)
    drag_array = np.asarray(drag)
    lift_array = np.asarray(lift)
    instantaneous_recirc, instantaneous_recirc_over_d = recirculation_length(
        final_u, solid, center_xy, config.diameter
    )
    mean_u = (
        mean_u_accumulator / mean_sample_count if mean_sample_count else final_u.copy()
    )
    mean_v = (
        mean_v_accumulator / mean_sample_count if mean_sample_count else final_v.copy()
    )
    recirc, recirc_over_d = recirculation_length(
        mean_u, solid, center_xy, config.diameter
    )
    frequency_diagnostics = strouhal_diagnostics(
        history_time,
        lift_array,
        config.diameter,
        config.inflow_velocity,
        transient_fraction=statistics_start_local / config.steps,
    )
    strouhal = float(frequency_diagnostics["strouhal"])
    cx, _ = center_xy
    upstream_center_d = cx / config.diameter
    downstream_center_d = (config.nx - cx) / config.diameter
    convective_time = config.steps * config.inflow_velocity / config.diameter
    validation_criteria = {
        "lattice_mach_below_0.1": config.mach < 0.1,
        "relaxation_time_at_least_0.53": config.relaxation_time >= 0.53,
        "nodes_per_diameter_at_least_24": config.diameter >= 24,
        "blockage_ratio_at_most_0.05": config.blockage_ratio <= 0.05,
        "inlet_to_center_at_least_8D": upstream_center_d >= 8.0,
        "center_to_outlet_at_least_22D": downstream_center_d >= 22.0,
        "simulated_time_at_least_150D_over_U": convective_time >= 150.0,
    }
    validation_candidate = all(validation_criteria.values())
    metadata = {
        "solver": f"D2Q9 {config.collision_model.upper()} lattice-Boltzmann",
        "solver_version": CYLINDER_LBM_VERSION,
        "collision": (
            "two-relaxation-time TRT (magic parameter 3/16)"
            if config.collision_model == "trt"
            else "single-relaxation-time BGK"
        ),
        "cylinder_boundary": (
            "Bouzidi interpolated bounce-back on the analytical circle"
            if config.cylinder_boundary == "bouzidi"
            else "halfway link bounce-back"
        ),
        "inlet_boundary": "Zou-He uniform velocity",
        "outlet_boundary": "first-order convective distributions",
        "transverse_boundary": "periodic",
        "pressure_definition": "p=(rho-rho0)/3 (gauge, lattice units)",
        "force_definition": "momentum exchange on fluid-solid links",
        "reynolds": config.reynolds,
        "viscosity": config.viscosity,
        "relaxation_time": config.relaxation_time,
        "lattice_mach": config.mach,
        "blockage_ratio": config.blockage_ratio,
        "cylinder_center": center_xy,
        "deterministic_seed": config.seed,
        "completed_steps_before_restart": completed_steps,
        "statistics_start_step": completed_steps + statistics_start_local,
        "time_mean_sample_count": mean_sample_count,
        "fidelity_classification": (
            "validation_candidate" if validation_candidate else "quick_qualitative"
        ),
        "validation_criteria": validation_criteria,
        "validation_note": (
            "A validation candidate still requires explicit grid, domain, and "
            "sampling-time convergence; quick runs are qualitative."
        ),
        "config": asdict(config),
    }
    result: dict[str, Any] = {
        "rho": final_rho,
        "u": final_u,
        "v": final_v,
        "p": final_p,
        "vorticity": final_omega,
        "solid": solid,
        "x": np.arange(config.nx, dtype=float),
        "y": np.arange(config.ny, dtype=float),
        "time": history_time,
        "drag_coefficient": drag_array,
        "lift_coefficient": lift_array,
        "mean_density_ratio": np.asarray(mass),
        "strouhal": float(strouhal),
        "strouhal_diagnostics": frequency_diagnostics,
        "recirculation_length": float(recirc),
        "recirculation_length_over_diameter": float(recirc_over_d),
        "instantaneous_recirculation_length": float(instantaneous_recirc),
        "instantaneous_recirculation_length_over_diameter": float(
            instantaneous_recirc_over_d
        ),
        "time_mean_u": mean_u,
        "time_mean_v": mean_v,
        "metadata": metadata,
    }
    if config.snapshot_stride is not None:
        result["snapshot_time"] = np.asarray(snapshot_times)
        result["snapshots"] = {
            key: np.stack(values) if values else np.empty((0, config.ny, config.nx))
            for key, values in snapshots.items()
        }
    if return_restart_state:
        result["restart_state"] = {
            "populations": f.copy(),
            "outlet_previous": outlet_previous.copy(),
            "completed_steps": completed_steps + config.steps,
        }
    return result


__all__ = [
    "CS2",
    "CYLINDER_LBM_VERSION",
    "CylinderLBMConfig",
    "LATTICE_VELOCITIES",
    "LATTICE_WEIGHTS",
    "OPPOSITE",
    "compute_vorticity",
    "curved_link_fractions",
    "cylinder_mask",
    "equilibrium",
    "estimate_strouhal",
    "grid_convergence_diagnostics",
    "macroscopic",
    "recirculation_length",
    "recommended_parameters",
    "simulate_cylinder",
    "strouhal_diagnostics",
]
