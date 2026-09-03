"""Phase-stable Fourier decoder for saturated periodic cylinder wakes.

Unlike recursive one-step CNN rollout, this representation advances an
explicit learned phase and therefore cannot accumulate numerical diffusion.
Only development Reynolds cases are used to fit spatial Fourier coefficients.
Four initial fields align the phase of a new case; all later fields are then
predicted without future CFD input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PhaseDecoder:
    reynolds: float
    strouhal: float
    harmonics: int
    coefficients: np.ndarray


def design(phase: np.ndarray, harmonics: int) -> np.ndarray:
    phase = np.asarray(phase, dtype=float).reshape(-1)
    if harmonics < 1:
        raise ValueError("harmonics must be positive")
    columns = [np.ones_like(phase)]
    for order in range(1, harmonics + 1):
        columns.extend((np.sin(order * phase), np.cos(order * phase)))
    return np.stack(columns, axis=1)


def _fields(case: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.stack([case[name] for name in ("u", "v", "p")], axis=-1)
    if values.ndim != 4 or not np.isfinite(values).all():
        raise ValueError("case fields must be finite arrays with shape (time,y,x)")
    return np.asarray(values, dtype=np.float32)


def fit_case(case: Mapping[str, np.ndarray], *, reynolds: float, harmonics: int,
             velocity: float = 0.05, diameter: float = 12.0) -> PhaseDecoder:
    """Fit one development case with phase referenced to its lift signal."""
    st = float(case["strouhal"])
    if not np.isfinite(st):
        raise ValueError("a resolved development Strouhal number is required")
    snapshot_t = np.asarray(case["snapshot_time"], dtype=float) * velocity / diameter
    history_t = np.asarray(case["time"], dtype=float) * velocity / diameter
    lift = np.interp(snapshot_t, history_t, np.asarray(case["lift_coefficient"], dtype=float))
    omega = 2.0 * np.pi * st
    basis = np.stack((np.sin(omega * snapshot_t), np.cos(omega * snapshot_t)), axis=1)
    sine, cosine = np.linalg.lstsq(basis, lift, rcond=None)[0]
    phase = omega * snapshot_t + np.arctan2(cosine, sine)
    matrix = design(phase, harmonics)
    values = _fields(case)
    coefficients = np.linalg.lstsq(
        matrix, values.reshape(values.shape[0], -1), rcond=None
    )[0].reshape(matrix.shape[1], *values.shape[1:])
    return PhaseDecoder(float(reynolds), st, harmonics, coefficients.astype(np.float32))


def interpolate(models: list[PhaseDecoder], reynolds: float) -> PhaseDecoder:
    """Linearly interpolate the two development models bracketing Reynolds."""
    ordered = sorted(models, key=lambda model: model.reynolds)
    if len(ordered) < 2 or not ordered[0].reynolds <= reynolds <= ordered[-1].reynolds:
        raise ValueError("target Reynolds must be bracketed by development cases")
    lower, upper = next(
        (a, b) for a, b in zip(ordered[:-1], ordered[1:])
        if a.reynolds <= reynolds <= b.reynolds
    )
    if lower.harmonics != upper.harmonics:
        raise ValueError("development models must use the same harmonic order")
    weight = (float(reynolds) - lower.reynolds) / (upper.reynolds - lower.reynolds)
    return PhaseDecoder(
        float(reynolds),
        (1.0 - weight) * lower.strouhal + weight * upper.strouhal,
        lower.harmonics,
        ((1.0 - weight) * lower.coefficients + weight * upper.coefficients).astype(np.float32),
    )


def align_and_predict(model: PhaseDecoder, initial_fields: np.ndarray, frame_count: int,
                      *, delta_t_star: float, search_points: int = 1441) -> tuple[np.ndarray, float]:
    """Align from initial fields and autonomously predict all requested frames."""
    initial = np.asarray(initial_fields, dtype=np.float32)
    if initial.ndim != 4 or initial.shape[0] < 2 or initial.shape[-1] != 3:
        raise ValueError("initial_fields must have shape (history,y,x,3)")
    if frame_count < initial.shape[0] or delta_t_star <= 0:
        raise ValueError("invalid frame_count or delta_t_star")
    offsets = np.linspace(-np.pi, np.pi, int(search_points), endpoint=False)
    increments = 2.0 * np.pi * model.strouhal * delta_t_star * np.arange(initial.shape[0])
    best_error, best_phase = np.inf, 0.0
    for offset in offsets:
        estimate = np.tensordot(design(offset + increments, model.harmonics),
                                model.coefficients, axes=(1, 0))
        error = float(np.mean((estimate - initial) ** 2))
        if error < best_error:
            best_error, best_phase = error, float(offset)
    phase = best_phase + 2.0 * np.pi * model.strouhal * delta_t_star * np.arange(frame_count)
    prediction = np.tensordot(design(phase, model.harmonics), model.coefficients, axes=(1, 0))
    return np.asarray(prediction, dtype=np.float32), best_phase
