"""Leakage-safe educational machine-learning baselines for cylinder wakes.

The routines in this module intentionally stay small and transparent.  They
provide a centered proper orthogonal decomposition (POD), followed by a
deterministic ridge fit of the modal coefficients as functions of Reynolds
number and shedding phase.  This is a useful baseline for a later neural
operator lesson; it is not presented as a new research method.

Snapshots are expected in ``(sample, y, x)`` form for each field.  All
snapshots at the same Reynolds number must remain on the same side of a split;
:func:`casewise_reynolds_split` enforces that rule explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class FieldLayout:
    """Description needed to map flattened snapshots back to CFD fields."""

    names: tuple[str, ...]
    spatial_shape: tuple[int, ...]
    field_size: int

    @property
    def vector_size(self) -> int:
        return len(self.names) * self.field_size


@dataclass(frozen=True)
class PODBasis:
    """Centered POD basis whose modes are stored column-wise."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.modes.shape[1])


@dataclass(frozen=True)
class ReynoldsSplit:
    """Indices for a Reynolds-case-wise train/test partition."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_reynolds: np.ndarray
    test_reynolds: np.ndarray


@dataclass(frozen=True)
class CylinderPODRegressor:
    """POD plus a small Reynolds/phase modal-coefficient regressor."""

    layout: FieldLayout
    pod: PODBasis
    weights: np.ndarray
    reynolds_center: float
    reynolds_scale: float
    reynolds_degree: int
    phase_harmonics: int
    ridge: float

    def predict_coefficients(
        self, reynolds: np.ndarray | Sequence[float] | float,
        phase: np.ndarray | Sequence[float] | float,
    ) -> np.ndarray:
        """Predict POD coefficients; Reynolds number and phase broadcast."""
        design = _design_matrix(
            reynolds,
            phase,
            self.reynolds_center,
            self.reynolds_scale,
            self.reynolds_degree,
            self.phase_harmonics,
        )
        return design @ self.weights

    def predict_vectors(
        self, reynolds: np.ndarray | Sequence[float] | float,
        phase: np.ndarray | Sequence[float] | float,
    ) -> np.ndarray:
        """Predict flattened multi-field snapshots."""
        return reconstruct_pod(self.pod, self.predict_coefficients(reynolds, phase))

    def predict_fields(
        self, reynolds: np.ndarray | Sequence[float] | float,
        phase: np.ndarray | Sequence[float] | float,
    ) -> dict[str, np.ndarray]:
        """Predict a dictionary of fields in ``(sample, ...)`` form."""
        return unpack_fields(self.predict_vectors(reynolds, phase), self.layout)


def pack_fields(
    fields: Mapping[str, np.ndarray],
    field_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, FieldLayout]:
    """Flatten equally shaped CFD fields into one feature vector per sample."""
    if not fields:
        raise ValueError("fields must not be empty")
    names = tuple(fields.keys()) if field_names is None else tuple(field_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("field_names must contain unique names")
    missing = [name for name in names if name not in fields]
    if missing:
        raise KeyError(f"missing fields: {missing}")

    arrays = [np.asarray(fields[name], dtype=float) for name in names]
    if any(array.ndim < 2 for array in arrays):
        raise ValueError("each field must have shape (sample, spatial dimensions...)")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("all fields must have identical shapes")
    if shape[0] < 1:
        raise ValueError("at least one snapshot is required")
    if any(not np.isfinite(array).all() for array in arrays):
        raise FloatingPointError("fields contain a non-finite value")

    field_size = int(np.prod(shape[1:]))
    vectors = np.concatenate(
        [array.reshape(shape[0], field_size) for array in arrays], axis=1
    )
    layout = FieldLayout(names, tuple(shape[1:]), field_size)
    return vectors, layout


def unpack_fields(vectors: np.ndarray, layout: FieldLayout) -> dict[str, np.ndarray]:
    """Invert :func:`pack_fields` while retaining an explicit sample axis."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != layout.vector_size:
        raise ValueError(
            f"vectors must have shape (samples, {layout.vector_size})"
        )
    return {
        name: values[:, i * layout.field_size : (i + 1) * layout.field_size].reshape(
            (values.shape[0],) + layout.spatial_shape
        )
        for i, name in enumerate(layout.names)
    }


def fit_pod(
    snapshots: np.ndarray,
    rank: int | None = None,
    energy: float = 0.999,
) -> PODBasis:
    """Fit a centered POD basis by singular-value decomposition.

    If ``rank`` is omitted, the smallest rank retaining the requested energy is
    selected.  Supplying ``rank`` makes classroom comparisons reproducible.
    """
    values = np.asarray(snapshots, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("snapshots must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise FloatingPointError("snapshots contain a non-finite value")
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must lie in (0, 1]")

    mean = values.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(values - mean, full_matrices=False)
    modal_energy = singular_values**2
    total = float(modal_energy.sum())
    cumulative = (
        np.cumsum(modal_energy) / total
        if total > np.finfo(float).eps
        else np.ones_like(modal_energy)
    )
    max_rank = min(values.shape)
    if rank is None:
        selected = int(np.searchsorted(cumulative, energy, side="left") + 1)
    else:
        if int(rank) != rank or not 1 <= int(rank) <= max_rank:
            raise ValueError(f"rank must be an integer in [1, {max_rank}]")
        selected = int(rank)
    return PODBasis(
        mean=mean,
        modes=vt[:selected].T,
        singular_values=singular_values,
        cumulative_energy=cumulative,
    )


def project_pod(pod: PODBasis, snapshots: np.ndarray) -> np.ndarray:
    """Project flattened snapshots onto a fitted POD basis."""
    values = np.asarray(snapshots, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != pod.mean.size:
        raise ValueError(f"snapshots must have {pod.mean.size} features")
    return (values - pod.mean) @ pod.modes


def reconstruct_pod(pod: PODBasis, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct flattened snapshots from POD coefficients."""
    values = np.asarray(coefficients, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != pod.rank:
        raise ValueError(f"coefficients must have {pod.rank} columns")
    return pod.mean + values @ pod.modes.T


def casewise_reynolds_split(
    reynolds: np.ndarray | Sequence[float],
    test_reynolds: Sequence[float] | None = None,
    test_fraction: float = 0.25,
    seed: int = 7,
) -> ReynoldsSplit:
    """Split complete Reynolds-number cases, never individual snapshots.

    ``test_reynolds`` is recommended for a stated blind-test protocol.  When it
    is omitted, a deterministic random subset of unique cases is selected.
    """
    labels = np.asarray(reynolds, dtype=float).reshape(-1)
    if labels.size < 2 or not np.isfinite(labels).all():
        raise ValueError("reynolds must contain at least two finite labels")
    unique = np.unique(labels)
    if unique.size < 2:
        raise ValueError("at least two distinct Reynolds cases are required")

    if test_reynolds is None:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must lie in (0, 1)")
        count = min(unique.size - 1, max(1, int(np.ceil(test_fraction * unique.size))))
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(unique, size=count, replace=False))
    else:
        requested = np.unique(np.asarray(test_reynolds, dtype=float).reshape(-1))
        if requested.size < 1 or not np.isfinite(requested).all():
            raise ValueError("test_reynolds must contain finite values")
        found = np.array([np.any(np.isclose(unique, value)) for value in requested])
        if not found.all():
            raise ValueError(f"test cases absent from labels: {requested[~found].tolist()}")
        selected = requested

    test_mask = np.zeros(labels.size, dtype=bool)
    for value in selected:
        test_mask |= np.isclose(labels, value)
    if test_mask.all():
        raise ValueError("the split must retain at least one training case")
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    return ReynoldsSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_reynolds=np.unique(labels[train_indices]),
        test_reynolds=np.unique(labels[test_indices]),
    )


def _design_matrix(
    reynolds: np.ndarray | Sequence[float] | float,
    phase: np.ndarray | Sequence[float] | float,
    center: float,
    scale: float,
    reynolds_degree: int,
    phase_harmonics: int,
) -> np.ndarray:
    re, phi = np.broadcast_arrays(
        np.asarray(reynolds, dtype=float), np.asarray(phase, dtype=float)
    )
    re = re.reshape(-1)
    phi = np.mod(phi.reshape(-1), 2.0 * np.pi)
    if not np.isfinite(re).all() or not np.isfinite(phi).all():
        raise ValueError("reynolds and phase must be finite")
    z = (re - center) / scale
    phase_terms = [np.ones_like(phi)]
    for harmonic in range(1, phase_harmonics + 1):
        phase_terms.extend((np.sin(harmonic * phi), np.cos(harmonic * phi)))
    # Tensor-product features let both the mean wake and its harmonics vary
    # smoothly with Reynolds number.
    return np.column_stack(
        [z**degree * term for degree in range(reynolds_degree + 1) for term in phase_terms]
    )


def fit_pod_regressor(
    fields: Mapping[str, np.ndarray],
    reynolds: np.ndarray | Sequence[float],
    phase: np.ndarray | Sequence[float],
    *,
    field_names: Sequence[str] | None = None,
    rank: int | None = None,
    energy: float = 0.999,
    reynolds_degree: int = 2,
    phase_harmonics: int = 2,
    ridge: float = 1.0e-10,
) -> CylinderPODRegressor:
    """Fit the compact educational POD--Reynolds/phase baseline."""
    if int(reynolds_degree) != reynolds_degree or reynolds_degree < 0:
        raise ValueError("reynolds_degree must be a non-negative integer")
    if int(phase_harmonics) != phase_harmonics or phase_harmonics < 0:
        raise ValueError("phase_harmonics must be a non-negative integer")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    vectors, layout = pack_fields(fields, field_names)
    re = np.asarray(reynolds, dtype=float).reshape(-1)
    phi = np.asarray(phase, dtype=float).reshape(-1)
    if re.size != vectors.shape[0] or phi.size != vectors.shape[0]:
        raise ValueError("reynolds and phase must provide one label per snapshot")
    if not np.isfinite(re).all() or not np.isfinite(phi).all():
        raise ValueError("reynolds and phase must be finite")

    pod = fit_pod(vectors, rank=rank, energy=energy)
    coefficients = project_pod(pod, vectors)
    center = 0.5 * (float(re.min()) + float(re.max()))
    scale = 0.5 * (float(re.max()) - float(re.min()))
    if scale <= np.finfo(float).eps:
        scale = 1.0
    design = _design_matrix(
        re, phi, center, scale, int(reynolds_degree), int(phase_harmonics)
    )
    gram = design.T @ design
    regularizer = float(ridge) * np.eye(gram.shape[0])
    # Do not shrink the constant term.  lstsq also handles ridge=0 and a
    # deliberately over-complete feature library without platform surprises.
    regularizer[0, 0] = 0.0
    weights = np.linalg.lstsq(
        np.vstack((design, np.sqrt(regularizer))),
        np.vstack((coefficients, np.zeros((design.shape[1], pod.rank)))),
        rcond=None,
    )[0]
    return CylinderPODRegressor(
        layout=layout,
        pod=pod,
        weights=weights,
        reynolds_center=center,
        reynolds_scale=scale,
        reynolds_degree=int(reynolds_degree),
        phase_harmonics=int(phase_harmonics),
        ridge=float(ridge),
    )


def reconstruction_diagnostics(
    truth: Mapping[str, np.ndarray],
    prediction: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Return transparent absolute and relative field reconstruction errors."""
    if set(truth) != set(prediction) or not truth:
        raise ValueError("truth and prediction must contain the same fields")
    metrics: dict[str, float] = {}
    residual_vectors = []
    truth_vectors = []
    for name in truth:
        exact = np.asarray(truth[name], dtype=float)
        estimated = np.asarray(prediction[name], dtype=float)
        if exact.shape != estimated.shape:
            raise ValueError(f"shape mismatch for field {name!r}")
        residual = estimated - exact
        denominator = max(float(np.linalg.norm(exact)), np.finfo(float).eps)
        metrics[f"{name}_relative_l2"] = float(np.linalg.norm(residual) / denominator)
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(residual**2)))
        residual_vectors.append(residual.reshape(-1))
        truth_vectors.append(exact.reshape(-1))
    all_residual = np.concatenate(residual_vectors)
    all_truth = np.concatenate(truth_vectors)
    metrics["combined_relative_l2"] = float(
        np.linalg.norm(all_residual)
        / max(float(np.linalg.norm(all_truth)), np.finfo(float).eps)
    )
    return metrics


def flow_diagnostics(
    fields: Mapping[str, np.ndarray],
    dx: float,
    dy: float,
    solid_mask: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    """Compute finite-difference divergence, vorticity, and no-slip diagnostics.

    Arrays may be ``(y, x)`` or ``(sample, y, x)``.  A Boolean ``solid_mask``
    over ``(y, x)`` enables an RMS speed check inside the rasterized cylinder.
    """
    if "u" not in fields or "v" not in fields:
        raise KeyError("flow diagnostics require u and v fields")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive")
    u = np.asarray(fields["u"], dtype=float)
    v = np.asarray(fields["v"], dtype=float)
    if u.shape != v.shape or u.ndim not in (2, 3):
        raise ValueError("u and v must have matching (y,x) or (sample,y,x) shapes")
    divergence = np.gradient(u, dx, axis=-1) + np.gradient(v, dy, axis=-2)
    vorticity = np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2)
    result: dict[str, float | np.ndarray] = {
        "divergence_rms": float(np.sqrt(np.mean(divergence**2))),
        "divergence_max_abs": float(np.max(np.abs(divergence))),
        "vorticity": vorticity,
    }
    if solid_mask is not None:
        mask = np.asarray(solid_mask, dtype=bool)
        if mask.shape != u.shape[-2:] or not mask.any():
            raise ValueError("solid_mask must match (y,x) and contain a solid cell")
        speed_squared = u**2 + v**2
        result["solid_speed_rms"] = float(np.sqrt(np.mean(speed_squared[..., mask])))
    return result


__all__ = [
    "CylinderPODRegressor",
    "FieldLayout",
    "PODBasis",
    "ReynoldsSplit",
    "casewise_reynolds_split",
    "fit_pod",
    "fit_pod_regressor",
    "flow_diagnostics",
    "pack_fields",
    "project_pod",
    "reconstruct_pod",
    "reconstruction_diagnostics",
    "unpack_fields",
]
