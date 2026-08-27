"""Classical POD--Galerkin and POD--DEIM models for the FlowMLLab cavity.

The module deliberately uses the same nondimensional square lid-driven cavity,
streamfunction--vorticity equations, second-order spatial differences, and DST
Poisson solve as :mod:`w4utils`.  The reduced state contains interior vorticity;
streamfunction and velocity therefore remain kinematically coupled and the
velocity reconstructed from every reduced state is discretely divergence-free.

The standard POD--Galerkin path evaluates the complete nonlinear full-order
right-hand side at every time step.  The POD--DEIM path evaluates convection at
selected interpolation points and projects an exact affine diffusion operator.
This separation makes the nonlinear online-cost bottleneck measurable rather
than merely describing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy.fft import dstn, idstn
from scipy.linalg import qr

CAVITY_ROM_VERSION = "0.1.0"


def _solve_poisson_dst(omega: np.ndarray, h: float) -> np.ndarray:
    """Solve ``laplacian(psi)=-omega`` with zero streamfunction walls."""
    rhs = -np.asarray(omega[1:-1, 1:-1], dtype=float)
    size_y, size_x = rhs.shape
    ky = np.arange(1, size_y + 1)
    kx = np.arange(1, size_x + 1)
    lambda_y = 2.0 * (np.cos(np.pi * ky / (size_y + 1)) - 1.0) / h**2
    lambda_x = 2.0 * (np.cos(np.pi * kx / (size_x + 1)) - 1.0) / h**2
    transformed = dstn(rhs, type=1, norm="ortho")
    psi = np.zeros_like(omega, dtype=float)
    psi[1:-1, 1:-1] = idstn(
        transformed / (lambda_y[:, None] + lambda_x[None, :]),
        type=1,
        norm="ortho",
    )
    return psi


def _apply_vorticity_bc(
    psi: np.ndarray, omega: np.ndarray, u_lid: float, h: float
) -> np.ndarray:
    """Apply the same second-order Thom wall closure as the Week-4 FOM."""
    omega[0, 1:-1] = -2.0 * psi[1, 1:-1] / h**2
    omega[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / h**2 - 2.0 * u_lid / h
    omega[1:-1, 0] = -2.0 * psi[1:-1, 1] / h**2
    omega[1:-1, -1] = -2.0 * psi[1:-1, -2] / h**2
    omega[0, 0] = 0.5 * (omega[0, 1] + omega[1, 0])
    omega[0, -1] = 0.5 * (omega[0, -2] + omega[1, -1])
    omega[-1, 0] = 0.5 * (omega[-1, 1] + omega[-2, 0])
    omega[-1, -1] = 0.5 * (omega[-1, -2] + omega[-2, -1])
    return omega


def _compute_velocity(
    psi: np.ndarray, u_lid: float, h: float
) -> tuple[np.ndarray, np.ndarray]:
    """Recover velocity from streamfunction and impose the cavity walls."""
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * h)
    v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * h)
    u[-1, :] = u_lid
    u[:, [0, -1]] = 0.0
    v[[0, -1], :] = 0.0
    v[:, [0, -1]] = 0.0
    return u, v


@dataclass(frozen=True)
class PODModel:
    """Centered POD representation of the interior-vorticity state."""

    n: int
    u_lid: float
    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray

    @property
    def rank(self) -> int:
        return int(self.modes.shape[1])


@dataclass(frozen=True)
class NonlinearBasis:
    """Centered POD basis of full-order convection snapshots."""

    mean: np.ndarray
    modes: np.ndarray
    singular_values: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.modes.shape[1])


@dataclass(frozen=True)
class DEIMModel:
    """Hyper-reduced cavity model with sampled nonlinear convection."""

    pod: PODModel
    indices: np.ndarray
    nonlinear_basis: np.ndarray
    nonlinear_mean: np.ndarray
    diffusion_offset_unit: np.ndarray
    diffusion_matrix_unit: np.ndarray
    convection_offset: np.ndarray
    deim_projection: np.ndarray
    sampled_convection_mean: np.ndarray
    sampled_u_offset: np.ndarray
    sampled_v_offset: np.ndarray
    sampled_dwdx_offset: np.ndarray
    sampled_dwdy_offset: np.ndarray
    sampled_u_matrix: np.ndarray
    sampled_v_matrix: np.ndarray
    sampled_dwdx_matrix: np.ndarray
    sampled_dwdy_matrix: np.ndarray

    @property
    def rank(self) -> int:
        return self.pod.rank

    @property
    def deim_dimension(self) -> int:
        return int(len(self.indices))


def _validate_grid(n: int) -> None:
    if int(n) != n or n < 17:
        raise ValueError("n must be an integer with n >= 17")


def _validate_state(q: np.ndarray, n: int) -> np.ndarray:
    state = np.asarray(q, dtype=float).reshape(-1)
    expected = (n - 2) ** 2
    if state.size != expected:
        raise ValueError(f"interior state has {state.size} entries; expected {expected}")
    if not np.isfinite(state).all():
        raise FloatingPointError("interior state contains a non-finite value")
    return state


def state_to_fields(q: np.ndarray, n: int, u_lid: float = 1.0) -> dict[str, np.ndarray]:
    """Recover full vorticity, streamfunction, and velocity from an interior state."""
    _validate_grid(n)
    q = _validate_state(q, n)
    h = 1.0 / (n - 1)
    omega = np.zeros((n, n), dtype=float)
    omega[1:-1, 1:-1] = q.reshape(n - 2, n - 2)
    psi = _solve_poisson_dst(omega, h)
    omega = _apply_vorticity_bc(psi, omega, float(u_lid), h)
    u, v = _compute_velocity(psi, float(u_lid), h)
    return {"omega": omega, "psi": psi, "u": u, "v": v}


def rhs_terms(
    q: np.ndarray,
    reynolds: float,
    n: int,
    u_lid: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior convection and diffusion terms of the vorticity equation."""
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")
    fields = state_to_fields(q, n, u_lid)
    omega, u, v = fields["omega"], fields["u"], fields["v"]
    h = 1.0 / (n - 1)
    dwdx = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * h)
    dwdy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * h)
    lap = (
        omega[1:-1, 2:]
        - 2.0 * omega[1:-1, 1:-1]
        + omega[1:-1, :-2]
        + omega[2:, 1:-1]
        - 2.0 * omega[1:-1, 1:-1]
        + omega[:-2, 1:-1]
    ) / h**2
    convection = -(u[1:-1, 1:-1] * dwdx + v[1:-1, 1:-1] * dwdy)
    diffusion = lap / float(reynolds)
    return convection.reshape(-1), diffusion.reshape(-1)


def rhs(q: np.ndarray, reynolds: float, n: int, u_lid: float = 1.0) -> np.ndarray:
    """Return the semi-discrete full-order vorticity right-hand side."""
    convection, diffusion = rhs_terms(q, reynolds, n, u_lid)
    return convection + diffusion


def simulate_fom(
    reynolds: float,
    n: int = 33,
    dt: float = 2.0e-3,
    steps: int = 2000,
    snapshot_stride: int = 25,
    u_lid: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Integrate the cavity FOM and retain a deterministic snapshot trajectory.

    Forward Euler is used intentionally because it is the time integrator in the
    existing Week-4 cavity data-production solver.  The routine shares the same
    spatial kernels through :mod:`w4utils` and adds only snapshot collection.
    """
    _validate_grid(n)
    if reynolds <= 0 or dt <= 0 or steps < 1 or snapshot_stride < 1:
        raise ValueError("require reynolds>0, dt>0, steps>=1, and snapshot_stride>=1")
    diffusion_limit = reynolds * (1.0 / (n - 1)) ** 2 / 4.0
    if dt > 0.8 * diffusion_limit:
        raise ValueError(
            f"dt={dt:g} violates the conservative diffusion gate "
            f"0.8*Re*h^2/4={0.8*diffusion_limit:g}"
        )
    size = (n - 2) ** 2
    q = np.zeros(size, dtype=float) if initial_state is None else _validate_state(initial_state, n).copy()
    states = [q.copy()]
    times = [0.0]
    started = perf_counter()
    for step in range(1, int(steps) + 1):
        q += float(dt) * rhs(q, reynolds, n, u_lid)
        if not np.isfinite(q).all():
            raise FloatingPointError("FOM became non-finite; reduce dt")
        if step % snapshot_stride == 0 or step == steps:
            states.append(q.copy())
            times.append(step * float(dt))
    elapsed = perf_counter() - started
    trajectory = np.stack(states)
    final_fields = state_to_fields(trajectory[-1], n, u_lid)
    return {
        "time": np.asarray(times),
        "states": trajectory,
        "final_omega": final_fields["omega"],
        "final_psi": final_fields["psi"],
        "final_u": final_fields["u"],
        "final_v": final_fields["v"],
        "reynolds": float(reynolds),
        "n": int(n),
        "dt": float(dt),
        "steps": int(steps),
        "snapshot_stride": int(snapshot_stride),
        "elapsed_seconds": float(elapsed),
    }


def fit_pod(snapshot_sets: Iterable[np.ndarray], rank: int) -> PODModel:
    """Fit a centered, uniformly weighted POD basis to leakage-free snapshots."""
    arrays = [np.asarray(values, dtype=float) for values in snapshot_sets]
    if not arrays or any(values.ndim != 2 for values in arrays):
        raise ValueError("snapshot_sets must contain two-dimensional arrays")
    state_size = arrays[0].shape[1]
    if any(values.shape[1] != state_size for values in arrays):
        raise ValueError("all snapshot sets must have the same state dimension")
    snapshots = np.concatenate(arrays, axis=0).T
    mean = snapshots.mean(axis=1)
    centered = snapshots - mean[:, None]
    modes, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    available = int(np.count_nonzero(singular_values > np.finfo(float).eps * singular_values[0]))
    if not 1 <= rank <= available:
        raise ValueError(f"rank must be between 1 and {available}")
    modes = modes[:, :rank].copy()
    # Resolve the arbitrary SVD sign so saved evidence is platform-stable.
    for column in range(rank):
        pivot = int(np.argmax(np.abs(modes[:, column])))
        if modes[pivot, column] < 0:
            modes[:, column] *= -1.0
    energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    n = int(round(np.sqrt(state_size))) + 2
    if (n - 2) ** 2 != state_size:
        raise ValueError("snapshot state dimension is not a square cavity interior")
    return PODModel(
        n=n,
        u_lid=1.0,
        mean=mean,
        modes=modes,
        singular_values=singular_values,
        cumulative_energy=energy,
    )


def truncate_pod(model: PODModel, rank: int) -> PODModel:
    """Return a lower-rank view without recomputing the development-set SVD."""
    if not 1 <= rank <= model.rank:
        raise ValueError(f"rank must be between 1 and {model.rank}")
    return PODModel(
        n=model.n,
        u_lid=model.u_lid,
        mean=model.mean,
        modes=model.modes[:, :rank],
        singular_values=model.singular_values,
        cumulative_energy=model.cumulative_energy,
    )


def project_state(model: PODModel, q: np.ndarray) -> np.ndarray:
    """Orthogonally project a full state onto the centered POD trial space."""
    return model.modes.T @ (_validate_state(q, model.n) - model.mean)


def reconstruct_state(model: PODModel, coefficients: np.ndarray) -> np.ndarray:
    """Reconstruct an interior-vorticity state from modal coefficients."""
    a = np.asarray(coefficients, dtype=float).reshape(-1)
    if a.size != model.rank:
        raise ValueError(f"coefficient vector has length {a.size}; expected {model.rank}")
    return model.mean + model.modes @ a


def _integrate_modal(
    modal_rhs,
    initial_coefficients: np.ndarray,
    dt: float,
    steps: int,
    snapshot_stride: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    a = np.asarray(initial_coefficients, dtype=float).copy()
    coefficients = [a.copy()]
    times = [0.0]
    started = perf_counter()
    for step in range(1, int(steps) + 1):
        a += float(dt) * modal_rhs(a)
        if not np.isfinite(a).all():
            raise FloatingPointError("reduced trajectory became non-finite")
        if step % snapshot_stride == 0 or step == steps:
            coefficients.append(a.copy())
            times.append(step * float(dt))
    elapsed = perf_counter() - started
    return np.asarray(times), np.stack(coefficients), float(elapsed)


def simulate_pod_galerkin(
    model: PODModel,
    reynolds: float,
    dt: float,
    steps: int,
    snapshot_stride: int,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Integrate standard POD--Galerkin with full-order nonlinear evaluation."""
    q0 = np.zeros((model.n - 2) ** 2) if initial_state is None else initial_state
    a0 = project_state(model, q0)

    def modal_rhs(a: np.ndarray) -> np.ndarray:
        q = reconstruct_state(model, a)
        return model.modes.T @ rhs(q, reynolds, model.n, model.u_lid)

    times, coefficients, elapsed = _integrate_modal(
        modal_rhs, a0, dt, steps, snapshot_stride
    )
    states = model.mean[None, :] + coefficients @ model.modes.T
    return {
        "time": times,
        "coefficients": coefficients,
        "states": states,
        "elapsed_seconds": elapsed,
    }


def convection_snapshots(
    snapshot_sets: Iterable[np.ndarray],
    reynolds: float,
    n: int,
    u_lid: float = 1.0,
) -> np.ndarray:
    """Evaluate nonlinear convection for offline DEIM basis construction."""
    values = []
    for trajectory in snapshot_sets:
        for q in np.asarray(trajectory, dtype=float):
            convection, _ = rhs_terms(q, reynolds, n, u_lid)
            values.append(convection)
    return np.stack(values)


def fit_nonlinear_basis(
    nonlinear_snapshots: np.ndarray, max_dimension: int | None = None
) -> NonlinearBasis:
    """Fit the convection basis once for subsequent leakage-free DEIM sweeps."""
    nonlinear = np.asarray(nonlinear_snapshots, dtype=float)
    if nonlinear.ndim != 2 or nonlinear.shape[0] < 2:
        raise ValueError("nonlinear_snapshots must be a two-dimensional sample matrix")
    mean = nonlinear.mean(axis=0)
    modes, singular_values, _ = np.linalg.svd((nonlinear - mean).T, full_matrices=False)
    available = int(np.count_nonzero(singular_values > np.finfo(float).eps * singular_values[0]))
    dimension = available if max_dimension is None else int(max_dimension)
    if not 1 <= dimension <= available:
        raise ValueError(f"max_dimension must be between 1 and {available}")
    modes = modes[:, :dimension].copy()
    for column in range(dimension):
        pivot = int(np.argmax(np.abs(modes[:, column])))
        if modes[pivot, column] < 0:
            modes[:, column] *= -1.0
    return NonlinearBasis(mean=mean, modes=modes, singular_values=singular_values)


def _sample_kinematic_maps(
    pod: PODModel, indices: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Build affine sampled maps for velocity and vorticity gradients."""
    h = 1.0 / (pod.n - 1)

    def sample(q: np.ndarray, u_lid: float) -> tuple[np.ndarray, ...]:
        fields = state_to_fields(q, pod.n, u_lid)
        omega = fields["omega"]
        dwdx = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2.0 * h)
        dwdy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2.0 * h)
        return (
            fields["u"][1:-1, 1:-1].reshape(-1)[indices],
            fields["v"][1:-1, 1:-1].reshape(-1)[indices],
            dwdx.reshape(-1)[indices],
            dwdy.reshape(-1)[indices],
        )

    offsets = sample(pod.mean, pod.u_lid)
    columns = [sample(pod.modes[:, j], 0.0) for j in range(pod.rank)]
    matrices = tuple(
        np.column_stack([column[k] for column in columns]) for k in range(4)
    )
    return (*offsets, *matrices)


def fit_deim(
    pod: PODModel,
    nonlinear_snapshots: np.ndarray | NonlinearBasis,
    deim_dimension: int,
) -> DEIMModel:
    """Fit QDEIM and the exact unit-viscosity affine diffusion operator.

    The online model obtains diffusion at any Reynolds number by multiplication
    by ``1/Re``; blind parameters therefore require no data-dependent refit.
    """
    fitted = (
        nonlinear_snapshots
        if isinstance(nonlinear_snapshots, NonlinearBasis)
        else fit_nonlinear_basis(nonlinear_snapshots)
    )
    if fitted.mean.size != pod.mean.size:
        raise ValueError("nonlinear basis and POD state dimensions differ")
    if not 1 <= deim_dimension <= fitted.dimension:
        raise ValueError(f"deim_dimension must be between 1 and {fitted.dimension}")
    nonlinear_mean = fitted.mean
    nonlinear_basis = fitted.modes[:, :deim_dimension]
    _, _, pivots = qr(nonlinear_basis.T, pivoting=True, mode="economic")
    indices = np.asarray(pivots[:deim_dimension], dtype=int)
    sampled_basis = nonlinear_basis[indices, :]
    deim_projection = (pod.modes.T @ nonlinear_basis) @ np.linalg.inv(sampled_basis)

    _, diffusion_at_mean = rhs_terms(pod.mean, 1.0, pod.n, pod.u_lid)
    diffusion_columns = []
    for column in range(pod.rank):
        _, shifted = rhs_terms(
            pod.mean + pod.modes[:, column], 1.0, pod.n, pod.u_lid
        )
        diffusion_columns.append(shifted - diffusion_at_mean)
    diffusion_linear = np.column_stack(diffusion_columns)

    maps = _sample_kinematic_maps(pod, indices)
    return DEIMModel(
        pod=pod,
        indices=indices,
        nonlinear_basis=nonlinear_basis,
        nonlinear_mean=nonlinear_mean,
        diffusion_offset_unit=pod.modes.T @ diffusion_at_mean,
        diffusion_matrix_unit=pod.modes.T @ diffusion_linear,
        convection_offset=pod.modes.T @ nonlinear_mean,
        deim_projection=deim_projection,
        sampled_convection_mean=nonlinear_mean[indices],
        sampled_u_offset=maps[0],
        sampled_v_offset=maps[1],
        sampled_dwdx_offset=maps[2],
        sampled_dwdy_offset=maps[3],
        sampled_u_matrix=maps[4],
        sampled_v_matrix=maps[5],
        sampled_dwdx_matrix=maps[6],
        sampled_dwdy_matrix=maps[7],
    )


def sampled_convection(model: DEIMModel, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate cavity convection only at the DEIM interpolation locations."""
    a = np.asarray(coefficients, dtype=float).reshape(-1)
    u = model.sampled_u_offset + model.sampled_u_matrix @ a
    v = model.sampled_v_offset + model.sampled_v_matrix @ a
    dwdx = model.sampled_dwdx_offset + model.sampled_dwdx_matrix @ a
    dwdy = model.sampled_dwdy_offset + model.sampled_dwdy_matrix @ a
    return -(u * dwdx + v * dwdy)


def deim_modal_rhs(
    model: DEIMModel, coefficients: np.ndarray, reynolds: float
) -> np.ndarray:
    """Return the hyper-reduced modal right-hand side."""
    if reynolds <= 0:
        raise ValueError("reynolds must be positive")
    a = np.asarray(coefficients, dtype=float).reshape(-1)
    sampled = sampled_convection(model, a)
    convection = model.convection_offset + model.deim_projection @ (
        sampled - model.sampled_convection_mean
    )
    diffusion = (
        model.diffusion_offset_unit + model.diffusion_matrix_unit @ a
    ) / float(reynolds)
    return convection + diffusion


def simulate_pod_deim(
    model: DEIMModel,
    reynolds: float,
    dt: float,
    steps: int,
    snapshot_stride: int,
    initial_state: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Integrate the POD--DEIM model without full-field nonlinear evaluation."""
    q0 = np.zeros((model.pod.n - 2) ** 2) if initial_state is None else initial_state
    a0 = project_state(model.pod, q0)
    times, coefficients, elapsed = _integrate_modal(
        lambda a: deim_modal_rhs(model, a, reynolds), a0, dt, steps, snapshot_stride
    )
    states = model.pod.mean[None, :] + coefficients @ model.pod.modes.T
    return {
        "time": times,
        "coefficients": coefficients,
        "states": states,
        "elapsed_seconds": elapsed,
    }


def save_deim_model(path: str | Path, model: DEIMModel) -> Path:
    """Store a fitted POD--DEIM model as a portable, pickle-free NPZ archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format_version=np.asarray(CAVITY_ROM_VERSION),
        n=np.asarray(model.pod.n),
        u_lid=np.asarray(model.pod.u_lid),
        pod_mean=model.pod.mean,
        pod_modes=model.pod.modes,
        pod_singular_values=model.pod.singular_values,
        pod_cumulative_energy=model.pod.cumulative_energy,
        deim_indices=model.indices,
        nonlinear_basis=model.nonlinear_basis,
        nonlinear_mean=model.nonlinear_mean,
        diffusion_offset_unit=model.diffusion_offset_unit,
        diffusion_matrix_unit=model.diffusion_matrix_unit,
        convection_offset=model.convection_offset,
        deim_projection=model.deim_projection,
        sampled_convection_mean=model.sampled_convection_mean,
        sampled_u_offset=model.sampled_u_offset,
        sampled_v_offset=model.sampled_v_offset,
        sampled_dwdx_offset=model.sampled_dwdx_offset,
        sampled_dwdy_offset=model.sampled_dwdy_offset,
        sampled_u_matrix=model.sampled_u_matrix,
        sampled_v_matrix=model.sampled_v_matrix,
        sampled_dwdx_matrix=model.sampled_dwdx_matrix,
        sampled_dwdy_matrix=model.sampled_dwdy_matrix,
    )
    return target


def load_deim_model(path: str | Path) -> DEIMModel:
    """Load and dimension-check a model written by :func:`save_deim_model`."""
    with np.load(Path(path), allow_pickle=False) as archive:
        version = str(np.asarray(archive["format_version"]).item())
        if version != CAVITY_ROM_VERSION:
            raise ValueError(
                f"model format {version!r} is incompatible with {CAVITY_ROM_VERSION!r}"
            )
        pod = PODModel(
            n=int(np.asarray(archive["n"]).item()),
            u_lid=float(np.asarray(archive["u_lid"]).item()),
            mean=np.asarray(archive["pod_mean"], dtype=float),
            modes=np.asarray(archive["pod_modes"], dtype=float),
            singular_values=np.asarray(archive["pod_singular_values"], dtype=float),
            cumulative_energy=np.asarray(archive["pod_cumulative_energy"], dtype=float),
        )
        model = DEIMModel(
            pod=pod,
            indices=np.asarray(archive["deim_indices"], dtype=int),
            nonlinear_basis=np.asarray(archive["nonlinear_basis"], dtype=float),
            nonlinear_mean=np.asarray(archive["nonlinear_mean"], dtype=float),
            diffusion_offset_unit=np.asarray(archive["diffusion_offset_unit"], dtype=float),
            diffusion_matrix_unit=np.asarray(archive["diffusion_matrix_unit"], dtype=float),
            convection_offset=np.asarray(archive["convection_offset"], dtype=float),
            deim_projection=np.asarray(archive["deim_projection"], dtype=float),
            sampled_convection_mean=np.asarray(archive["sampled_convection_mean"], dtype=float),
            sampled_u_offset=np.asarray(archive["sampled_u_offset"], dtype=float),
            sampled_v_offset=np.asarray(archive["sampled_v_offset"], dtype=float),
            sampled_dwdx_offset=np.asarray(archive["sampled_dwdx_offset"], dtype=float),
            sampled_dwdy_offset=np.asarray(archive["sampled_dwdy_offset"], dtype=float),
            sampled_u_matrix=np.asarray(archive["sampled_u_matrix"], dtype=float),
            sampled_v_matrix=np.asarray(archive["sampled_v_matrix"], dtype=float),
            sampled_dwdx_matrix=np.asarray(archive["sampled_dwdx_matrix"], dtype=float),
            sampled_dwdy_matrix=np.asarray(archive["sampled_dwdy_matrix"], dtype=float),
        )
    expected = (model.pod.n - 2) ** 2
    if model.pod.mean.size != expected or model.pod.modes.shape != (expected, model.rank):
        raise ValueError("stored POD arrays have inconsistent dimensions")
    if model.nonlinear_basis.shape != (expected, model.deim_dimension):
        raise ValueError("stored nonlinear basis has inconsistent dimensions")
    return model


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return a guarded relative Euclidean error."""
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    return float(np.linalg.norm(prediction - reference) / (np.linalg.norm(reference) + 1e-30))


def velocity_error_trajectory(
    reference_states: np.ndarray,
    prediction_states: np.ndarray,
    n: int,
    u_lid: float = 1.0,
) -> np.ndarray:
    """Return relative velocity-field error at each retained time."""
    reference_states = np.asarray(reference_states, dtype=float)
    prediction_states = np.asarray(prediction_states, dtype=float)
    if reference_states.shape != prediction_states.shape:
        raise ValueError("reference and prediction trajectories differ in shape")
    errors = []
    for truth, prediction in zip(reference_states, prediction_states):
        true_fields = state_to_fields(truth, n, u_lid)
        pred_fields = state_to_fields(prediction, n, u_lid)
        reference = np.concatenate([true_fields["u"].reshape(-1), true_fields["v"].reshape(-1)])
        estimate = np.concatenate([pred_fields["u"].reshape(-1), pred_fields["v"].reshape(-1)])
        errors.append(relative_l2(reference, estimate))
    return np.asarray(errors)


def physical_diagnostics(q: np.ndarray, n: int, u_lid: float = 1.0) -> dict[str, float]:
    """Return wall, divergence, and primary-vortex diagnostics for one state."""
    fields = state_to_fields(q, n, u_lid)
    h = 1.0 / (n - 1)
    u, v, psi = fields["u"], fields["v"], fields["psi"]
    divergence = (
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * h)
        + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * h)
    )
    wall_values = np.concatenate(
        [
            u[0, :],
            u[1:-1, 0],
            u[1:-1, -1],
            u[-1, 1:-1] - float(u_lid),
            v[0, :],
            v[-1, :],
            v[1:-1, 0],
            v[1:-1, -1],
        ]
    )
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    return {
        "divergence_l2": float(np.sqrt(np.mean(divergence**2))),
        "wall_rms_error": float(np.sqrt(np.mean(wall_values**2))),
        "vortex_x": float(i * h),
        "vortex_y": float(j * h),
        "psi_min": float(psi[j, i]),
    }


def nonlinear_deim_error(model: DEIMModel, states: np.ndarray) -> np.ndarray:
    """Measure a-posteriori full nonlinear error of the DEIM approximation."""
    errors = []
    for q in np.asarray(states, dtype=float):
        a = project_state(model.pod, q)
        full, _ = rhs_terms(reconstruct_state(model.pod, a), 1.0, model.pod.n, model.pod.u_lid)
        sampled = sampled_convection(model, a)
        approximation = (
            model.nonlinear_mean
            + model.nonlinear_basis
            @ np.linalg.solve(
                model.nonlinear_basis[model.indices, :],
                sampled - model.sampled_convection_mean,
            )
        )
        errors.append(relative_l2(full, approximation))
    return np.asarray(errors)
