"""Auditable linear-Gaussian state estimation for reduced flow coordinates."""
from dataclasses import dataclass
import numpy as np

from .cylinder_ml import fit_pod, project_pod, reconstruct_pod
from .modal_tools import fit_dmd, rollout_dmd, select_sensors


def _sym_psd(matrix, floor=1e-10):
    matrix = np.asarray(matrix, float)
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    scale = max(float(values.max(initial=0)), 1.0)
    return (vectors * np.maximum(values, floor * scale)) @ vectors.T


@dataclass
class ReducedKalmanModel:
    mean: np.ndarray
    modes: np.ndarray
    transition: np.ndarray
    process_covariance: np.ndarray
    observation_covariance: np.ndarray
    sensor_indices: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray

    @property
    def observation(self):
        return self.modes[self.sensor_indices]


def fit_reduced_kalman(fields, rank, sensor_count, noise_sigma, process_scale=1.0):
    """Fit POD, dynamics, sensors and noise covariances on training fields only."""
    fields = np.asarray(fields, float)
    if fields.ndim != 2 or len(fields) < 5 or rank < 1 or sensor_count < rank:
        raise ValueError('Invalid training fields, rank or sensor count')
    if not np.isfinite(fields).all() or not np.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError('Training fields and noise_sigma must be finite')
    pod = fit_pod(fields, rank=int(rank))
    states = project_pod(pod, fields)
    transition = fit_dmd(states)
    residual = states[1:] - states[:-1] @ transition
    process = np.cov(residual, rowvar=False, ddof=1)
    if not np.isfinite(process_scale) or process_scale <= 0:
        raise ValueError('process_scale must be positive')
    process = _sym_psd(np.atleast_2d(process)) * float(process_scale)
    sensors = select_sensors(pod.modes, int(sensor_count))
    observation_covariance = np.eye(len(sensors)) * float(noise_sigma) ** 2
    initial_covariance = _sym_psd(np.cov(states, rowvar=False, ddof=1))
    return ReducedKalmanModel(pod.mean, pod.modes, transition, process,
        observation_covariance, sensors, states[-1].copy(), initial_covariance)


def kalman_filter(model, observations):
    """Causal predict-update recursion; output at k uses observations through k."""
    y = np.asarray(observations, float)
    h = model.observation
    if y.ndim != 2 or y.shape[1] != len(model.sensor_indices) or not np.isfinite(y).all():
        raise ValueError('Invalid observation array')
    x = model.initial_mean.copy()
    p = model.initial_covariance.copy()
    identity = np.eye(len(x))
    states, covariances, innovations = [], [], []
    for measurement in y:
        x = x @ model.transition
        p = model.transition.T @ p @ model.transition + model.process_covariance
        innovation = measurement - model.mean[model.sensor_indices] - h @ x
        innovation_cov = _sym_psd(h @ p @ h.T + model.observation_covariance)
        gain = np.linalg.solve(innovation_cov, h @ p).T
        x = x + gain @ innovation
        # Joseph form avoids loss of positive semidefiniteness.
        correction = identity - gain @ h
        p = correction @ p @ correction.T + gain @ model.observation_covariance @ gain.T
        p = _sym_psd(p)
        states.append(x.copy()); covariances.append(p.copy()); innovations.append(innovation.copy())
    return np.asarray(states), np.asarray(covariances), np.asarray(innovations)


def reconstruct_states(model, states):
    states = np.asarray(states, float)
    return model.mean + states @ model.modes.T


def open_loop(model, steps):
    return reconstruct_states(model, rollout_dmd(model.transition, model.initial_mean, steps))


def sensor_only(model, observations):
    y = np.asarray(observations, float) - model.mean[model.sensor_indices]
    states = np.linalg.lstsq(model.observation, y.T, rcond=None)[0].T
    return reconstruct_states(model, states)


def pointwise_interval_coverage(model, states, covariances, truth, z=1.96):
    """Descriptive marginal POD-space Gaussian intervals, not simultaneous bands."""
    prediction = reconstruct_states(model, states)
    variances = np.einsum('ir,trs,is->ti', model.modes, covariances, model.modes)
    standard = np.sqrt(np.maximum(variances, 0))
    truth = np.asarray(truth, float)
    return float(np.mean(np.abs(truth - prediction) <= z * standard)), float(np.mean(2*z*standard))
