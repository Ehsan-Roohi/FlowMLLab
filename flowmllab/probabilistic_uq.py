"""Small, auditable probabilistic models and scores for CFD surrogate studies.

The routines in this module deliberately favor transparent linear algebra and
fixed-protocol Gaussian processes over a large probabilistic-programming stack.
They are intended for teaching, calibration checks, and bounded CFD surrogate
experiments rather than production uncertainty certification.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


Array = np.ndarray


@dataclass(frozen=True)
class GaussianPrediction:
    """Pointwise Gaussian predictive mean and standard deviation."""

    mean: Array
    std: Array

    def interval(self, level: float = 0.95) -> tuple[Array, Array]:
        """Return the central Gaussian interval at ``level``."""
        return gaussian_prediction_interval(self.mean, self.std, level=level)


@dataclass(frozen=True)
class BayesianLinearPosterior:
    """Conjugate posterior for a Gaussian linear observation model."""

    mean: Array
    covariance: Array
    noise_std: float


@dataclass(frozen=True)
class CalibrationCurve:
    """Nominal interval levels, observed coverage, and mean interval width."""

    nominal: Array
    observed: Array
    mean_width: Array


@dataclass(frozen=True)
class PODGaussianProcess:
    """POD field representation with one independent GP per coefficient.

    Independence is a declared approximation. The POD basis and every GP are
    fitted only from ``train_reynolds`` and the corresponding field snapshots.
    """

    field_mean: Array
    basis: Array
    regressors: tuple[GaussianProcessRegressor, ...]
    reynolds_center: float
    reynolds_scale: float
    train_reynolds: Array
    field_shape: tuple[int, ...]

    @property
    def rank(self) -> int:
        """Number of retained POD coefficients."""
        return int(self.basis.shape[0])


def _as_design(design: Array) -> Array:
    matrix = np.asarray(design, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("design must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("design contains non-finite values")
    return matrix


def fit_bayesian_linear_regression(
    design: Array,
    observations: Array,
    *,
    prior_mean: Array,
    prior_covariance: Array,
    noise_std: float,
) -> BayesianLinearPosterior:
    """Fit an exact Gaussian-prior, Gaussian-likelihood linear model.

    The observation model is ``y = design @ weights + epsilon`` with
    ``epsilon ~ Normal(0, noise_std**2)``.
    """
    matrix = _as_design(design)
    targets = np.asarray(observations, dtype=float).reshape(-1)
    mean0 = np.asarray(prior_mean, dtype=float).reshape(-1)
    covariance0 = np.asarray(prior_covariance, dtype=float)
    parameter_count = matrix.shape[1]

    if targets.shape != (matrix.shape[0],):
        raise ValueError("observations must have one value per design row")
    if mean0.shape != (parameter_count,):
        raise ValueError("prior_mean has the wrong number of parameters")
    if covariance0.shape != (parameter_count, parameter_count):
        raise ValueError("prior_covariance has the wrong shape")
    if not np.isfinite(targets).all() or not np.isfinite(mean0).all():
        raise ValueError("observations and prior_mean must be finite")
    if not np.isfinite(covariance0).all() or noise_std <= 0.0:
        raise ValueError("prior_covariance must be finite and noise_std positive")

    try:
        np.linalg.cholesky(covariance0)
        prior_precision = np.linalg.solve(covariance0, np.eye(parameter_count))
    except np.linalg.LinAlgError as error:
        raise ValueError("prior_covariance must be positive definite") from error
    precision = prior_precision + matrix.T @ matrix / noise_std**2
    natural_mean = prior_precision @ mean0 + matrix.T @ targets / noise_std**2
    try:
        covariance = np.linalg.inv(precision)
    except np.linalg.LinAlgError as error:
        raise ValueError("posterior precision is singular") from error
    mean = covariance @ natural_mean
    return BayesianLinearPosterior(
        mean=mean,
        covariance=0.5 * (covariance + covariance.T),
        noise_std=float(noise_std),
    )


def predict_bayesian_linear_regression(
    posterior: BayesianLinearPosterior,
    design: Array,
    *,
    include_observation_noise: bool = True,
) -> GaussianPrediction:
    """Evaluate the latent or observation-level posterior predictive."""
    matrix = _as_design(design)
    if matrix.shape[1] != posterior.mean.size:
        raise ValueError("design and posterior parameter counts do not match")
    mean = matrix @ posterior.mean
    variance = np.einsum(
        "ij,jk,ik->i", matrix, posterior.covariance, matrix, optimize=True
    )
    if include_observation_noise:
        variance = variance + posterior.noise_std**2
    return GaussianPrediction(mean=mean, std=np.sqrt(np.maximum(variance, 0.0)))


def gaussian_prediction_interval(
    mean: Array, std: Array, *, level: float = 0.95
) -> tuple[Array, Array]:
    """Return a central Gaussian interval with broadcast-compatible inputs."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    center, scale = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    )
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("mean and std must be finite")
    if np.any(scale < 0.0):
        raise ValueError("std cannot be negative")
    quantile = float(norm.ppf(0.5 * (1.0 + level)))
    return center - quantile * scale, center + quantile * scale


def gaussian_negative_log_likelihood(
    observations: Array, mean: Array, std: Array, *, reduction: str = "mean"
) -> float | Array:
    """Evaluate Gaussian negative log likelihood as a proper score."""
    truth, center, scale = np.broadcast_arrays(
        np.asarray(observations, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(std, dtype=float),
    )
    if not np.isfinite(truth).all() or not np.isfinite(center).all():
        raise ValueError("observations and mean must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("std must be finite and strictly positive")
    values = 0.5 * np.log(2.0 * np.pi * scale**2) + 0.5 * (
        (truth - center) / scale
    ) ** 2
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def gaussian_crps(
    observations: Array, mean: Array, std: Array, *, reduction: str = "mean"
) -> float | Array:
    """Evaluate the continuous ranked probability score for Gaussians."""
    truth, center, scale = np.broadcast_arrays(
        np.asarray(observations, dtype=float),
        np.asarray(mean, dtype=float),
        np.asarray(std, dtype=float),
    )
    if not np.isfinite(truth).all() or not np.isfinite(center).all():
        raise ValueError("observations and mean must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("std must be finite and strictly positive")
    z_score = (truth - center) / scale
    values = scale * (
        z_score * (2.0 * norm.cdf(z_score) - 1.0)
        + 2.0 * norm.pdf(z_score)
        - 1.0 / np.sqrt(np.pi)
    )
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def calibration_curve(
    observations: Array,
    prediction: GaussianPrediction,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> CalibrationCurve:
    """Measure central-interval coverage and width at declared levels."""
    truth, center, scale = np.broadcast_arrays(
        np.asarray(observations, dtype=float),
        np.asarray(prediction.mean, dtype=float),
        np.asarray(prediction.std, dtype=float),
    )
    nominal = np.asarray(tuple(levels), dtype=float)
    if nominal.ndim != 1 or nominal.size == 0:
        raise ValueError("levels must be a non-empty one-dimensional sequence")
    if np.any((nominal <= 0.0) | (nominal >= 1.0)):
        raise ValueError("every calibration level must lie between zero and one")
    if not np.isfinite(truth).all() or not np.isfinite(center).all():
        raise ValueError("observations and prediction mean must be finite")
    if not np.isfinite(scale).all() or np.any(scale < 0.0):
        raise ValueError("prediction std must be finite and nonnegative")

    observed = np.empty_like(nominal)
    mean_width = np.empty_like(nominal)
    for index, level in enumerate(nominal):
        lower, upper = gaussian_prediction_interval(center, scale, level=float(level))
        observed[index] = np.mean((truth >= lower) & (truth <= upper))
        mean_width[index] = np.mean(upper - lower)
    return CalibrationCurve(
        nominal=nominal,
        observed=observed,
        mean_width=mean_width,
    )


def validation_scale_factor(
    observations: Array,
    prediction: GaussianPrediction,
    *,
    target_level: float = 0.9,
    minimum_std: float = 1.0e-12,
) -> float:
    """Fit one interval-width multiplier using validation residuals only.

    This is a descriptive calibration aid, not a coverage guarantee for
    spatially correlated CFD nodes or shifted physical cases.
    """
    if not 0.0 < target_level < 1.0:
        raise ValueError("target_level must lie strictly between zero and one")
    if minimum_std <= 0.0:
        raise ValueError("minimum_std must be positive")
    truth, center, scale = np.broadcast_arrays(
        np.asarray(observations, dtype=float),
        np.asarray(prediction.mean, dtype=float),
        np.asarray(prediction.std, dtype=float),
    )
    if not np.isfinite(truth).all() or not np.isfinite(center).all():
        raise ValueError("observations and prediction mean must be finite")
    if not np.isfinite(scale).all() or np.any(scale < 0.0):
        raise ValueError("prediction std must be finite and nonnegative")
    standardized_error = np.abs(truth - center) / np.maximum(scale, minimum_std)
    required_radius = float(
        np.quantile(standardized_error.reshape(-1), target_level, method="higher")
    )
    gaussian_radius = float(norm.ppf(0.5 * (1.0 + target_level)))
    return max(required_radius / gaussian_radius, np.finfo(float).eps)


def rescale_prediction(
    prediction: GaussianPrediction, scale_factor: float
) -> GaussianPrediction:
    """Multiply predictive standard deviations without changing their mean."""
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("scale_factor must be finite and positive")
    return GaussianPrediction(
        mean=np.asarray(prediction.mean, dtype=float).copy(),
        std=np.asarray(prediction.std, dtype=float) * float(scale_factor),
    )


def fit_pod_gaussian_process(
    reynolds: Array,
    fields: Array,
    *,
    rank: int,
    length_scale: float,
    noise_level: float = 1.0e-8,
) -> PODGaussianProcess:
    """Fit a fixed-hyperparameter POD--GP model to complete CFD cases."""
    re_values = np.asarray(reynolds, dtype=float).reshape(-1)
    snapshots = np.asarray(fields, dtype=float)
    if snapshots.ndim < 2 or snapshots.shape[0] != re_values.size:
        raise ValueError("fields must have one complete snapshot per Reynolds number")
    if re_values.size < 3 or np.unique(re_values).size != re_values.size:
        raise ValueError("at least three unique Reynolds-number cases are required")
    if not np.isfinite(re_values).all() or not np.isfinite(snapshots).all():
        raise ValueError("reynolds and fields must be finite")
    maximum_rank = min(re_values.size - 1, int(np.prod(snapshots.shape[1:])))
    if not 1 <= rank <= maximum_rank:
        raise ValueError(f"rank must be between 1 and {maximum_rank}")
    if length_scale <= 0.0 or noise_level <= 0.0:
        raise ValueError("length_scale and noise_level must be positive")

    field_shape = tuple(int(value) for value in snapshots.shape[1:])
    flattened = snapshots.reshape(re_values.size, -1)
    field_mean = flattened.mean(axis=0)
    centered = flattened - field_mean
    _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = right_vectors[:rank]
    coefficients = centered @ basis.T

    reynolds_center = float(re_values.mean())
    reynolds_scale = float(re_values.std())
    if reynolds_scale <= 0.0:
        raise ValueError("training Reynolds numbers must have nonzero spread")
    inputs = ((re_values - reynolds_center) / reynolds_scale).reshape(-1, 1)

    regressors: list[GaussianProcessRegressor] = []
    for coefficient in coefficients.T:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale, length_scale_bounds="fixed"
        ) + WhiteKernel(noise_level, noise_level_bounds="fixed")
        regressor = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            optimizer=None,
            normalize_y=True,
        )
        regressor.fit(inputs, coefficient)
        regressors.append(regressor)

    return PODGaussianProcess(
        field_mean=field_mean,
        basis=basis,
        regressors=tuple(regressors),
        reynolds_center=reynolds_center,
        reynolds_scale=reynolds_scale,
        train_reynolds=re_values.copy(),
        field_shape=field_shape,
    )


def predict_pod_gaussian_process(
    model: PODGaussianProcess, reynolds: Array | float
) -> GaussianPrediction:
    """Predict field means and marginal standard deviations from a POD--GP."""
    queries = np.asarray(reynolds, dtype=float)
    scalar_input = queries.ndim == 0
    query_vector = queries.reshape(-1)
    if not np.isfinite(query_vector).all():
        raise ValueError("query Reynolds numbers must be finite")
    inputs = (
        (query_vector - model.reynolds_center) / model.reynolds_scale
    ).reshape(-1, 1)

    coefficient_means = []
    coefficient_variances = []
    for regressor in model.regressors:
        mean, std = regressor.predict(inputs, return_std=True)
        coefficient_means.append(mean)
        coefficient_variances.append(std**2)
    means = np.column_stack(coefficient_means)
    variances = np.column_stack(coefficient_variances)
    flattened_mean = model.field_mean + means @ model.basis
    flattened_variance = variances @ (model.basis**2)
    output_shape = (query_vector.size, *model.field_shape)
    prediction = GaussianPrediction(
        mean=flattened_mean.reshape(output_shape),
        std=np.sqrt(np.maximum(flattened_variance, 0.0)).reshape(output_shape),
    )
    if scalar_input:
        return GaussianPrediction(mean=prediction.mean[0], std=prediction.std[0])
    return prediction


def interpolate_complete_cases(
    train_reynolds: Array, train_fields: Array, query_reynolds: Array | float
) -> Array:
    """Piecewise-linear complete-case baseline without extrapolation."""
    re_values = np.asarray(train_reynolds, dtype=float).reshape(-1)
    fields = np.asarray(train_fields, dtype=float)
    queries = np.asarray(query_reynolds, dtype=float)
    scalar_input = queries.ndim == 0
    query_vector = queries.reshape(-1)
    if fields.ndim < 2 or fields.shape[0] != re_values.size:
        raise ValueError("train_fields must have one field per Reynolds number")
    if re_values.size < 2 or np.unique(re_values).size != re_values.size:
        raise ValueError("at least two unique training Reynolds numbers are required")
    if not np.isfinite(re_values).all() or not np.isfinite(fields).all():
        raise ValueError("training data must be finite")
    if not np.isfinite(query_vector).all():
        raise ValueError("query Reynolds numbers must be finite")

    order = np.argsort(re_values)
    re_sorted = re_values[order]
    fields_sorted = fields[order]
    if np.any(query_vector < re_sorted[0]) or np.any(query_vector > re_sorted[-1]):
        raise ValueError("interpolation queries must stay inside training support")
    upper = np.searchsorted(re_sorted, query_vector, side="right")
    upper = np.clip(upper, 1, re_sorted.size - 1)
    lower = upper - 1
    weight = (query_vector - re_sorted[lower]) / (
        re_sorted[upper] - re_sorted[lower]
    )
    reshape = (query_vector.size,) + (1,) * (fields.ndim - 1)
    result = (
        (1.0 - weight.reshape(reshape)) * fields_sorted[lower]
        + weight.reshape(reshape) * fields_sorted[upper]
    )
    return result[0] if scalar_input else result


def relative_l2_per_case(truth: Array, prediction: Array) -> Array:
    """Return one relative-L2 error for each complete case."""
    reference = np.asarray(truth, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if reference.shape != estimate.shape or reference.ndim < 2:
        raise ValueError("truth and prediction must have matching complete-case shapes")
    residual = (estimate - reference).reshape(reference.shape[0], -1)
    denominator = reference.reshape(reference.shape[0], -1)
    norms = np.linalg.norm(denominator, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("each truth case must have a nonzero norm")
    return np.linalg.norm(residual, axis=1) / norms


def validate_probabilistic_uq_evidence(root: str | Path) -> dict[str, object]:
    """Fail closed if the retained probabilistic-UQ protocol or evidence drifts."""
    repository = Path(root)
    result_dir = repository / "results" / "probabilistic_uq"
    required = (
        "blind_metrics.csv",
        "calibration.csv",
        "protocol.json",
        "summary.json",
        "probabilistic_uq_validation.png",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise ValueError("Missing probabilistic-UQ evidence: " + ", ".join(missing))

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((result_dir / "protocol.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(result_dir / "blind_metrics.csv")
    calibration = pd.read_csv(result_dir / "calibration.csv")
    dataset_path = repository / protocol["dataset"]
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_hash != protocol["dataset_sha256"]:
        raise ValueError("Probabilistic-UQ cavity dataset hash mismatch")

    training = set(float(value) for value in protocol["training_cases"])
    validation = {float(protocol["validation_case"])}
    blind = set(float(value) for value in protocol["blind_cases"])
    if training & validation or training & blind or validation & blind:
        raise ValueError("Probabilistic-UQ physical-case roles overlap")
    if blind != {175.0, 275.0, 375.0} or set(metrics["Re"]) != blind:
        raise ValueError("Probabilistic-UQ blind-case contract drifted")
    if len(calibration) != 12 or set(calibration["Re"]) != blind:
        raise ValueError("Probabilistic-UQ calibration table is incomplete")
    numeric_metrics = metrics.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric_metrics).all():
        raise ValueError("Probabilistic-UQ metrics contain non-finite values")
    if float(summary["mean_pod_gp_relative_L2_uv"]) >= 0.01:
        raise ValueError("Probabilistic-UQ mean-field accuracy gate failed")
    if float(summary["max_wall_error"]) >= 1.0e-12:
        raise ValueError("Probabilistic-UQ wall-condition gate failed")
    if float(summary["max_divergence_L2"]) >= 1.0e-12:
        raise ValueError("Probabilistic-UQ divergence gate failed")
    raw_coverage = float(summary["aggregate_raw_90_coverage"])
    calibrated_coverage = float(summary["aggregate_calibrated_90_coverage"])
    if not raw_coverage < calibrated_coverage < 0.9:
        raise ValueError("The retained blind under-coverage result drifted")
    declaration = str(protocol.get("originality_declaration", ""))
    if "No restricted course" not in declaration:
        raise ValueError("Probabilistic-UQ originality declaration is missing")
    if "not an independent-sample coverage guarantee" not in str(
        summary["coverage_interpretation"]
    ):
        raise ValueError("Probabilistic-UQ coverage claim boundary is missing")
    return summary


__all__ = [
    "BayesianLinearPosterior",
    "CalibrationCurve",
    "GaussianPrediction",
    "PODGaussianProcess",
    "calibration_curve",
    "fit_bayesian_linear_regression",
    "fit_pod_gaussian_process",
    "gaussian_crps",
    "gaussian_negative_log_likelihood",
    "gaussian_prediction_interval",
    "interpolate_complete_cases",
    "predict_bayesian_linear_regression",
    "predict_pod_gaussian_process",
    "relative_l2_per_case",
    "rescale_prediction",
    "validate_probabilistic_uq_evidence",
    "validation_scale_factor",
]
