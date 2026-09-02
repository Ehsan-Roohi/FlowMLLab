"""Utilities for a four-frame, multi-scale cylinder-wake predictor.

The TensorFlow dependency is optional.  Pure NumPy helpers remain importable in
the base FlowMLLab installation, while :func:`build_multiscale_predictor` and
:func:`composite_flow_loss` provide the Week-7 CNN when ``flowmllab[ml]`` is
installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    """Indices of consecutive history frames and their future target."""

    history: np.ndarray
    target: int


def temporal_windows(
    snapshot_count: int,
    *,
    history: int = 4,
    horizon: int = 1,
    target_stride: int = 1,
) -> list[TemporalWindow]:
    """Return leakage-free within-case temporal windows.

    Windows never cross a Reynolds-case boundary because callers invoke this
    helper once per complete case.
    """
    for name, value in (
        ("snapshot_count", snapshot_count),
        ("history", history),
        ("horizon", horizon),
        ("target_stride", target_stride),
    ):
        if int(value) != value or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    snapshot_count = int(snapshot_count)
    history = int(history)
    horizon = int(horizon)
    target_stride = int(target_stride)
    first_target = history + horizon - 1
    return [
        TemporalWindow(
            history=np.arange(target - horizon - history + 1, target - horizon + 1),
            target=target,
        )
        for target in range(first_target, snapshot_count, target_stride)
    ]


def stack_history(
    fields: Mapping[str, np.ndarray],
    indices: Sequence[int],
    *,
    names: Sequence[str] = ("u", "v", "p"),
) -> np.ndarray:
    """Stack history in time-major channel order ``[u,v,p] x history``."""
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size < 1:
        raise ValueError("indices must contain at least one frame")
    arrays = [np.asarray(fields[name], dtype=np.float32) for name in names]
    shape = arrays[0].shape
    if len(shape) != 3 or any(array.shape != shape for array in arrays):
        raise ValueError("all fields must have shape (time, y, x)")
    if np.any(idx < 0) or np.any(idx >= shape[0]):
        raise IndexError("a history index is outside the available snapshots")
    return np.concatenate(
        [np.stack([array[i] for array in arrays], axis=-1) for i in idx], axis=-1
    )


def vorticity(u: np.ndarray, v: np.ndarray, *, diameter: float = 1.0) -> np.ndarray:
    """Dimensionless spanwise vorticity ``omega D/U`` on a unit lattice."""
    u_values = np.asarray(u, dtype=float)
    v_values = np.asarray(v, dtype=float)
    if u_values.shape != v_values.shape or u_values.ndim < 2:
        raise ValueError("u and v must have identical arrays with spatial axes last")
    return float(diameter) * (
        np.gradient(v_values, axis=-1) - np.gradient(u_values, axis=-2)
    )


def divergence(u: np.ndarray, v: np.ndarray, *, diameter: float = 1.0) -> np.ndarray:
    """Dimensionless divergence ``D div(u/U)`` on a unit lattice."""
    u_values = np.asarray(u, dtype=float)
    v_values = np.asarray(v, dtype=float)
    if u_values.shape != v_values.shape or u_values.ndim < 2:
        raise ValueError("u and v must have identical arrays with spatial axes last")
    return float(diameter) * (
        np.gradient(u_values, axis=-1) + np.gradient(v_values, axis=-2)
    )


def relative_l2(truth: np.ndarray, prediction: np.ndarray) -> float:
    """Return a transparent global relative L2 error."""
    exact = np.asarray(truth, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if exact.shape != estimate.shape:
        raise ValueError("truth and prediction must have identical shapes")
    denominator = max(float(np.linalg.norm(exact)), np.finfo(float).eps)
    return float(np.linalg.norm(estimate - exact) / denominator)


def stationwise_wake_metrics(
    truth_omega: np.ndarray,
    prediction_omega: np.ndarray,
    *,
    center_x: float,
    diameter: float,
    stations: Sequence[float] = (2.0, 4.0, 6.0, 8.0),
) -> list[dict[str, float]]:
    """Compare vorticity amplitude and cross-wake spectra downstream.

    Enstrophy is integrated over time and the transverse direction.  The PSD is
    the transverse spatial spectrum, averaged over time; a normalized PSD error
    separates spectral-shape loss from simple amplitude loss.
    """
    truth = np.asarray(truth_omega, dtype=float)
    prediction = np.asarray(prediction_omega, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 3:
        raise ValueError("vorticity arrays must share shape (time, y, x)")
    if diameter <= 0:
        raise ValueError("diameter must be positive")
    rows: list[dict[str, float]] = []
    for station in stations:
        column = int(round(float(center_x) + float(station) * float(diameter)))
        if not 0 <= column < truth.shape[-1]:
            raise ValueError(f"station x/D={station:g} lies outside the grid")
        exact = truth[:, :, column]
        estimate = prediction[:, :, column]
        exact_enstrophy = float(np.mean(exact**2))
        estimate_enstrophy = float(np.mean(estimate**2))
        enstrophy_ratio = estimate_enstrophy / max(
            exact_enstrophy, np.finfo(float).eps
        )
        exact_psd = np.mean(np.abs(np.fft.rfft(exact, axis=1)) ** 2, axis=0)
        estimate_psd = np.mean(
            np.abs(np.fft.rfft(estimate, axis=1)) ** 2, axis=0
        )
        exact_psd_norm = exact_psd / max(float(exact_psd.sum()), np.finfo(float).eps)
        estimate_psd_norm = estimate_psd / max(
            float(estimate_psd.sum()), np.finfo(float).eps
        )
        rows.append(
            {
                "x_over_d": float(station),
                "grid_column": float(column),
                "vorticity_profile_relative_l2": relative_l2(exact, estimate),
                "enstrophy_ratio": float(enstrophy_ratio),
                "enstrophy_relative_error": float(abs(enstrophy_ratio - 1.0)),
                "normalized_psd_relative_l2": relative_l2(
                    exact_psd_norm, estimate_psd_norm
                ),
            }
        )
    return rows


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise ImportError(
            "The cylinder CNN requires the optional ML dependencies: "
            "python -m pip install 'flowmllab[ml]'"
        ) from error
    return tf


def build_multiscale_predictor(
    *,
    history: int = 4,
    fields_per_frame: int = 3,
    filters: int = 24,
) -> Any:
    """Build a fully convolutional residual predictor with three spatial scales.

    Input channels are time-major flow fields followed by standardized Reynolds
    number and a fluid mask.  The model starts from exact persistence because the
    final residual convolution is zero initialized.
    """
    if history < 1 or fields_per_frame != 3 or filters < 4:
        raise ValueError("history>=1, fields_per_frame=3, and filters>=4 are required")
    tf = _tensorflow()
    keras = tf.keras
    flow_channels = int(history) * int(fields_per_frame)
    inputs = keras.Input(shape=(None, None, flow_channels + 2), name="history_re_mask")

    def conv_block(values: Any, width: int, kernel: int, name: str) -> Any:
        values = keras.layers.Conv2D(
            width, kernel, padding="same", activation="swish", name=f"{name}_conv1"
        )(values)
        return keras.layers.Conv2D(
            width, 3, padding="same", activation="swish", name=f"{name}_conv2"
        )(values)

    fine = conv_block(inputs, filters, 5, "fine")
    half = keras.layers.AveragePooling2D(2, name="half_pool")(inputs)
    half = conv_block(half, filters, 5, "half")
    half = keras.layers.UpSampling2D(2, interpolation="bilinear", name="half_up")(half)
    quarter = keras.layers.AveragePooling2D(4, name="quarter_pool")(inputs)
    quarter = conv_block(quarter, filters, 5, "quarter")
    quarter = keras.layers.UpSampling2D(
        4, interpolation="bilinear", name="quarter_up"
    )(quarter)
    merged = keras.layers.Concatenate(name="multiscale_concat")([fine, half, quarter])
    merged = keras.layers.Conv2D(
        2 * filters, 3, padding="same", activation="swish", name="fusion"
    )(merged)
    merged = keras.layers.Conv2D(
        filters, 3, padding="same", activation="swish", name="refine"
    )(merged)
    delta = keras.layers.Conv2D(
        3,
        3,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="future_increment",
    )(merged)
    previous = keras.layers.Lambda(
        lambda values: values[..., flow_channels - 3 : flow_channels],
        name="latest_frame",
    )(inputs)
    raw = keras.layers.Add(name="residual_prediction")([previous, delta])
    fluid = keras.layers.Lambda(lambda values: values[..., -1:], name="fluid_mask")(
        inputs
    )
    velocity = keras.layers.Multiply(name="exact_no_slip")([raw[..., :2], fluid])
    outputs = keras.layers.Concatenate(name="predicted_u_v_p")(
        [velocity, raw[..., 2:3]]
    )
    return keras.Model(inputs=inputs, outputs=outputs, name="cylinder_multiscale_cnn")


def composite_flow_loss(
    *,
    field_scales: Sequence[float],
    diameter: float,
    vorticity_scale: float,
    gradient_weight: float = 0.20,
    vorticity_weight: float = 0.20,
    divergence_weight: float = 0.05,
) -> Any:
    """Construct field + gradient + vorticity + divergence loss."""
    scales = np.asarray(field_scales, dtype=np.float32).reshape(-1)
    if scales.size != 3 or np.any(scales <= 0):
        raise ValueError("field_scales must contain three positive values")
    if diameter <= 0 or vorticity_scale <= 0:
        raise ValueError("diameter and vorticity_scale must be positive")
    if min(gradient_weight, vorticity_weight, divergence_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    tf = _tensorflow()
    scales_tensor = tf.constant(scales.reshape((1, 1, 1, 3)))
    diameter_tensor = tf.constant(float(diameter), dtype=tf.float32)
    omega_scale_tensor = tf.constant(float(vorticity_scale), dtype=tf.float32)

    def centered_x(values: Any) -> Any:
        return 0.5 * (values[:, 1:-1, 2:, :] - values[:, 1:-1, :-2, :])

    def centered_y(values: Any) -> Any:
        return 0.5 * (values[:, 2:, 1:-1, :] - values[:, :-2, 1:-1, :])

    def loss(y_true: Any, y_prediction: Any) -> Any:
        field = tf.reduce_mean(tf.abs((y_prediction - y_true) / scales_tensor))
        true_dx, prediction_dx = centered_x(y_true), centered_x(y_prediction)
        true_dy, prediction_dy = centered_y(y_true), centered_y(y_prediction)
        gradient = tf.reduce_mean(
            tf.abs((prediction_dx - true_dx) / scales_tensor)
        ) + tf.reduce_mean(tf.abs((prediction_dy - true_dy) / scales_tensor))
        true_omega = diameter_tensor * (true_dx[..., 1] - true_dy[..., 0])
        prediction_omega = diameter_tensor * (
            prediction_dx[..., 1] - prediction_dy[..., 0]
        )
        omega = tf.reduce_mean(
            tf.abs(prediction_omega - true_omega) / omega_scale_tensor
        )
        prediction_divergence = diameter_tensor * (
            prediction_dx[..., 0] + prediction_dy[..., 1]
        )
        div = tf.reduce_mean(tf.abs(prediction_divergence))
        return (
            field
            + float(gradient_weight) * gradient
            + float(vorticity_weight) * omega
            + float(divergence_weight) * div
        )

    loss.__name__ = "field_gradient_vorticity_divergence_loss"
    return loss


__all__ = [
    "TemporalWindow",
    "build_multiscale_predictor",
    "composite_flow_loss",
    "divergence",
    "relative_l2",
    "stack_history",
    "stationwise_wake_metrics",
    "temporal_windows",
    "vorticity",
]
