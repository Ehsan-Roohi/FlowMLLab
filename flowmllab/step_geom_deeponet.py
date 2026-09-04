"""Leakage-free Geom-DeepONet utilities for the rarefied micro-step.

This module adapts the architecture of He et al. (2024) to the two-dimensional
backward-facing-step data used by FlowMLLab.  The geometry branch receives only
declared case parameters.  The point trunk receives normalized coordinates and
an analytic signed-distance value.  No velocity patch, target-derived mask, or
held-out field can enter either model input.

The implementation retains the distinctive Geom-DeepONet stages: an initial
branch/trunk encoding, element-wise feature fusion, global pooling, a SIREN
post-fusion trunk, and a final branch--trunk contraction.  It supports any
number of query points at inference, although the present dataset establishes
generalization only over the single geometry parameter ``h/H`` at fixed
``Kn=0.01``.

Reference
---------
J. He et al., "Geom-DeepONet: A point-cloud-based deep operator network for
field predictions on 3D parameterized geometries," Computer Methods in Applied
Mechanics and Engineering 429 (2024) 117130.
https://doi.org/10.1016/j.cma.2024.117130
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

GEOM_DEEPONET_DOI = "10.1016/j.cma.2024.117130"


@dataclass(frozen=True)
class StepDomain:
    """Physical bounds for one rectangular channel with a backward-facing step."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    step_x_m: float = 25.0e-9

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.x_min_m, self.x_max_m, self.y_min_m, self.y_max_m, self.step_x_m],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("step-domain coordinates must be finite")
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("step-domain upper bounds must exceed lower bounds")
        if not self.x_min_m < self.step_x_m < self.x_max_m:
            raise ValueError("step_x_m must lie strictly inside the streamwise bounds")

    @property
    def length_m(self) -> float:
        return float(self.x_max_m - self.x_min_m)

    @property
    def height_m(self) -> float:
        return float(self.y_max_m - self.y_min_m)

    @property
    def length_over_height(self) -> float:
        return self.length_m / self.height_m


def infer_step_domain(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    step_x_m: float = 25.0e-9,
) -> StepDomain:
    """Infer physical wall locations from a cell-centred Cartesian point cloud.

    The published step coordinates are cell centres.  Using their extrema as
    wall locations would incorrectly assign zero SDF to the first fluid cells.
    The half-cell extension below recovers the physical channel bounds while
    tolerating the small decimal round-off in the exported Tecplot files.
    """

    x = np.unique(np.asarray(x_m, dtype=float).ravel())
    y = np.unique(np.asarray(y_m, dtype=float).ravel())
    if len(x) < 2 or len(y) < 2:
        raise ValueError("at least two unique x and y cell centres are required")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("step coordinates must be finite")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("step cell-centre coordinates must be strictly increasing")
    if not np.allclose(np.diff(x), dx, rtol=5.0e-3, atol=0.0):
        raise ValueError("x coordinates are not a near-uniform cell-centred grid")
    if not np.allclose(np.diff(y), dy, rtol=5.0e-3, atol=0.0):
        raise ValueError("y coordinates are not a near-uniform cell-centred grid")
    return StepDomain(
        x_min_m=float(x[0] - 0.5 * dx),
        x_max_m=float(x[-1] + 0.5 * dx),
        y_min_m=float(y[0] - 0.5 * dy),
        y_max_m=float(y[-1] + 0.5 * dy),
        step_x_m=float(step_x_m),
    )


def _distance_to_segment(
    x: np.ndarray,
    y: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    x0, y0 = start
    dx, dy = end[0] - x0, end[1] - y0
    denominator = dx * dx + dy * dy
    projection = np.clip(((x - x0) * dx + (y - y0) * dy) / denominator, 0.0, 1.0)
    return np.hypot(x - (x0 + projection * dx), y - (y0 + projection * dy))


def step_signed_distance(
    height_ratio: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    domain: StepDomain,
    normalized: bool = True,
) -> np.ndarray:
    """Return the exact polygon SDF for the backward-facing-step fluid domain.

    Positive values are inside the fluid, negative values are inside the solid
    step or outside the channel, and zero lies on the boundary.  With
    ``normalized=True`` (the model default), distance is divided by channel
    height so the Euclidean metric preserves the physical ``L/H`` aspect ratio.
    """

    h = float(height_ratio)
    if not 0.0 < h < 1.0:
        raise ValueError("height_ratio must lie strictly between zero and one")
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("SDF query coordinates must be finite")

    x0, x1 = domain.x_min_m, domain.x_max_m
    y0, y1 = domain.y_min_m, domain.y_max_m
    xs = domain.step_x_m
    ys = y0 + h * domain.height_m
    vertices = (
        (x0, ys),
        (xs, ys),
        (xs, y0),
        (x1, y0),
        (x1, y1),
        (x0, y1),
    )
    distances = [
        _distance_to_segment(x, y, start, end)
        for start, end in zip(vertices, vertices[1:] + vertices[:1], strict=True)
    ]
    distance = np.minimum.reduce(distances)
    in_box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    in_solid_step = (x < xs) & (y < ys)
    in_fluid = in_box & ~in_solid_step
    signed = np.where(in_fluid, distance, -distance)
    signed = np.where(distance <= 8.0 * np.finfo(float).eps * domain.height_m, 0.0, signed)
    if normalized:
        signed = signed / domain.height_m
    return signed


def step_geom_deeponet_inputs(
    height_ratio: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    domain: StepDomain,
    dtype: np.dtype[Any] | type = np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the only two inputs accepted by the micro-step Geom-DeepONet.

    Returns a one-row geometry branch ``[h/H]`` and a point trunk containing
    ``[2*x/L-1, 2*y/H-1, SDF/H]``.  The deliberately narrow signature makes it
    impossible to pass U, V, pressure, a vortex mask, or a target-field patch.
    """

    h = float(height_ratio)
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    x_normalized = 2.0 * (x - domain.x_min_m) / domain.length_m - 1.0
    y_normalized = 2.0 * (y - domain.y_min_m) / domain.height_m - 1.0
    sdf = step_signed_distance(h, x, y, domain=domain, normalized=True)
    branch = np.asarray([[h]], dtype=dtype)
    trunk = np.column_stack(
        (x_normalized.ravel(), y_normalized.ravel(), sdf.ravel())
    ).astype(dtype, copy=False)
    if not np.isfinite(trunk).all():
        raise ValueError("Geom-DeepONet trunk features contain NaN or infinity")
    return branch, trunk


@dataclass(frozen=True)
class StepVelocityScale:
    """Zero-preserving RMS scales for the two velocity components."""

    u: float
    v: float

    def __post_init__(self) -> None:
        scales = np.asarray([self.u, self.v], dtype=float)
        if not np.isfinite(scales).all() or np.any(scales <= 0.0):
            raise ValueError("velocity scales must be finite and positive")

    @property
    def array(self) -> np.ndarray:
        return np.asarray([self.u, self.v], dtype=float)

    def transform(self, velocity: np.ndarray) -> np.ndarray:
        values = np.asarray(velocity, dtype=float)
        if values.shape[-1] != 2:
            raise ValueError("velocity must have final dimension [U,V]")
        return values / self.array

    def inverse_transform(self, velocity: np.ndarray) -> np.ndarray:
        values = np.asarray(velocity, dtype=float)
        if values.shape[-1] != 2:
            raise ValueError("velocity must have final dimension [U,V]")
        return values * self.array


def fit_step_velocity_scale(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    selected_heights: Sequence[int] | np.ndarray,
) -> StepVelocityScale:
    """Fit target scales from explicitly supplied training geometries only."""

    sum_squares = np.zeros(2, dtype=float)
    count = 0
    for height in np.asarray(selected_heights, dtype=int):
        if int(height) not in cases:
            raise KeyError(f"H{height} is not present in the supplied training store")
        case = cases[int(height)]
        velocity = np.column_stack((case["u"], case["v"])).astype(float, copy=False)
        if not np.isfinite(velocity).all():
            raise ValueError(f"H{height} contains non-finite velocity targets")
        sum_squares += np.sum(velocity * velocity, axis=0)
        count += len(velocity)
    if count == 0:
        raise ValueError("selected_heights must contain at least one geometry")
    rms = np.sqrt(sum_squares / count)
    return StepVelocityScale(float(rms[0]), float(rms[1]))


@dataclass(frozen=True)
class StepGeomTrainingBatch:
    """Fixed-size case batches for case-wise Geom-DeepONet training."""

    parameters: np.ndarray
    trunk: np.ndarray
    targets: np.ndarray
    height_percent: np.ndarray
    point_indices: np.ndarray


def sample_step_geom_training_batch(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    selected_heights: Sequence[int] | np.ndarray,
    *,
    domain: StepDomain,
    velocity_scale: StepVelocityScale,
    points_per_case: int = 4096,
    seed: int = 690,
) -> StepGeomTrainingBatch:
    """Uniformly sample complete geometries without target-conditioned inputs.

    Sampling depends only on row count and ``seed``.  U and V are read after
    indices and model inputs are fixed, solely to construct supervised targets.
    This means altering target values cannot alter the branch, trunk, or sampled
    locations.
    """

    heights = np.asarray(selected_heights, dtype=int).reshape(-1)
    if len(heights) == 0 or len(np.unique(heights)) != len(heights):
        raise ValueError("selected_heights must be non-empty and unique")
    if int(points_per_case) < 1:
        raise ValueError("points_per_case must be positive")
    rng = np.random.default_rng(seed)
    parameters, trunks, targets, indices_by_case = [], [], [], []
    for height in heights:
        integer_height = int(height)
        if integer_height not in cases:
            raise KeyError(f"H{height} is not present in the supplied training store")
        case = cases[integer_height]
        x = np.asarray(case["x"], dtype=float).reshape(-1)
        y = np.asarray(case["y"], dtype=float).reshape(-1)
        u = np.asarray(case["u"], dtype=float).reshape(-1)
        v = np.asarray(case["v"], dtype=float).reshape(-1)
        if not (len(x) == len(y) == len(u) == len(v)):
            raise ValueError(f"H{height} has inconsistent point-array lengths")
        if points_per_case > len(x):
            raise ValueError(
                f"H{height} has {len(x)} rows, fewer than points_per_case={points_per_case}"
            )
        indices = np.sort(rng.choice(len(x), int(points_per_case), replace=False))
        branch, trunk = step_geom_deeponet_inputs(
            integer_height / 100.0,
            x[indices],
            y[indices],
            domain=domain,
        )
        parameters.append(branch[0])
        trunks.append(trunk)
        physical_targets = np.column_stack((u[indices], v[indices]))
        targets.append(velocity_scale.transform(physical_targets).astype(np.float32))
        indices_by_case.append(indices)
    return StepGeomTrainingBatch(
        parameters=np.asarray(parameters, dtype=np.float32),
        trunk=np.asarray(trunks, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
        height_percent=heights.copy(),
        point_indices=np.asarray(indices_by_case, dtype=int),
    )


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - exercised without ML extra
        raise ImportError(
            "Geom-DeepONet requires the optional ML dependencies: "
            "python -m pip install 'flowmllab[ml]'"
        ) from error
    return tf


_KERAS_COMPONENTS: tuple[type[Any], type[Any], type[Any]] | None = None


def _keras_components() -> tuple[type[Any], type[Any], type[Any]]:
    """Create serializable Keras layers lazily so base imports need no TensorFlow."""

    global _KERAS_COMPONENTS
    if _KERAS_COMPONENTS is not None:
        return _KERAS_COMPONENTS
    tf = _tensorflow()
    keras = tf.keras

    @keras.utils.register_keras_serializable(package="flowmllab")
    class SirenDense(keras.layers.Layer):
        def __init__(
            self,
            units: int,
            *,
            omega_0: float = 10.0,
            first: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.units = int(units)
            self.omega_0 = float(omega_0)
            self.first = bool(first)

        def build(self, input_shape: Any) -> None:
            input_width = int(input_shape[-1])
            bound = (
                1.0 / input_width
                if self.first
                else np.sqrt(6.0 / input_width) / self.omega_0
            )
            initializer = keras.initializers.RandomUniform(-bound, bound)
            self.kernel = self.add_weight(
                name="kernel",
                shape=(input_width, self.units),
                initializer=initializer,
                trainable=True,
            )
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=initializer,
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Any) -> Any:
            return tf.sin(self.omega_0 * (tf.linalg.matmul(inputs, self.kernel) + self.bias))

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "units": self.units,
                "omega_0": self.omega_0,
                "first": self.first,
            }

    @keras.utils.register_keras_serializable(package="flowmllab")
    class BroadcastMultiply(keras.layers.Layer):
        def call(self, inputs: Sequence[Any]) -> Any:
            branch, point_features = inputs
            return point_features * tf.expand_dims(branch, axis=1)

    @keras.utils.register_keras_serializable(package="flowmllab")
    class OperatorContraction(keras.layers.Layer):
        def __init__(self, rank: int, output_dim: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.rank = int(rank)
            self.output_dim = int(output_dim)

        def build(self, input_shape: Any) -> None:
            self.output_bias = self.add_weight(
                name="output_bias",
                shape=(self.output_dim,),
                initializer="zeros",
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs: Sequence[Any]) -> Any:
            branch, point_features = inputs
            shape = tf.shape(point_features)
            trunk = tf.reshape(
                point_features,
                (shape[0], shape[1], self.rank, self.output_dim),
            )
            return tf.einsum("br,bnro->bno", branch, trunk) + self.output_bias

        def get_config(self) -> dict[str, Any]:
            return {
                **super().get_config(),
                "rank": self.rank,
                "output_dim": self.output_dim,
            }

    _KERAS_COMPONENTS = SirenDense, BroadcastMultiply, OperatorContraction
    return _KERAS_COMPONENTS


def build_step_geom_deeponet(
    *,
    parameter_dim: int = 1,
    query_dim: int = 3,
    output_dim: int = 2,
    width: int = 48,
    omega_0: float = 10.0,
    activation: str = "swish",
    seed: int = 690,
) -> Any:
    """Build the SDF/SIREN Geom-DeepONet used for the step experiment.

    Input shapes are ``(batch, parameter_dim)`` and
    ``(batch, arbitrary_points, query_dim)``.  For the current micro-step,
    ``parameter_dim=1`` is ``h/H``, ``query_dim=3`` is ``x, y, SDF``, and the
    two outputs are U and V.  The architecture is created lazily so importing
    FlowMLLab does not require TensorFlow.
    """

    if min(parameter_dim, query_dim, output_dim) < 1:
        raise ValueError("input and output dimensions must be positive")
    if width < 4 or omega_0 <= 0.0:
        raise ValueError("width must be at least four and omega_0 must be positive")
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    keras = tf.keras
    SirenDense, BroadcastMultiply, OperatorContraction = _keras_components()

    parameters = keras.Input(shape=(parameter_dim,), name="geometry_parameters")
    query = keras.Input(shape=(None, query_dim), name="query_x_y_sdf")

    branch = parameters
    trunk = query
    for index, units in enumerate((50, 50, width), start=1):
        branch = keras.layers.Dense(
            units, activation=activation, name=f"branch_pre_{index}"
        )(branch)
        trunk = keras.layers.Dense(
            units, activation=activation, name=f"trunk_pre_{index}"
        )(trunk)

    mixed = BroadcastMultiply(name="intermediate_geometry_fusion")([branch, trunk])
    pooled = keras.layers.GlobalAveragePooling1D(name="geometry_point_pool")(mixed)

    branch_post = pooled
    for index, units in enumerate((2 * width, 2 * width, width), start=1):
        branch_post = keras.layers.Dense(
            units, activation=activation, name=f"branch_post_{index}"
        )(branch_post)

    trunk_post = mixed
    for index, units in enumerate((2 * width, 2 * width, width * output_dim), start=1):
        trunk_post = SirenDense(
            units, omega_0=omega_0, name=f"siren_trunk_{index}"
        )(trunk_post)

    outputs = OperatorContraction(
        rank=width,
        output_dim=output_dim,
        name="branch_trunk_contraction",
    )([branch_post, trunk_post])
    return keras.Model(
        inputs=(parameters, query),
        outputs=outputs,
        name="step_geom_deeponet",
    )


def step_zonal_loss(alpha: float | None = None) -> Any:
    """Return unweighted MSE or a separately normalized reverse-flow loss.

    Because :class:`StepVelocityScale` scales about zero, ``U < 0`` has the
    same meaning in scaled and physical units.  The target-derived region is
    used only inside the training loss and never enters either model input.
    """

    tf = _tensorflow()
    if alpha is None:
        return tf.keras.losses.MeanSquaredError(name="velocity_mse")
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    def loss(y_true: Any, y_prediction: Any) -> Any:
        squared = tf.reduce_sum(tf.square(y_prediction - y_true), axis=-1)
        vortex = y_true[..., 0] < 0.0
        main = ~vortex
        vortex_count = tf.reduce_sum(tf.cast(vortex, squared.dtype))
        main_count = tf.reduce_sum(tf.cast(main, squared.dtype))
        vortex_mean = tf.math.divide_no_nan(
            tf.reduce_sum(tf.where(vortex, squared, tf.zeros_like(squared))),
            vortex_count,
        )
        main_mean = tf.math.divide_no_nan(
            tf.reduce_sum(tf.where(main, squared, tf.zeros_like(squared))),
            main_count,
        )
        vortex_weight = value * tf.cast(vortex_count > 0.0, squared.dtype)
        main_weight = (1.0 - value) * tf.cast(main_count > 0.0, squared.dtype)
        return tf.math.divide_no_nan(
            vortex_weight * vortex_mean + main_weight * main_mean,
            vortex_weight + main_weight,
        )

    loss.__name__ = f"zonal_velocity_alpha_{value:.2f}"
    return loss


@dataclass
class StepGeomDeepONetFit:
    """Fitted model plus the preprocessing contract needed for inference."""

    model: Any
    domain: StepDomain
    velocity_scale: StepVelocityScale
    training_heights: tuple[int, ...]
    history: dict[str, list[float]]
    configuration: dict[str, Any]


def fit_step_geom_deeponet(
    cases: Mapping[int, Mapping[str, np.ndarray]],
    selected_heights: Sequence[int] | np.ndarray,
    *,
    domain: StepDomain | None = None,
    validation_heights: Sequence[int] | np.ndarray | None = None,
    points_per_case: int = 4096,
    alpha: float | None = 0.6,
    width: int = 48,
    omega_0: float = 10.0,
    learning_rate: float = 8.0e-4,
    epochs: int = 500,
    batch_size: int = 1,
    seed: int = 690,
    verbose: int = 0,
    callbacks: Sequence[Any] | None = None,
) -> StepGeomDeepONetFit:
    """Fit one leakage-free case-wise micro-step Geom-DeepONet.

    Pass only the learning archive.  ``selected_heights`` and optional
    ``validation_heights`` are looked up in that supplied mapping, so a sealed
    geometry cannot be accessed accidentally.  Hyperparameter selection and a
    final seven-case refit should happen before loading the separate test file.
    """

    heights = np.asarray(selected_heights, dtype=int).reshape(-1)
    if len(heights) == 0:
        raise ValueError("selected_heights must contain at least one geometry")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    first_height = int(heights[0])
    if first_height not in cases:
        raise KeyError(f"H{first_height} is not present in the supplied training store")
    if domain is None:
        first = cases[first_height]
        domain = infer_step_domain(first["x"], first["y"])
    scale = fit_step_velocity_scale(cases, heights)
    training = sample_step_geom_training_batch(
        cases,
        heights,
        domain=domain,
        velocity_scale=scale,
        points_per_case=points_per_case,
        seed=seed,
    )

    validation_data = None
    validation_tuple: tuple[int, ...] = ()
    if validation_heights is not None:
        validation_array = np.asarray(validation_heights, dtype=int).reshape(-1)
        validation_tuple = tuple(int(value) for value in validation_array)
        validation = sample_step_geom_training_batch(
            cases,
            validation_array,
            domain=domain,
            velocity_scale=scale,
            points_per_case=points_per_case,
            seed=seed + 1,
        )
        validation_data = (
            (validation.parameters, validation.trunk),
            validation.targets,
        )

    tf = _tensorflow()
    model = build_step_geom_deeponet(
        width=width,
        omega_0=omega_0,
        seed=seed,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
        loss=step_zonal_loss(alpha),
    )
    fitted_history = model.fit(
        (training.parameters, training.trunk),
        training.targets,
        validation_data=validation_data,
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=True,
        verbose=int(verbose),
        callbacks=list(callbacks or ()),
    )
    return StepGeomDeepONetFit(
        model=model,
        domain=domain,
        velocity_scale=scale,
        training_heights=tuple(int(value) for value in heights),
        history={
            key: [float(item) for item in values]
            for key, values in fitted_history.history.items()
        },
        configuration={
            "points_per_case": int(points_per_case),
            "validation_heights": validation_tuple,
            "alpha": None if alpha is None else float(alpha),
            "width": int(width),
            "omega_0": float(omega_0),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
        },
    )


def predict_step_geom_deeponet(
    fitted: StepGeomDeepONetFit,
    height_percent: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> np.ndarray:
    """Predict physical U,V using geometry and coordinates only."""

    branch, trunk = step_geom_deeponet_inputs(
        float(height_percent) / 100.0,
        x_m,
        y_m,
        domain=fitted.domain,
    )
    scaled = np.asarray(
        fitted.model((branch, trunk[None, ...]), training=False), dtype=float
    )[0]
    return fitted.velocity_scale.inverse_transform(scaled)


def evaluate_step_geom_deeponet(
    fitted: StepGeomDeepONetFit,
    cases: Mapping[int, Mapping[str, np.ndarray]],
    selected_heights: Sequence[int] | np.ndarray,
) -> list[dict[str, float | int]]:
    """Evaluate explicitly supplied cases without using targets as inputs."""

    from .mahdavi_deeponet import zonal_velocity_metrics

    rows: list[dict[str, float | int]] = []
    metric_alpha = 0.5 if fitted.configuration["alpha"] is None else float(
        fitted.configuration["alpha"]
    )
    for height in np.asarray(selected_heights, dtype=int):
        integer_height = int(height)
        if integer_height not in cases:
            raise KeyError(f"H{height} is not present in the supplied evaluation store")
        case = cases[integer_height]
        prediction = predict_step_geom_deeponet(
            fitted,
            integer_height,
            case["x"],
            case["y"],
        )
        metrics = zonal_velocity_metrics(
            case["u"],
            case["v"],
            prediction[:, 0],
            prediction[:, 1],
            alpha=metric_alpha,
        )
        rows.append({"height_percent": integer_height, **metrics})
    return rows


__all__ = [
    "GEOM_DEEPONET_DOI",
    "StepDomain",
    "StepGeomDeepONetFit",
    "StepGeomTrainingBatch",
    "StepVelocityScale",
    "build_step_geom_deeponet",
    "evaluate_step_geom_deeponet",
    "fit_step_geom_deeponet",
    "fit_step_velocity_scale",
    "infer_step_domain",
    "predict_step_geom_deeponet",
    "sample_step_geom_training_batch",
    "step_geom_deeponet_inputs",
    "step_signed_distance",
    "step_zonal_loss",
]
