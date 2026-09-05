"""Training-only registration and POD surrogates for the public nozzle sweep.

The polynomial branch is a non-neural baseline. The tanh branch is a fitted
POD neural surrogate. Neither is the article's six-output Fusion--DeepONet.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FIELDS = ("density", "u_ms", "v_ms", "temperature_k", "mach", "pressure_tecplot")


def compression_surface(x: np.ndarray, density: np.ndarray, throat: float) -> np.ndarray:
    """Find a positive interior density gradient, with subcell peak refinement.

    Four grid intervals after the throat and before the outlet are excluded.
    This separates the internal compression from the expansion and the outlet
    boundary gradient. The sensor does not edit, smooth, or replace labels.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(density, dtype=float)
    if x.ndim != 1 or rho.ndim != 2 or rho.shape[1] != x.size:
        raise ValueError("density must have shape (rows, x)")
    dx = np.diff(x)
    if x.size < 15 or not np.all(dx > 0) or not np.allclose(dx, dx[0], rtol=1e-4):
        raise ValueError("compression sensor requires a uniform increasing x grid")
    candidates = np.flatnonzero((x > throat + 4 * dx[0]) & (x < x[-1] - 4 * dx[0]))
    if candidates.size < 3 or not np.isfinite(rho).all():
        raise ValueError("no finite interior compression search interval")
    gradient = np.gradient(gaussian_filter1d(rho, 0.7, axis=-1), x, axis=-1)
    index = candidates[np.argmax(gradient[:, candidates], axis=1)]
    rows = np.arange(rho.shape[0])
    if np.any(gradient[rows, index] <= 0):
        raise ValueError("a positive internal compression is required in every row")
    denominator = gradient[rows, index - 1] - 2 * gradient[rows, index] + gradient[rows, index + 1]
    offset = np.divide(
        0.5 * (gradient[rows, index - 1] - gradient[rows, index + 1]),
        denominator, out=np.zeros_like(denominator), where=np.abs(denominator) > 1e-15,
    )
    return x[index] + np.clip(offset, -0.5, 0.5) * dx[0]


def warp_fields(
    fields: np.ndarray, x: np.ndarray, throat: float,
    source_surface: np.ndarray, target_surface: np.ndarray,
) -> np.ndarray:
    """Register rows while fixing the inlet, throat, and outlet coordinates.

    ``fields`` has shape (rows, x, channels). The piecewise-affine coordinate
    map is monotone; it translates the compression without translating the
    complete nozzle or extrapolating beyond a boundary. It is not conservative
    remapping, so conservation is measured separately.
    """
    values = np.asarray(fields, dtype=float)
    x = np.asarray(x, dtype=float)
    origin = np.asarray(source_surface, dtype=float)
    target = np.asarray(target_surface, dtype=float)
    if values.ndim != 3 or values.shape[1] != x.size:
        raise ValueError("fields must have shape (rows, x, channels)")
    if origin.shape != (values.shape[0],) or target.shape != origin.shape:
        raise ValueError("one source and target station per row are required")
    if not (np.all(origin > throat) and np.all(origin < x[-1])
            and np.all(target > throat) and np.all(target < x[-1])):
        raise ValueError("compression stations must be between throat and outlet")
    result = np.empty_like(values)
    for row in range(values.shape[0]):
        mapped_x = np.interp(x, [x[0], throat, target[row], x[-1]],
                             [x[0], throat, origin[row], x[-1]])
        for channel in range(values.shape[-1]):
            result[row, :, channel] = np.interp(mapped_x, x, values[row, :, channel])
    return result


def fit_transport_pod(
    pressure_kpa: np.ndarray, fields: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
    *, rank: int = 6, branch: str = "polynomial", width: int = 16, seed: int = 690,
) -> dict[str, np.ndarray]:
    """Fit only the supplied training cases, including all preprocessing.

    A Huber quadratic maps pressure to each compression station. Registered
    fields are RMS-scaled per physical channel, then a joint POD is fitted.
    The coefficient branch is either degree-two least squares or a one-hidden-
    layer tanh MLP. No target field or target-derived station is accepted by
    the prediction API.
    """
    pressure = np.asarray(pressure_kpa, dtype=float)
    values = np.asarray(fields, dtype=float)
    xgrid, ygrid = np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float)
    x = xgrid[0] if xgrid.ndim == 2 else xgrid
    if (pressure.ndim != 1 or len(pressure) < 5 or not np.all(np.diff(pressure) > 0)
            or values.ndim != 4 or values.shape[0] != len(pressure)
            or values.shape[-1] != len(FIELDS) or values.shape[1:3] != ygrid.shape
            or values.shape[2] != len(x) or not np.isfinite(values).all()):
        raise ValueError("need sorted training pressures and finite six-field grids")
    if xgrid.ndim == 2 and not np.allclose(xgrid, x[None, :], rtol=0, atol=1e-15):
        raise ValueError("the nozzle transport requires shared x rows")
    if not 1 <= rank <= len(pressure) - 1 or branch not in {"polynomial", "neural"}:
        raise ValueError("invalid POD rank or coefficient branch")
    throat = float(x[np.argmin(ygrid[-1] - ygrid[0])])
    surfaces = np.stack([compression_surface(x, field[:, :, 0], throat) for field in values])
    center = 0.5 * (pressure[0] + pressure[-1])
    half_range = 0.5 * (pressure[-1] - pressure[0])
    p = (pressure - center) / half_range
    features = np.column_stack((p, p * p))
    station_coefficients = []
    for row in range(values.shape[1]):
        locator = HuberRegressor(epsilon=1.35, alpha=0, max_iter=500).fit(
            features, surfaces[:, row] * 1e6,
        )
        station_coefficients.append(np.r_[locator.intercept_, locator.coef_] / 1e6)
    reference_surface = np.median(surfaces, axis=0)
    registered = np.stack([
        warp_fields(field, x, throat, surface, reference_surface)
        for field, surface in zip(values, surfaces, strict=True)
    ])
    channel_scale = np.sqrt(np.mean(registered**2, axis=(0, 1, 2)))
    if np.any(channel_scale <= 0):
        raise ValueError("each channel needs a nonzero training norm")
    matrix = (registered / channel_scale).reshape(len(pressure), -1)
    mean = matrix.mean(axis=0)
    _, _, basis = np.linalg.svd(matrix - mean, full_matrices=False)
    modes = basis[:rank]
    coefficients = (matrix - mean) @ modes.T
    fitted = {
        "schema_version": np.array(1), "branch": np.array(branch),
        "rank": np.array(rank), "width": np.array(width), "seed": np.array(seed),
        "training_pressures_kpa": pressure.copy(), "parameter_center": np.array(center),
        "parameter_half_range": np.array(half_range), "x_m": x.copy(),
        "y_m": ygrid.copy(), "throat_m": np.array(throat),
        "training_surfaces_m": surfaces, "station_coefficients": np.array(station_coefficients),
        "reference_surface_m": reference_surface, "field_shape": np.array(values.shape[1:]),
        "channel_scale": channel_scale, "mean": mean, "modes": modes,
        "field_names": np.array(FIELDS),
    }
    if branch == "polynomial":
        fitted["polynomial_coefficients"] = np.polynomial.polynomial.polyfit(p, coefficients, 2)
        fitted["fit_converged"] = np.array(True)
    else:
        scaler = StandardScaler().fit(coefficients)
        model = MLPRegressor(
            hidden_layer_sizes=(width,), activation="tanh", solver="lbfgs",
            alpha=0.01, max_iter=1500, random_state=seed, tol=1e-8,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(p[:, None], scaler.transform(coefficients))
        fitted.update({
            "neural_weight_0": model.coefs_[0], "neural_bias_0": model.intercepts_[0],
            "neural_weight_1": model.coefs_[1], "neural_bias_1": model.intercepts_[1],
            "coefficient_mean": scaler.mean_, "coefficient_scale": scaler.scale_,
            "fit_converged": np.array(not any(issubclass(w.category, ConvergenceWarning) for w in caught)),
            "training_iterations": np.array(model.n_iter_),
        })
    return fitted


def predict_transport_pod(fitted: dict[str, np.ndarray], pressure_kpa: np.ndarray) -> np.ndarray:
    """Predict six fields from pressure alone; reject extrapolation explicitly."""
    query = np.asarray(pressure_kpa, dtype=float).reshape(-1)
    train = fitted["training_pressures_kpa"]
    if not np.isfinite(query).all() or np.any(query < train[0]) or np.any(query > train[-1]):
        raise ValueError("query pressures must lie inside the training interval")
    p = (query - fitted["parameter_center"]) / fitted["parameter_half_range"]
    if str(fitted["branch"]) == "polynomial":
        coefficients = np.polynomial.polynomial.polyval(p, fitted["polynomial_coefficients"]).T
    else:
        hidden = np.tanh(p[:, None] @ fitted["neural_weight_0"] + fitted["neural_bias_0"])
        coefficients = (hidden @ fitted["neural_weight_1"] + fitted["neural_bias_1"])
        coefficients = coefficients * fitted["coefficient_scale"] + fitted["coefficient_mean"]
    fields = (fitted["mean"] + coefficients @ fitted["modes"]).reshape(
        (len(query), *fitted["field_shape"].astype(int)),
    ) * fitted["channel_scale"]
    stations = np.column_stack((np.ones(len(query)), p, p * p)) @ fitted["station_coefficients"].T
    return np.stack([
        warp_fields(field, fitted["x_m"], float(fitted["throat_m"]),
                    fitted["reference_surface_m"], surface)
        for field, surface in zip(fields, stations, strict=True)
    ])


def save_transport_model(fitted: dict[str, np.ndarray], path: str | Path) -> None:
    """Save numerical weights and provenance without Python pickle objects."""
    np.savez_compressed(path, **fitted)


def predict_with_symmetry(
    fitted: dict[str, np.ndarray], pressure_kpa: np.ndarray, *, symmetry_y_m: float,
) -> np.ndarray:
    """Apply the supplied horizontal symmetry boundary to predicted nodal V.

    This is a boundary-condition projection, not an improvement to the raw-data
    regression score or a repair of the source DSMC export. Interior predictions
    are unchanged. The caller must supply a known symmetry coordinate; a max-y
    row alone does not establish that a boundary is a symmetry plane.
    """
    y = fitted["y_m"]
    rows = np.flatnonzero(np.all(np.isclose(y, symmetry_y_m, rtol=0, atol=1e-11), axis=1))
    if len(rows) != 1 or rows[0] not in (0, y.shape[0]-1):
        raise ValueError("symmetry coordinate must identify one complete boundary row")
    prediction = predict_transport_pod(fitted, pressure_kpa)
    prediction[:, rows[0], :, FIELDS.index("v_ms")] = 0.0
    return prediction


def load_transport_model(path: str | Path) -> dict[str, np.ndarray]:
    """Load the numerical checkpoint with pickle disabled."""
    with np.load(path, allow_pickle=False) as archive:
        model = {name: np.asarray(archive[name]) for name in archive.files}
    if int(model.get("schema_version", -1)) != 1 or tuple(model["field_names"]) != FIELDS:
        raise ValueError("unsupported nozzle transport checkpoint")
    return model
