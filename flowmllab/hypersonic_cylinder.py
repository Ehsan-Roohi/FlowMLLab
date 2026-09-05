"""Leakage-controlled teaching tools for a rarefied hypersonic cylinder.

The committed data are a compact derivative of author-supplied DSMC fields
associated with Roohi et al., Physics of Fluids 38, 057108 (2026).  The default
CPU surrogate is a separable random-feature ridge ensemble.  It exposes the
branch/trunk inductive bias and deep-ensemble workflow without claiming to
reproduce the full Fusion-DeepONet or its published accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DEFAULT_TRAIN_MACH = np.asarray([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=float)
DEFAULT_VALIDATION_MACH = np.asarray([8.25, 8.75, 9.25, 9.75], dtype=float)
DEFAULT_INTERPOLATION_MACH = np.asarray([5.5, 6.5, 7.5, 8.5, 9.5], dtype=float)
DEFAULT_EXTRAPOLATION_MACH = np.asarray([15.0], dtype=float)
TARGET_NAMES = ("local_mach", "temperature_ratio", "pressure_ratio")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_mach_set(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(values), dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("Mach selections must be nonempty finite 1-D sequences")
    return result


@dataclass(frozen=True)
class CylinderTeachingData:
    mach_inf: np.ndarray
    x: np.ndarray
    y: np.ndarray
    targets: np.ndarray
    case_id: np.ndarray
    source_row: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.mach_inf)
        one_dimensional = (self.x, self.y, self.case_id, self.source_row)
        if any(np.asarray(array).shape != (count,) for array in one_dimensional):
            raise ValueError("All point-wise arrays must have equal 1-D length")
        if np.asarray(self.targets).shape != (count, 3):
            raise ValueError("targets must have shape (n_points, 3)")
        if count == 0 or not all(
            np.isfinite(array).all()
            for array in (self.mach_inf, self.x, self.y, self.targets)
        ):
            raise ValueError("Cylinder data must be nonempty and finite")


def load_cylinder_teaching_data(root: Path) -> CylinderTeachingData:
    """Load and hash-check the committed Week-7.1 DSMC derivative."""
    data_dir = Path(root) / "data" / "hypersonic_cylinder"
    data_path = data_dir / "cylinder_teaching_subset.npz"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    actual_hash = _sha256(data_path)
    if actual_hash != manifest["artifact_sha256"]:
        raise ValueError(
            "Hypersonic-cylinder data hash mismatch: "
            f"expected {manifest['artifact_sha256']}, found {actual_hash}"
        )
    with np.load(data_path, allow_pickle=False) as archive:
        names = tuple(str(item) for item in archive["target_names"])
        if names != TARGET_NAMES:
            raise ValueError(f"Unexpected target order: {names}")
        return CylinderTeachingData(
            mach_inf=archive["mach_inf"].astype(float),
            x=archive["x"].astype(float),
            y=archive["y"].astype(float),
            targets=archive["targets"].astype(float),
            case_id=archive["case_id"].astype(int),
            source_row=archive["source_row"].astype(int),
        )


def mach_mask(mach_inf: np.ndarray, selected: Iterable[float]) -> np.ndarray:
    """Return a tolerant membership mask for whole-case Mach selections."""
    mach = np.asarray(mach_inf, dtype=float)
    choices = _as_mach_set(selected)
    return np.any(np.isclose(mach[:, None], choices[None, :], atol=1.0e-8), axis=1)


def casewise_split_masks(
    mach_inf: np.ndarray,
    *,
    train: Iterable[float] = DEFAULT_TRAIN_MACH,
    validation: Iterable[float] = DEFAULT_VALIDATION_MACH,
    interpolation: Iterable[float] = DEFAULT_INTERPOLATION_MACH,
    extrapolation: Iterable[float] = DEFAULT_EXTRAPOLATION_MACH,
) -> dict[str, np.ndarray]:
    """Create disjoint masks and reject missing or overlapping flow cases."""
    selections = {
        "train": _as_mach_set(train),
        "validation": _as_mach_set(validation),
        "interpolation": _as_mach_set(interpolation),
        "extrapolation": _as_mach_set(extrapolation),
    }
    labels = list(selections)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            if np.any(
                np.isclose(
                    selections[left][:, None], selections[right][None, :], atol=1.0e-8
                )
            ):
                raise ValueError(f"Mach cases overlap between {left} and {right}")
    masks = {name: mach_mask(mach_inf, values) for name, values in selections.items()}
    missing = [name for name, mask in masks.items() if not np.any(mask)]
    if missing:
        raise ValueError(f"No data found for split(s): {', '.join(missing)}")
    return masks


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return one relative-L2 value per output column."""
    truth = np.asarray(reference, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("reference and prediction must have matching 2-D shapes")
    denominator = np.linalg.norm(truth, axis=0)
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError("relative_l2 received a zero-norm target")
    return np.linalg.norm(estimate - truth, axis=0) / denominator


def weighted_standardized_mse(
    reference_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    *,
    weights: Iterable[float] = (1.0, 1.0, 5.0),
) -> float:
    """Evaluate the reviewed research weighting in standardized target space."""
    truth = np.asarray(reference_scaled, dtype=float)
    estimate = np.asarray(prediction_scaled, dtype=float)
    weight = np.asarray(tuple(weights), dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("paired standardized targets must have shape (n, 3)")
    if weight.shape != (3,) or np.any(weight <= 0):
        raise ValueError("weights must contain three positive values")
    return float(np.mean((truth - estimate) ** 2 * weight[None, :]))


def case_interpolation_baseline(
    data: CylinderTeachingData,
    train_mask: np.ndarray,
    query_mask: np.ndarray,
) -> np.ndarray:
    """Interpolate structured fields in Mach at matching source-grid rows.

    This is the mandatory strong baseline for the teaching lab.  Within the
    training Mach range it uses the two bracketing cases; outside that range it
    uses the nearest two cases for a declared linear extrapolation.
    """
    train = np.asarray(train_mask, dtype=bool)
    query = np.asarray(query_mask, dtype=bool)
    expected_shape = (len(data.mach_inf),)
    if train.shape != expected_shape or query.shape != expected_shape:
        raise ValueError("train_mask and query_mask must match the dataset length")
    if not np.any(train) or not np.any(query):
        raise ValueError("train and query masks must both be nonempty")
    train_mach = np.unique(data.mach_inf[train])
    if len(train_mach) < 2:
        raise ValueError("at least two training Mach cases are required")

    lookup = {
        (float(mach), int(row)): target
        for mach, row, target in zip(
            data.mach_inf[train], data.source_row[train], data.targets[train], strict=True
        )
    }
    query_indices = np.flatnonzero(query)
    prediction = np.empty((len(query_indices), 3), dtype=float)
    for output_index, point_index in enumerate(query_indices):
        mach = float(data.mach_inf[point_index])
        insertion = int(np.searchsorted(train_mach, mach))
        if insertion == 0:
            lower, upper = train_mach[:2]
        elif insertion == len(train_mach):
            lower, upper = train_mach[-2:]
        else:
            lower, upper = train_mach[insertion - 1 : insertion + 1]
        row = int(data.source_row[point_index])
        try:
            lower_target = lookup[(float(lower), row)]
            upper_target = lookup[(float(upper), row)]
        except KeyError as exc:
            raise ValueError(
                "Structured interpolation requires matching retained source rows"
            ) from exc
        fraction = (mach - lower) / (upper - lower)
        prediction[output_index] = lower_target + fraction * (
            upper_target - lower_target
        )
    return prediction


@dataclass
class SeparableRidgeMember:
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    branch_weight: np.ndarray
    branch_bias: np.ndarray
    trunk_weight: np.ndarray
    trunk_bias: np.ndarray
    ridge: Ridge

    def features(self, mach_inf: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        inputs = np.column_stack((mach_inf, x, y))
        scaled = self.input_scaler.transform(inputs)
        branch = np.tanh(scaled[:, :1] @ self.branch_weight + self.branch_bias)
        trunk = np.tanh(scaled[:, 1:] @ self.trunk_weight + self.trunk_bias)
        return branch * trunk

    def predict(self, mach_inf: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        scaled = self.ridge.predict(self.features(mach_inf, x, y))
        return self.target_scaler.inverse_transform(scaled)


def fit_separable_ridge_ensemble(
    data: CylinderTeachingData,
    train_mask: np.ndarray,
    *,
    members: int = 5,
    latent_dim: int = 64,
    alpha: float = 1.0e-3,
    seed: int = 760,
) -> list[SeparableRidgeMember]:
    """Fit a fast DeepONet-shaped teaching analog on training cases only."""
    selected = np.asarray(train_mask, dtype=bool)
    if selected.shape != (len(data.mach_inf),) or not np.any(selected):
        raise ValueError("train_mask must select at least one point")
    if members < 1 or latent_dim < 4 or alpha <= 0:
        raise ValueError("members >= 1, latent_dim >= 4, and alpha > 0 are required")

    inputs = np.column_stack((data.mach_inf, data.x, data.y))
    input_scaler = StandardScaler().fit(inputs[selected])
    target_scaler = StandardScaler().fit(data.targets[selected])
    scaled_inputs = input_scaler.transform(inputs[selected])
    scaled_targets = target_scaler.transform(data.targets[selected])

    ensemble: list[SeparableRidgeMember] = []
    for member_index in range(members):
        rng = np.random.default_rng(seed + member_index)
        branch_weight = rng.normal(scale=1.0, size=(1, latent_dim))
        branch_bias = rng.uniform(-np.pi, np.pi, size=latent_dim)
        trunk_weight = rng.normal(scale=1.0, size=(2, latent_dim))
        trunk_bias = rng.uniform(-np.pi, np.pi, size=latent_dim)
        branch = np.tanh(
            scaled_inputs[:, :1] @ branch_weight + branch_bias
        )
        trunk = np.tanh(scaled_inputs[:, 1:] @ trunk_weight + trunk_bias)
        ridge = Ridge(alpha=alpha).fit(branch * trunk, scaled_targets)
        ensemble.append(
            SeparableRidgeMember(
                input_scaler=input_scaler,
                target_scaler=target_scaler,
                branch_weight=branch_weight,
                branch_bias=branch_bias,
                trunk_weight=trunk_weight,
                trunk_bias=trunk_bias,
                ridge=ridge,
            )
        )
    return ensemble


def ensemble_predict(
    ensemble: Iterable[SeparableRidgeMember],
    mach_inf: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and sample standard deviation."""
    members = tuple(ensemble)
    if not members:
        raise ValueError("ensemble must contain at least one fitted member")
    predictions = np.stack(
        [member.predict(mach_inf, x, y) for member in members], axis=0
    )
    ddof = 1 if len(members) > 1 else 0
    return np.mean(predictions, axis=0), np.std(predictions, axis=0, ddof=ddof)


def build_fusion_deeponet(
    *,
    latent_dim: int = 256,
    hidden_layers: int = 4,
    dropout_rate: float = 0.2,
):
    """Build the reviewed Fusion-DeepONet topology when TensorFlow is installed.

    This constructs the architecture only.  The full paper protocol additionally
    requires the original 50,000-point case sampling, five independent fits, and
    the frozen validation/test design; the classroom notebook does not claim that
    reproduction.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "TensorFlow is optional. Install FlowMLLab with the 'ml' extra to "
            "build the full Fusion-DeepONet topology."
        ) from exc
    if latent_dim < 4 or hidden_layers < 1 or not 0.0 <= dropout_rate < 1.0:
        raise ValueError("invalid Fusion-DeepONet architecture settings")

    branch_input = keras.layers.Input(shape=(1,), name="mach_inf")
    trunk_input = keras.layers.Input(shape=(2,), name="coordinates")
    branch_layers = []
    branch = branch_input
    for index in range(hidden_layers):
        branch = keras.layers.Dense(
            latent_dim, activation="tanh", name=f"branch_dense_{index + 1}"
        )(branch)
        branch = keras.layers.Dropout(
            dropout_rate, name=f"branch_dropout_{index + 1}"
        )(branch)
        branch_layers.append(branch)

    trunk = trunk_input
    for index, branch_state in enumerate(branch_layers):
        trunk = keras.layers.Dense(
            latent_dim, activation="tanh", name=f"trunk_dense_{index + 1}"
        )(trunk)
        trunk = keras.layers.Dropout(
            dropout_rate, name=f"trunk_dropout_{index + 1}"
        )(trunk)
        trunk = keras.layers.Add(name=f"fusion_{index + 1}")(
            [trunk, branch_state]
        )

    dot = keras.layers.Lambda(
        lambda values: tf.reduce_sum(values[0] * values[1], axis=-1, keepdims=True),
        name="branch_trunk_dot",
    )([branch_layers[-1], trunk])
    output = keras.layers.Dense(3, name="ma_temperature_pressure")(dot)
    return keras.Model([branch_input, trunk_input], output, name="fusion_deeponet")


def validate_hypersonic_cylinder_evidence(root: Path) -> dict[str, object]:
    """Validate data provenance, cases, finite values, and frozen split coverage."""
    root = Path(root)
    data = load_cylinder_teaching_data(root)
    manifest_path = root / "data" / "hypersonic_cylinder" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    masks = casewise_split_masks(data.mach_inf)
    unique_mach = np.unique(data.mach_inf)
    expected = np.asarray([case["mach_inf"] for case in manifest["cases"]], dtype=float)
    if not np.allclose(unique_mach, expected):
        raise ValueError("Manifest cases do not match the NPZ Mach cases")
    if manifest["total_points"] != len(data.mach_inf):
        raise ValueError("Manifest point count does not match the NPZ")
    if any(np.any(masks[left] & masks[right]) for left in masks for right in masks if left < right):
        raise ValueError("Frozen split masks overlap")
    return {
        "status": "pass",
        "cases": int(len(unique_mach)),
        "points": int(len(data.mach_inf)),
        "targets": list(TARGET_NAMES),
        "split_cases": {
            name: sorted(np.unique(data.mach_inf[mask]).tolist())
            for name, mask in masks.items()
        },
        "paper_doi": manifest["paper_doi"],
        "source_archive_sha256": manifest["source_archive_sha256"],
    }
