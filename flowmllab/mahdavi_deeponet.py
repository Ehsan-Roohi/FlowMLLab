"""Teaching utilities for the Roohi--Mahdavi rarefied-flow case studies.

The micro-nozzle helpers operate on a compact, attributed derivative of the
authors' public DSMC snapshots.  The micro-step helpers use two compact,
file-separated teaching derivatives of the authors' DSMC height fields.  The
learning and sealed-test archives preserve the pinned source identity while
preventing accidental pre-gate access to the final cases.
"""

from __future__ import annotations

import hashlib
import json
import csv
import os
import re
import warnings
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from scipy.ndimage import gaussian_filter1d


NOZZLE_PRESSURES_KPA = np.array(
    [15, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33],
    dtype=float,
)
NOZZLE_HELD_OUT_KPA = np.array([16, 25, 30], dtype=float)

STEP_SOURCE_REPOSITORY = "https://github.com/Ehsan-Roohi/roohi-step-dnn-mahdavi"
STEP_SOURCE_COMMIT = "c3f211376b42b8dc30daad380eaef5e0ab800b5c"
STEP_HEIGHT_PERCENT = np.array([16, 21, 25, 33, 44, 50, 58, 67, 75], dtype=int)
STEP_HEIGHT_HELD_OUT_PERCENT = np.array([44, 67], dtype=int)
STEP_HEIGHT_VALIDATION_PERCENT = np.array([33, 58], dtype=int)
STEP_HEIGHT_DEVELOPMENT_PERCENT = np.array([16, 21, 25, 50, 75], dtype=int)
STEP_HEIGHT_FILE_SHA256 = {
    "H16_smoothed.dat": "3be4aa40a2308007437a69f8cf7498afec8808db593d48f07c1aa44aa9538a84",
    "H21_smoothed.dat": "5a6da8c4f2e7f615407265a4ca4bc342904806e3b41510daacd7afa0e164fc6a",
    "H25_smoothed.dat": "b040556b5d463f4df86a16e542fced4f5200a94f7d5d3e7be49600c356d0614c",
    "H33_smoothed.dat": "bc642868fae5422978110dc23ffef417bc185fc96dd26c2ce8c1746908c1445e",
    "H44_smoothed.dat": "eeb1002e4697466609cac8485a041e8c591b4374ad895230cef560f85dadea74",
    "H50_smoothed.dat": "d52f3564ec0a0055387848d274b0c8e90bfc8d9f4b9b60c055bbec69b25d18bd",
    "H58_smoothed.dat": "7505219574928a7891ba65f9789553ddd1df3b80c5b3260fcf9c200a203b8059",
    "H67_smoothed.dat": "79166f5a00982165f1c39daf174a94a9745a42cbced05b554d20d111e25a8b84",
    "H75_smoothed.dat": "7b1885f595de55703046652cea79f56f6f57826e851700d61b33c261102cd147",
}
STEP_HEIGHT_EXPECTED_ROWS = {
    16: 22980,
    21: 22680,
    25: 22380,
    33: 21780,
    44: 21180,
    50: 20580,
    58: 19980,
    67: 19380,
    75: 18780,
}
STEP_HEIGHT_ARCHIVE_SHA256 = {
    "step_height_learning_7cases.npz": "410907d46a040d53cbbd19fd8d44eeb7b41c05150f953fd4d9c6bb479da3479d",
    "step_height_test_2cases.npz": "f833ea965511eb690bfef094afd8f9fde660eafb24fa889ce3ec1582551b382c",
}
STEP_TEACHING_RESULT_SHA256 = {
    "step_teaching_selection.csv": "10f48f88c7e848ec6dbbc808dee630984b80ca18f65c8e0bf4ec64b0d1364d9f",
    "step_teaching_test_metrics.csv": "638bb802ff15f0535607537491a07c8fd0221f75f101624021255f79d6c43f69",
    "step_teaching_protocol.json": "f7e933aa86373662d7ce27baf93a0e73f55b52b277bf5b3f8e0d993c5ef8b833",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_smoothed_directory(source: str | Path) -> Path:
    """Resolve a source checkout or a direct smoothed-data directory."""
    candidate = Path(source).expanduser().resolve()
    if (candidate / "H16_smoothed.dat").is_file():
        return candidate
    nested = candidate / "data" / "height" / "smoothed"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(
        f"no step-height smoothed Tecplot directory found under {candidate}"
    )


def discover_step_source(root: str | Path, explicit: str | Path | None = None) -> Path | None:
    """Find an optional checkout of the pinned public step-data repository."""
    root_path = Path(root).resolve()
    configured = os.environ.get("FLOWMLLAB_STEP_SOURCE")
    candidates = [
        explicit,
        configured,
        root_path / ".external" / "roohi-step-dnn-mahdavi",
        root_path.parent / "roohi-step-dnn-mahdavi",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        try:
            _step_smoothed_directory(path)
        except FileNotFoundError:
            continue
        return path
    return None


def download_step_height_data(target: str | Path) -> Path:
    """Download and checksum the nine published smoothed height fields.

    The upstream repository has no reuse license at the pinned commit.  This
    helper therefore creates only a local cache for inspection/reproduction;
    FlowMLLab does not redistribute the source fields.
    """
    output = Path(target).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = (
        "https://raw.githubusercontent.com/Ehsan-Roohi/"
        f"roohi-step-dnn-mahdavi/{STEP_SOURCE_COMMIT}/data/height/smoothed"
    )
    for name, expected in STEP_HEIGHT_FILE_SHA256.items():
        destination = output / name
        if destination.is_file() and _sha256(destination) == expected:
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = Request(f"{base}/{name}", headers={"User-Agent": "FlowMLLab/1.2"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
        observed = _sha256(temporary)
        if observed != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for downloaded {name}")
        temporary.replace(destination)
    return output


def read_step_velocity(path: str | Path) -> dict[str, np.ndarray]:
    """Read X, Y, U, and V from an ASCII Tecplot step-flow file."""
    source = Path(path)
    variables: list[str] = []
    rows: list[list[float]] = []
    collecting_variables = False
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("VARIABLES"):
            variables.extend(re.findall(r'"([^"]+)"', stripped))
            collecting_variables = True
            continue
        if collecting_variables and stripped.startswith('"'):
            variables.extend(re.findall(r'"([^"]+)"', stripped))
            continue
        collecting_variables = False
        if not variables or stripped.upper().startswith(("TITLE", "ZONE", "AUXDATA")):
            continue
        try:
            values = [float(token.replace("D", "E")) for token in stripped.split()]
        except ValueError:
            continue
        if len(values) >= len(variables):
            rows.append(values[: len(variables)])
    names = [name.upper().strip() for name in variables]
    required = ("X", "Y", "U", "V")
    if not rows or any(name not in names for name in required):
        raise ValueError(f"{source}: missing Tecplot X/Y/U/V point data")
    data = np.asarray(rows, dtype=float)
    result = {name.lower(): data[:, names.index(name)] for name in required}
    if not all(np.isfinite(result[name]).all() for name in ("x", "y", "u", "v")):
        raise ValueError(f"{source}: X/Y/U/V contains NaN or infinity")
    return result


def load_step_height_cases(
    source: str | Path,
    *,
    heights: list[int] | np.ndarray | None = None,
    verify_hashes: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    """Load selected real public DSMC height cases from an upstream checkout/cache."""
    directory = _step_smoothed_directory(source)
    cases: dict[int, dict[str, np.ndarray]] = {}
    selected = STEP_HEIGHT_PERCENT if heights is None else np.asarray(heights, dtype=int)
    unknown = np.setdiff1d(selected, STEP_HEIGHT_PERCENT)
    if len(unknown):
        raise ValueError(f"unknown step-height cases: {unknown.tolist()}")
    for height in selected:
        name = f"H{height}_smoothed.dat"
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes and _sha256(path) != STEP_HEIGHT_FILE_SHA256[name]:
            raise ValueError(f"step source hash mismatch: {name}")
        case = read_step_velocity(path)
        case["height_ratio"] = np.asarray(float(height) / 100.0)
        cases[int(height)] = case
    return cases


def load_step_height_archive(
    root: str | Path,
    *,
    split: str,
) -> dict[int, dict[str, np.ndarray]]:
    """Load one compact micro-step split without opening the other split file."""
    split_contract = {
        "learning": (
            "step_height_learning_7cases.npz",
            np.concatenate((STEP_HEIGHT_DEVELOPMENT_PERCENT, STEP_HEIGHT_VALIDATION_PERCENT)),
        ),
        "test": ("step_height_test_2cases.npz", STEP_HEIGHT_HELD_OUT_PERCENT),
    }
    if split not in split_contract:
        raise ValueError("split must be 'learning' or 'test'")
    filename, expected_heights = split_contract[split]
    path = Path(root).resolve() / "results" / "mahdavi_deeponet" / filename
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "height_percent", "height_ratio", "case_offset", "x_m", "y_m",
            "u", "v", "fixed_knudsen", "step_x_m", "source_commit",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"{filename}: compact archive schema is incomplete")
        heights = np.asarray(archive["height_percent"], dtype=int)
        ratios = np.asarray(archive["height_ratio"], dtype=float)
        x_m = np.asarray(archive["x_m"], dtype=float)
        y_m = np.asarray(archive["y_m"], dtype=float)
        u = np.asarray(archive["u"], dtype=float)
        v = np.asarray(archive["v"], dtype=float)
        offsets = np.asarray(archive["case_offset"], dtype=int)
        source_commit = str(np.asarray(archive["source_commit"]).item())
        fixed_knudsen = float(np.asarray(archive["fixed_knudsen"]).item())
    if not np.array_equal(heights, expected_heights):
        raise ValueError(f"{filename}: unexpected or reordered height cases")
    if not np.allclose(ratios, heights / 100.0, rtol=0.0, atol=1.0e-15):
        raise ValueError(f"{filename}: height ratios do not match case labels")
    if source_commit != STEP_SOURCE_COMMIT or fixed_knudsen != 0.01:
        raise ValueError(f"{filename}: source identity or Knudsen number mismatch")
    if offsets.shape != (len(heights) + 1,) or offsets[0] != 0:
        raise ValueError(f"{filename}: invalid ragged case offsets")
    if offsets[-1] != len(x_m) or any(len(array) != len(x_m) for array in (y_m, u, v)):
        raise ValueError(f"{filename}: inconsistent compact point arrays")
    if not all(np.isfinite(array).all() for array in (x_m, y_m, u, v)):
        raise ValueError(f"{filename}: point arrays contain NaN or infinity")

    cases: dict[int, dict[str, np.ndarray]] = {}
    for index, height in enumerate(heights):
        section = slice(offsets[index], offsets[index + 1])
        case_x, case_y = x_m[section], y_m[section]
        if len(np.unique(case_x)) != 200 or len(np.unique(case_y)) != 120:
            raise ValueError(f"{filename}: H{height} does not map to a 200 by 120 parent grid")
        cases[int(height)] = {
            "x": case_x,
            "y": case_y,
            "u": u[section],
            "v": v[section],
            "height_ratio": np.asarray(ratios[index]),
        }
    return cases


def validate_step_height_archives(root: str | Path) -> dict[str, Any]:
    """Validate file identity, split isolation, topology, and row counts."""
    root_path = Path(root).resolve()
    result_dir = root_path / "results" / "mahdavi_deeponet"
    observed_hashes = {}
    all_cases: dict[int, dict[str, np.ndarray]] = {}
    for split, filename in (
        ("learning", "step_height_learning_7cases.npz"),
        ("test", "step_height_test_2cases.npz"),
    ):
        path = result_dir / filename
        observed_hashes[filename] = _sha256(path)
        if observed_hashes[filename] != STEP_HEIGHT_ARCHIVE_SHA256[filename]:
            raise ValueError(f"compact micro-step archive SHA-256 mismatch: {filename}")
        cases = load_step_height_archive(root_path, split=split)
        overlap = set(all_cases).intersection(cases)
        if overlap:
            raise ValueError(f"micro-step learning/test archives overlap: {sorted(overlap)}")
        all_cases.update(cases)
    if set(all_cases) != set(STEP_HEIGHT_PERCENT.tolist()):
        raise ValueError("compact micro-step archives do not cover exactly nine cases")
    row_counts = {str(height): len(case["u"]) for height, case in all_cases.items()}
    if row_counts != {str(key): value for key, value in STEP_HEIGHT_EXPECTED_ROWS.items()}:
        raise ValueError("compact micro-step archive row counts do not match source")
    return {
        "status": "pass",
        "source_commit": STEP_SOURCE_COMMIT,
        "archive_sha256": observed_hashes,
        "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
        "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
        "held_out_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
        "file_level_test_isolation": True,
        "row_counts": row_counts,
    }


def validate_step_archives_against_source(
    root: str | Path, source: str | Path
) -> dict[str, Any]:
    """Prove that compact point rows equal the pinned source up to float32 U,V storage."""
    source_cases = load_step_height_cases(source, verify_hashes=True)
    compact_cases = load_step_height_archive(root, split="learning")
    compact_cases.update(load_step_height_archive(root, split="test"))
    maximum_quantization = {"u": 0.0, "v": 0.0}
    for height in STEP_HEIGHT_PERCENT:
        original = source_cases[int(height)]
        compact = compact_cases[int(height)]
        if not np.array_equal(original["x"], compact["x"]) or not np.array_equal(
            original["y"], compact["y"]
        ):
            raise ValueError(f"H{height}: compact coordinates differ from pinned source rows")
        for field in ("u", "v"):
            expected = original[field].astype(np.float32).astype(float)
            if not np.array_equal(expected, compact[field]):
                raise ValueError(f"H{height}: compact {field.upper()} differs from float32 source")
            maximum_quantization[field] = max(
                maximum_quantization[field],
                float(np.max(np.abs(original[field] - compact[field]))),
            )
    return {
        "status": "pass",
        "comparison": "x,y exact; U,V exactly equal after documented float32 conversion",
        "max_absolute_float32_quantization": maximum_quantization,
    }


def validate_step_height_dataset(source: str | Path) -> dict[str, Any]:
    """Validate identity, shape, topology, split, and finite values of the real data."""
    cases = load_step_height_cases(source, verify_hashes=True)
    row_counts: dict[str, int] = {}
    vortex_fraction: dict[str, float] = {}
    coordinate_bounds = None
    for height, case in cases.items():
        rows = len(case["u"])
        if rows != STEP_HEIGHT_EXPECTED_ROWS[height]:
            raise ValueError(f"H{height}: expected {STEP_HEIGHT_EXPECTED_ROWS[height]} rows, got {rows}")
        unique_x, unique_y = np.unique(case["x"]), np.unique(case["y"])
        if len(unique_x) != 200 or len(unique_y) != 120:
            raise ValueError(f"H{height}: expected a 200 by 120 parent grid")
        bounds = np.array([unique_x[0], unique_x[-1], unique_y[0], unique_y[-1]])
        if coordinate_bounds is None:
            coordinate_bounds = bounds
        elif not np.allclose(bounds, coordinate_bounds, rtol=0.0, atol=1.0e-18):
            raise ValueError(f"H{height}: coordinate bounds do not match other cases")
        row_counts[str(height)] = rows
        vortex_fraction[str(height)] = float(np.mean(case["u"] < 0.0))
    return {
        "status": "pass",
        "source_commit": STEP_SOURCE_COMMIT,
        "height_percent": STEP_HEIGHT_PERCENT.tolist(),
        "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
        "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
        "held_out_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
        "row_counts": row_counts,
        "vortex_fraction": vortex_fraction,
        "coordinate_bounds_m": coordinate_bounds.tolist(),
    }


def step_coordinate_features(
    height_ratio: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    bounds_m: tuple[float, float, float, float],
    step_x_m: float = 25.0e-9,
) -> np.ndarray:
    """Create leakage-free features from geometry and coordinates only.

    No velocity, pressure, test-field patch, or target-derived mask enters this
    feature map.  The wall-relative coordinate is computed from the known step
    geometry rather than inferred from the held-out flow solution.
    """
    h = float(height_ratio)
    if not 0.0 < h < 1.0:
        raise ValueError("height_ratio must be a fraction strictly between zero and one")
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    x_min, x_max, y_min, y_max = map(float, bounds_m)
    xn = (x - x_min) / (x_max - x_min)
    yn = (y - y_min) / (y_max - y_min)
    hh = np.full(x.shape, h, dtype=float)
    step_relative_x = (x - float(step_x_m)) / (x_max - x_min)
    lower_wall = np.where(x <= step_x_m, h, 0.0)
    wall_eta = (yn - lower_wall) / np.maximum(1.0 - lower_wall, 1.0e-12)
    return np.column_stack(
        (hh.ravel(), xn.ravel(), yn.ravel(), step_relative_x.ravel(), wall_eta.ravel(),
         (hh * xn).ravel(), (hh * yn).ravel(), (xn * yn).ravel())
    )


def fit_step_coordinate_surrogate(
    cases: dict[int, dict[str, np.ndarray]],
    selected_heights: list[int] | np.ndarray,
    alpha: float | None,
    *,
    bounds_m: tuple[float, float, float, float],
    seed: int = 690,
    sample_size: int = 60_000,
    max_iter: int = 90,
    hidden_layer_sizes: tuple[int, ...] = (48, 48),
) -> dict[str, Any]:
    """Fit the documented coordinate MLP using only explicitly supplied cases."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    if alpha is not None and not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    features, targets, vortex = [], [], []
    for height in np.asarray(selected_heights, dtype=int):
        if int(height) not in cases:
            raise KeyError(f"H{height} is not present in the supplied training store")
        case = cases[int(height)]
        features.append(
            step_coordinate_features(
                float(height) / 100.0, case["x"], case["y"], bounds_m=bounds_m
            )
        )
        targets.append(np.column_stack((case["u"], case["v"])))
        vortex.append(np.asarray(case["u"]) < 0.0)
    features_array = np.vstack(features)
    targets_array = np.vstack(targets)
    vortex_array = np.concatenate(vortex)

    x_scaler = StandardScaler().fit(features_array)
    y_scaler = StandardScaler().fit(targets_array)
    x_scaled = x_scaler.transform(features_array)
    y_scaled = y_scaler.transform(targets_array)
    rng = np.random.default_rng(seed)
    if alpha is None:
        count = min(int(sample_size), len(features_array))
        indices = rng.choice(len(features_array), count, replace=False)
    else:
        vortex_count = int(round(int(sample_size) * float(alpha)))
        main_count = int(sample_size) - vortex_count
        vortex_indices = np.flatnonzero(vortex_array)
        main_indices = np.flatnonzero(~vortex_array)
        indices = np.concatenate(
            (
                rng.choice(vortex_indices, vortex_count, replace=vortex_count > len(vortex_indices)),
                rng.choice(main_indices, main_count, replace=main_count > len(main_indices)),
            )
        )
        rng.shuffle(indices)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="tanh",
        solver="adam",
        alpha=1.0e-5,
        batch_size=1024,
        learning_rate_init=8.0e-4,
        max_iter=int(max_iter),
        tol=1.0e-5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_scaled[indices], y_scaled[indices])
    return {
        "model": model,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "alpha": alpha,
        "bounds_m": tuple(map(float, bounds_m)),
        "training_heights": np.asarray(selected_heights, dtype=int).tolist(),
        "seed": int(seed),
        "sample_size": int(sample_size),
    }


def predict_step_coordinate_surrogate(
    fitted: dict[str, Any], height: int, case: dict[str, np.ndarray]
) -> np.ndarray:
    """Predict U,V from height and coordinates; no reference flow enters features."""
    features = step_coordinate_features(
        float(height) / 100.0,
        case["x"],
        case["y"],
        bounds_m=fitted["bounds_m"],
    )
    scaled = fitted["model"].predict(fitted["x_scaler"].transform(features))
    return fitted["y_scaler"].inverse_transform(scaled)


def evaluate_step_coordinate_surrogate(
    fitted: dict[str, Any],
    cases: dict[int, dict[str, np.ndarray]],
    selected_heights: list[int] | np.ndarray,
) -> list[dict[str, float | int]]:
    """Evaluate explicitly supplied cases without modifying the fitted model."""
    rows: list[dict[str, float | int]] = []
    metric_alpha = 0.5 if fitted["alpha"] is None else float(fitted["alpha"])
    for height in np.asarray(selected_heights, dtype=int):
        case = cases[int(height)]
        prediction = predict_step_coordinate_surrogate(fitted, int(height), case)
        metrics = zonal_velocity_metrics(
            case["u"], case["v"], prediction[:, 0], prediction[:, 1], alpha=metric_alpha
        )
        rows.append({"height_percent": int(height), **metrics})
    return rows


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Return a dimensionless relative L2 error over finite paired values."""
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(prediction)
    if not np.any(mask):
        raise ValueError("relative_l2 received no finite paired values")
    denominator = np.linalg.norm(reference[mask])
    if denominator <= np.finfo(float).eps:
        raise ValueError("relative_l2 reference norm is zero")
    return float(np.linalg.norm(prediction[mask] - reference[mask]) / denominator)


def manufactured_step_velocity(
    x: np.ndarray,
    y: np.ndarray,
    height_ratio: float,
    *,
    step_x: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth pedagogical backward-facing-step field.

    This analytic field is not DSMC and is not article evidence.  It creates a
    height-dependent recirculation pocket so students can inspect how a global
    objective underweights a small but important ``u < 0`` region.
    """
    x_array, y_array = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    h = float(height_ratio)
    if not 0.15 <= h <= 0.80:
        raise ValueError("height_ratio must lie in [0.15, 0.80]")

    solid = (x_array < step_x) & (y_array < h)
    lower_wall = np.where(x_array < step_x, h, 0.0)
    eta = np.clip((y_array - lower_wall) / np.maximum(1.0 - lower_wall, 1.0e-12), 0, 1)
    base = 6.0 * eta * (1.0 - eta)

    recirculation_length = 0.65 + 2.1 * h
    xc = step_x + 0.43 * recirculation_length
    yc = 0.30 * h + 0.055
    sx = 0.34 * recirculation_length
    sy = 0.08 + 0.22 * h
    gaussian = np.exp(-((x_array - xc) / sx) ** 2 - ((y_array - yc) / sy) ** 2)
    u = base - (1.25 + 1.25 * h) * gaussian
    v = 0.23 * (x_array - xc) / sx * gaussian
    recovery = 1.0 - 0.08 * np.exp(-np.maximum(x_array - step_x, 0.0) / 1.8)
    u *= recovery
    u[solid] = 0.0
    v[solid] = 0.0
    return u, v, solid


def zonal_velocity_metrics(
    u_reference: np.ndarray,
    v_reference: np.ndarray,
    u_prediction: np.ndarray,
    v_prediction: np.ndarray,
    *,
    alpha: float = 0.7,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Evaluate the full field and the true-flow recirculation zone separately.

    The zonal objective is ``alpha * MSE_vortex + (1-alpha) * MSE_main``.
    Each regional MSE is normalized by its own point count, matching the
    scientific intent of the article's physics-guided zonal loss.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    arrays = np.broadcast_arrays(
        np.asarray(u_reference, dtype=float),
        np.asarray(v_reference, dtype=float),
        np.asarray(u_prediction, dtype=float),
        np.asarray(v_prediction, dtype=float),
    )
    u_ref, v_ref, u_pred, v_pred = arrays
    finite = np.isfinite(u_ref) & np.isfinite(v_ref) & np.isfinite(u_pred) & np.isfinite(v_pred)
    if valid_mask is not None:
        finite &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), finite.shape)
    vortex = finite & (u_ref < 0.0)
    main = finite & ~vortex
    if not np.any(vortex) or not np.any(main):
        raise ValueError("both vortex and main-flow points are required")

    squared = (u_pred - u_ref) ** 2 + (v_pred - v_ref) ** 2
    full_mse = float(np.mean(squared[finite]))
    vortex_mse = float(np.mean(squared[vortex]))
    main_mse = float(np.mean(squared[main]))
    return {
        "full_mse": full_mse,
        "vortex_mse": vortex_mse,
        "main_mse": main_mse,
        "zonal_loss": float(alpha * vortex_mse + (1.0 - alpha) * main_mse),
        "full_relative_l2": relative_l2(
            np.stack((u_ref[finite], v_ref[finite])),
            np.stack((u_pred[finite], v_pred[finite])),
        ),
        "vortex_relative_l2": relative_l2(
            np.stack((u_ref[vortex], v_ref[vortex])),
            np.stack((u_pred[vortex], v_pred[vortex])),
        ),
        "vortex_points": int(np.count_nonzero(vortex)),
        "main_points": int(np.count_nonzero(main)),
    }


def detect_density_shock(
    x_m: np.ndarray,
    density: np.ndarray,
    *,
    smooth_sigma: float = 0.7,
) -> dict[str, float]:
    """Detect the strongest interior centerline-density compression and width."""
    x = np.asarray(x_m, dtype=float)
    rho = np.asarray(density, dtype=float)
    if x.ndim != 1 or rho.shape != x.shape or len(x) < 15:
        raise ValueError("x_m and density must be paired 1-D profiles")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x_m must be strictly increasing")
    smoothed = gaussian_filter1d(rho, smooth_sigma, mode="nearest")
    gradient = np.gradient(smoothed, x, edge_order=2)
    span = float(np.ptp(x))
    candidates = (x >= x[0] + 0.05 * span) & (x <= x[0] + 0.95 * span)
    index = np.flatnonzero(candidates)[np.argmax(np.abs(gradient[candidates]))]
    xs = float(x[index])
    dx = float(np.median(np.diff(x)))
    left = (x >= xs - 12 * dx) & (x <= xs - 3 * dx)
    right = (x >= xs + 3 * dx) & (x <= xs + 12 * dx)
    rho_left = float(np.mean(rho[left])) if np.any(left) else float(rho[0])
    rho_right = float(np.mean(rho[right])) if np.any(right) else float(rho[-1])
    jump = abs(rho_left - rho_right)
    peak = float(abs(gradient[index]))
    width = jump / peak if peak > 0.0 else 5.0 * dx
    if not np.isfinite(width) or width <= 0.0:
        width = 5.0 * dx
    return {
        "shock_x_m": xs,
        "delta_jump_m": float(width),
        "density_left": rho_left,
        "density_right": rho_right,
        "peak_abs_gradient": peak,
    }


def normalize_density_jump(
    x_m: np.ndarray,
    density: np.ndarray,
    shock_x_m: float,
) -> np.ndarray:
    """Normalize a density profile by local pre/post-shock plateaus."""
    x = np.asarray(x_m, dtype=float)
    rho = np.asarray(density, dtype=float)
    dx = float(np.median(np.diff(x)))
    left = (x >= shock_x_m - 12 * dx) & (x <= shock_x_m - 3 * dx)
    right = (x >= shock_x_m + 3 * dx) & (x <= shock_x_m + 12 * dx)
    rho_left = float(np.mean(rho[left])) if np.any(left) else float(rho[0])
    rho_right = float(np.mean(rho[right])) if np.any(right) else float(rho[-1])
    denominator = rho_left - rho_right
    if abs(denominator) <= np.finfo(float).eps:
        denominator = float(np.ptp(rho)) or 1.0
    return (rho - rho_right) / denominator


def density_snapshot_matrix(
    x_m: np.ndarray,
    density: np.ndarray,
    shock_x_m: np.ndarray,
    delta_jump_m: np.ndarray,
    *,
    coordinate: str,
    points: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate jump-normalized profiles on physical or shock-centered axes."""
    x = np.asarray(x_m, dtype=float)
    fields = np.asarray(density, dtype=float)
    xs = np.asarray(shock_x_m, dtype=float)
    delta = np.asarray(delta_jump_m, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != len(x):
        raise ValueError("density must have shape (cases, len(x_m))")
    if xs.shape != (fields.shape[0],) or delta.shape != xs.shape:
        raise ValueError("shock arrays must contain one value per case")
    if coordinate == "physical":
        grid = np.linspace(0.0, 210.0, points)
    elif coordinate == "shock_centered":
        grid = np.linspace(-8.0, 8.0, points)
    else:
        raise ValueError("coordinate must be 'physical' or 'shock_centered'")

    rows = []
    for profile, location, width in zip(fields, xs, delta, strict=True):
        normalized = normalize_density_jump(x, profile, float(location))
        source_coordinate = x * 1.0e6 if coordinate == "physical" else (x - location) / width
        rows.append(
            np.interp(
                grid,
                source_coordinate,
                normalized,
                left=float(normalized[0]),
                right=float(normalized[-1]),
            )
        )
    return grid, np.vstack(rows)


def pod_spectrum(snapshot_matrix: np.ndarray) -> dict[str, np.ndarray | int | float]:
    """Return a centered snapshot POD and its cumulative energy diagnostics."""
    snapshots = np.asarray(snapshot_matrix, dtype=float)
    if snapshots.ndim != 2 or snapshots.shape[0] < 2:
        raise ValueError("snapshot_matrix must contain at least two row snapshots")
    mean = np.mean(snapshots, axis=0)
    _, singular_values, modes = np.linalg.svd(snapshots - mean, full_matrices=False)
    energy = singular_values**2
    fractions = energy / np.sum(energy)
    cumulative = np.cumsum(fractions)
    n99 = int(np.searchsorted(cumulative, 0.99) + 1)
    return {
        "mean": mean,
        "singular_values": singular_values,
        "modes": modes,
        "energy_fraction": fractions,
        "cumulative_energy": cumulative,
        "first_mode_percent": float(100.0 * fractions[0]),
        "first_two_percent": float(100.0 * cumulative[1]),
        "first_three_percent": float(100.0 * cumulative[2]),
        "n99": n99,
    }


def load_nozzle_centerlines(root: str | Path) -> dict[str, np.ndarray]:
    """Load the compact attributed 15-case DSMC centerline archive."""
    root_path = Path(root).resolve()
    path = root_path / "results" / "mahdavi_deeponet" / "nozzle_centerline_15cases.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def validate_step_teaching_results(root: str | Path) -> dict[str, Any]:
    """Validate the split, selection rule, and recorded leakage-free test evidence."""
    result_dir = Path(root).resolve() / "results" / "mahdavi_deeponet"
    for filename, expected_hash in STEP_TEACHING_RESULT_SHA256.items():
        if _sha256(result_dir / filename) != expected_hash:
            raise ValueError(f"micro-step teaching-result SHA-256 mismatch: {filename}")
    protocol = json.loads(
        (result_dir / "step_teaching_protocol.json").read_text(encoding="utf-8")
    )
    expected_split = {
        "development_percent": STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist(),
        "validation_percent": STEP_HEIGHT_VALIDATION_PERCENT.tolist(),
        "held_out_test_percent": STEP_HEIGHT_HELD_OUT_PERCENT.tolist(),
    }
    if protocol["source_commit"] != STEP_SOURCE_COMMIT or protocol["split"] != expected_split:
        raise ValueError("micro-step teaching protocol source or case split mismatch")
    if protocol["archive_sha256"] != STEP_HEIGHT_ARCHIVE_SHA256:
        raise ValueError("micro-step teaching protocol archive identity mismatch")
    if not protocol["file_level_test_isolation"]:
        raise ValueError("micro-step teaching protocol lacks file-level test isolation")
    if protocol["test_used_for_selection"]:
        raise ValueError("micro-step test cases were used for model selection")
    if not protocol["test_archive_opened_only_after_model_selection_and_final_fit"]:
        raise ValueError("micro-step test-open gate is not recorded")
    forbidden = set(protocol["forbidden_inputs"])
    if not {"held-out U", "held-out V", "target-field patch"}.issubset(forbidden):
        raise ValueError("micro-step forbidden-input contract is incomplete")

    with (result_dir / "step_teaching_selection.csv").open(encoding="utf-8") as stream:
        selection = list(csv.DictReader(stream))
    if len(selection) != 5 or selection[0]["model"] != "unweighted":
        raise ValueError("unexpected micro-step validation sweep")
    baseline_global = float(selection[0]["validation_global_relative_l2_percent"])
    candidates = [row for row in selection[1:] if row["model"] == "zonal"]
    selected = [row for row in candidates if row["selected"].lower() == "true"]
    if len(selected) != 1:
        raise ValueError("micro-step validation sweep must select exactly one alpha")
    eligible = [
        row
        for row in candidates
        if float(row["validation_global_relative_l2_percent"]) <= baseline_global + 2.0
    ]
    expected = min(
        eligible, key=lambda row: float(row["validation_vortex_relative_l2_percent"])
    )
    selected_alpha = float(selected[0]["alpha"])
    if selected[0] is not expected or selected_alpha != float(protocol["selected_alpha"]):
        raise ValueError("micro-step selected alpha does not follow the frozen validation rule")

    with (result_dir / "step_teaching_test_metrics.csv").open(encoding="utf-8") as stream:
        test_rows = list(csv.DictReader(stream))
    expected_pairs = {
        ("unweighted", 44), ("unweighted", 67),
        (f"zonal_alpha_{selected_alpha:.1f}", 44),
        (f"zonal_alpha_{selected_alpha:.1f}", 67),
    }
    if {(row["model"], int(row["height_percent"])) for row in test_rows} != expected_pairs:
        raise ValueError("unexpected micro-step held-out metric rows")
    metric_names = ("global_relative_l2_percent", "vortex_relative_l2_percent")
    if not all(
        np.isfinite(float(row[name])) and 0.0 <= float(row[name]) < 100.0
        for row in test_rows
        for name in metric_names
    ):
        raise ValueError("micro-step held-out errors are non-finite or outside sanity bounds")
    indexed = {(row["model"], int(row["height_percent"])): row for row in test_rows}
    zonal_label = f"zonal_alpha_{selected_alpha:.1f}"
    for height in STEP_HEIGHT_HELD_OUT_PERCENT:
        if not float(indexed[(zonal_label, int(height))]["vortex_relative_l2_percent"]) < float(
            indexed[("unweighted", int(height))]["vortex_relative_l2_percent"]
        ):
            raise ValueError(f"H{height}: selected zonal model did not improve vortex error")
    return {
        "status": "pass",
        "selected_alpha": selected_alpha,
        "test_used_for_selection": False,
        "file_level_test_isolation": True,
        "test_metrics": test_rows,
    }


def validate_step_contour_evidence(root: str | Path) -> dict[str, Any]:
    """Validate exact-case article contours and independent held-out contours."""
    root_path = Path(root).resolve()
    result_dir = root_path / "results" / "mahdavi_deeponet"
    article_manifest = json.loads(
        (result_dir / "step_article_contour_manifest.json").read_text(encoding="utf-8")
    )
    no_leak_manifest = json.loads(
        (result_dir / "step_leakage_free_contour_manifest.json").read_text(encoding="utf-8")
    )
    if article_manifest["source_commit"] != STEP_SOURCE_COMMIT:
        raise ValueError("step contour source commit mismatch")
    if article_manifest["schema_version"] != 2:
        raise ValueError("unexpected final-paper contour manifest schema")
    if article_manifest["article"]["final_paper_sha256"] != (
        "47d130cb4fae08608cef2578460665bebdf20a938b489eb5efc914302afcb980"
    ):
        raise ValueError("final-paper identity mismatch")
    if article_manifest["article"]["final_supplement_sha256"] != (
        "5fda3881c60bb77c5c6a7572e269822628e6e844cc438bf165c396f218ab4133"
    ):
        raise ValueError("final-supplement identity mismatch")
    if "privileged-input" not in article_manifest["evidence_boundary"]:
        raise ValueError("article contour evidence boundary is missing")
    if not any("Kn=1" in note for note in article_manifest["article_repository_inconsistencies"]):
        raise ValueError("missing stored Kn=1 prediction is not documented")

    article_metrics_path = result_dir / "step_article_contour_metrics.csv"
    if _sha256(article_metrics_path) != article_manifest["metrics_sha256"]:
        raise ValueError("step article-contour metrics hash mismatch")
    with article_metrics_path.open(encoding="utf-8") as stream:
        article_rows = list(csv.DictReader(stream))
    expected_article = {
        "Kn0p004": (6, 2.4738467759),
        "Kn0p02": (6, 2.2986139762),
        "H44": (15, 4.9249304594),
        "H67": (15, 8.2323163941),
    }
    if set(expected_article) != {row["case_id"] for row in article_rows}:
        raise ValueError("unexpected exact article contour cases")
    for row in article_rows:
        figure, expected_error = expected_article[row["case_id"]]
        if int(row["article_figure"]) != figure:
            raise ValueError(f"{row['case_id']}: article figure mismatch")
        if abs(float(row["combined_relative_l2_percent"]) - expected_error) > 1.0e-8:
            raise ValueError(f"{row['case_id']}: stored-field error mismatch")
        if abs(float(row["L_over_H_from_grid"]) - 5.0) > 2.0e-4:
            raise ValueError(f"{row['case_id']}: contour aspect-ratio contract failed")
        if float(row["max_coordinate_delta"]) > 1.0e-15:
            raise ValueError(f"{row['case_id']}: reference/prediction coordinates differ")

    coverage_path = result_dir / "step_article_case_coverage.csv"
    if _sha256(coverage_path) != article_manifest["case_coverage_sha256"]:
        raise ValueError("step article-case coverage hash mismatch")
    with coverage_path.open(encoding="utf-8") as stream:
        coverage_rows = list(csv.DictReader(stream))
    if {row["case_id"] for row in coverage_rows} != {
        "Kn0p004", "Kn0p02", "Kn1", "H44", "H67"
    }:
        raise ValueError("final-paper contour case coverage is incomplete")
    kn1 = next(row for row in coverage_rows if row["case_id"] == "Kn1")
    if kn1["stored_nn_field"] != "missing from pinned repository":
        raise ValueError("Kn=1 missing-prediction boundary is not explicit")
    kn1_metadata = article_manifest["kn1_reference_only"]
    if abs(float(kn1_metadata["L_over_H_from_grid"]) - 5.0) > 2.0e-4:
        raise ValueError("Kn=1 DSMC-only contour aspect-ratio contract failed")

    if no_leak_manifest["selected_alpha"] != 0.6:
        raise ValueError("no-leak contour manifest has unexpected selected alpha")
    if no_leak_manifest["test_used_for_selection"]:
        raise ValueError("no-leak contours used the test cases for selection")
    if not no_leak_manifest["test_archive_opened_after_final_fit"]:
        raise ValueError("no-leak contour test-open gate is missing")
    if "target-field patch" not in no_leak_manifest["forbidden_inputs"]:
        raise ValueError("no-leak contour forbidden-input contract is incomplete")
    no_leak_metrics_path = result_dir / "step_leakage_free_contour_metrics.csv"
    if _sha256(no_leak_metrics_path) != no_leak_manifest["metrics_sha256"]:
        raise ValueError("no-leak contour metrics hash mismatch")
    with no_leak_metrics_path.open(encoding="utf-8") as stream:
        no_leak_rows = list(csv.DictReader(stream))
    if {int(row["height_percent"]) for row in no_leak_rows} != {44, 67}:
        raise ValueError("unexpected no-leak contour test cases")
    teaching = validate_step_teaching_results(root_path)
    recorded = {
        int(row["height_percent"]): row
        for row in teaching["test_metrics"]
        if row["model"] == "zonal_alpha_0.6"
    }
    for row in no_leak_rows:
        height = int(row["height_percent"])
        for current, retained in (
            ("combined_relative_l2_percent", "global_relative_l2_percent"),
            ("vortex_relative_l2_percent", "vortex_relative_l2_percent"),
        ):
            if abs(float(row[current]) - float(recorded[height][retained])) > 5.0e-7:
                raise ValueError(f"H{height}: no-leak contour metrics differ from frozen test")

    for manifest in (article_manifest, no_leak_manifest):
        for relative, expected_hash in manifest["figure_sha256"].items():
            figure = (root_path / relative).resolve()
            if not figure.is_relative_to(root_path) or _sha256(figure) != expected_hash:
                raise ValueError(f"step contour figure hash mismatch: {relative}")
    return {
        "status": "pass",
        "article_cases": [row["case_id"] for row in article_rows],
        "final_paper_case_coverage": [row["case_id"] for row in coverage_rows],
        "no_leak_test_cases": [int(row["height_percent"]) for row in no_leak_rows],
        "article_results_are_privileged_input": True,
        "no_leak_test_used_for_selection": False,
    }


def validate_week9_evidence(
    root: str | Path,
    *,
    step_source: str | Path | None = None,
    require_step_source: bool = False,
) -> dict[str, Any]:
    """Validate Week-9 evidence and optionally cross-check the upstream step data."""
    root_path = Path(root).resolve()
    result_dir = root_path / "results" / "mahdavi_deeponet"
    provenance = json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
    archive_path = result_dir / "nozzle_centerline_15cases.npz"
    if _sha256(archive_path) != provenance["derived_files"][archive_path.name]:
        raise ValueError("compact nozzle archive SHA-256 mismatch")
    data = load_nozzle_centerlines(root_path)
    if not np.array_equal(data["pressure_kpa"], NOZZLE_PRESSURES_KPA):
        raise ValueError("unexpected nozzle pressure cases")
    required = (
        "density", "u_ms", "v_ms", "temperature_k", "mach",
        "pressure_tecplot", "knudsen", "shock_x_m", "delta_jump_m",
    )
    for name in required:
        if name not in data or not np.isfinite(data[name]).all():
            raise ValueError(f"invalid compact nozzle field: {name}")
    if data["density"].shape != (15, 101) or data["x_m"].shape != (101,):
        raise ValueError("unexpected compact nozzle data shape")

    pod_rows: dict[str, dict[str, float | int]] = {}
    for coordinate in ("physical", "shock_centered"):
        _, matrix = density_snapshot_matrix(
            data["x_m"], data["density"], data["shock_x_m"],
            data["delta_jump_m"], coordinate=coordinate,
        )
        spectrum = pod_spectrum(matrix)
        pod_rows[coordinate] = {
            "E1_percent": float(spectrum["first_mode_percent"]),
            "E12_percent": float(spectrum["first_two_percent"]),
            "E123_percent": float(spectrum["first_three_percent"]),
            "N99": int(spectrum["n99"]),
        }
    with (result_dir / "nozzle_pod_reference.csv").open(encoding="utf-8") as stream:
        reference = {row["coordinate"]: row for row in csv.DictReader(stream)}
    for coordinate, observed in pod_rows.items():
        expected = reference[coordinate]
        for key in ("E1_percent", "E12_percent", "E123_percent"):
            if abs(float(observed[key]) - float(expected[key])) > 0.03:
                raise ValueError(f"{coordinate} POD audit failed for {key}")
        if int(observed["N99"]) != int(expected["N99"]):
            raise ValueError(f"{coordinate} POD audit failed for N99")

    step_manifest = json.loads(
        (result_dir / "step_source_manifest.json").read_text(encoding="utf-8")
    )
    if step_manifest["source_commit"] != STEP_SOURCE_COMMIT:
        raise ValueError("unexpected micro-step source commit")
    if step_manifest["height_percent"] != STEP_HEIGHT_PERCENT.tolist():
        raise ValueError("unexpected micro-step height cases")
    split = step_manifest["split"]
    if split["development_percent"] != STEP_HEIGHT_DEVELOPMENT_PERCENT.tolist():
        raise ValueError("unexpected micro-step development split")
    if split["validation_percent"] != STEP_HEIGHT_VALIDATION_PERCENT.tolist():
        raise ValueError("unexpected micro-step validation split")
    if split["held_out_test_percent"] != STEP_HEIGHT_HELD_OUT_PERCENT.tolist():
        raise ValueError("unexpected micro-step held-out split")
    manifest_hashes = {
        name: record["sha256"] for name, record in step_manifest["smoothed_files"].items()
    }
    if manifest_hashes != STEP_HEIGHT_FILE_SHA256:
        raise ValueError("micro-step source hashes do not match the software contract")
    if step_manifest["study_scope"]["joint_generalization"] != "not demonstrated":
        raise ValueError("micro-step joint-generalization boundary is missing")
    if any(
        "not established" not in value
        for value in step_manifest["dsmc_provenance_status"].values()
    ):
        raise ValueError("micro-step DSMC provenance gaps are not explicit")
    step_report = validate_step_height_archives(root_path)
    teaching_report = validate_step_teaching_results(root_path)
    contour_report = validate_step_contour_evidence(root_path)
    if step_manifest["derived_archives_sha256"] != STEP_HEIGHT_ARCHIVE_SHA256:
        raise ValueError("micro-step archive hashes do not match the software contract")
    for filename, expected_hash in STEP_HEIGHT_ARCHIVE_SHA256.items():
        if provenance["derived_files"].get(filename) != expected_hash:
            raise ValueError(f"micro-step provenance hash mismatch: {filename}")
    for filename, expected_hash in STEP_TEACHING_RESULT_SHA256.items():
        if provenance["derived_files"].get(filename) != expected_hash:
            raise ValueError(f"micro-step result provenance hash mismatch: {filename}")

    with (result_dir / "step_privileged_input_audit.csv").open(encoding="utf-8") as stream:
        patch_audit = list(csv.DictReader(stream))
    if [int(row["height_percent"]) for row in patch_audit] != [44, 67]:
        raise ValueError("unexpected privileged-input audit cases")
    for row in patch_audit:
        stored = float(row["upstream_stored_prediction_relative_l2_percent"])
        direct = float(row["target_patch_nearest_sample_relative_l2_percent"])
        if not direct < stored:
            raise ValueError("target-derived patch baseline no longer exposes leakage")

    discovered_step_source = discover_step_source(root_path, step_source)
    if discovered_step_source is None:
        if require_step_source:
            raise FileNotFoundError(
                "real micro-step data are required; pass --step-source or set "
                "FLOWMLLAB_STEP_SOURCE"
            )
        source_crosscheck: dict[str, Any] = {
            "status": "not_checked",
            "reason": "pinned upstream checkout/cache was not supplied",
            "source_commit": STEP_SOURCE_COMMIT,
        }
    else:
        source_crosscheck = validate_step_height_dataset(discovered_step_source)
        source_crosscheck["compact_archive_match"] = validate_step_archives_against_source(
            root_path, discovered_step_source
        )

    boundary = provenance["claim_boundary"]
    if "real dsmc" not in boundary["microstep_data"].lower():
        raise ValueError("micro-step data claim boundary is missing")
    if "does not receive" not in boundary["microstep_teaching_model"].lower():
        raise ValueError("micro-step inference boundary is missing")
    if "privileged" not in boundary["microstep_published_model"].lower():
        raise ValueError("published micro-step patch boundary is missing")
    if "not" not in boundary["micro_nozzle_teaching_model"].lower():
        raise ValueError("micro-nozzle model claim boundary is missing")
    return {
        "status": "pass",
        "step_dataset": step_report,
        "step_source_crosscheck": source_crosscheck,
        "step_teaching_validation": teaching_report,
        "step_contour_evidence": contour_report,
        "step_privileged_input_audit": patch_audit,
        "nozzle_cases": int(len(data["pressure_kpa"])),
        "held_out_pressures_kpa": NOZZLE_HELD_OUT_KPA.astype(int).tolist(),
        "physical_density_pod": pod_rows["physical"],
        "shock_centered_density_pod": pod_rows["shock_centered"],
        "archive_sha256": provenance["derived_files"][archive_path.name],
    }
