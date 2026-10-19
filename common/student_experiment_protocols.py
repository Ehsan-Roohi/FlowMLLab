"""Small, inspectable experiment contracts; no training or solver starts on import."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np


def dataset_fingerprint(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        value = np.asarray(data[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def kinetic_request(configs, solver_file, solver_version):
    """The cache identity includes every dataclass setting and solver bytes."""
    return {"schema": "kinetic-case-contract", "configs": [asdict(c) for c in configs],
            "solver_version": solver_version,
            "solver_sha256": hashlib.sha256(Path(solver_file).read_bytes()).hexdigest(),
            "wall_model": "diffuse thermal walls", "collision_model": "teaching stochastic cell pairs",
            "temperature_estimator": "pooled central second moment with Bessel correction"}


def request_fingerprint(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_kinetic_cache(path, cases, request):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to replace evidence: {path}")
    if [c["config"] for c in cases] != request["configs"]:
        raise ValueError("Generated cases do not match the requested configuration order")
    manifest = {"request": request, "request_sha256": request_fingerprint(request),
                "cases": [{k: c[k] for k in ("config", "field_samples", "temperature_estimator",
                            "sampling_start_time", "sampling_end_time", "sampling_window_time",
                            "mean_particle_observations_per_cell", "runtime_s")} for c in cases]}
    payload = {k: np.asarray([c["config"][k] for c in cases]) for k in ("Kn", "Uwall", "seed")}
    payload.update({k: np.stack([c[k] for c in cases]) for k in ("u", "v", "T", "rho")})
    payload.update(x=cases[0]["x"], y=cases[0]["y"], metadata_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    # Exclusive creation protects retained archives even if another process races us.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)


def load_kinetic_cache(path, request):
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Legacy cache lacks a full budget contract; keep it as historical evidence and use a new cache")
        manifest = json.loads(str(archive["metadata_json"]))
        if manifest["request"] != request or manifest["request_sha256"] != request_fingerprint(request):
            raise ValueError("Cache settings/source differ from this request; choose a new cache path")
        data = {k: archive[k] for k in archive.files if k != "metadata_json"}
    expected = request["configs"]
    actual = list(zip(data["Kn"].tolist(), data["Uwall"].tolist(), data["seed"].tolist()))
    if actual != [(c["Kn"], c["Uwall"], c["seed"]) for c in expected]:
        raise ValueError("Cached case identities/order do not match the manifest")
    return data, manifest


def interpolate_coefficients(re_values, coefficients, query):
    """Interpolate within support only; extrapolation must be a separate declared baseline."""
    order = np.argsort(re_values)
    x = np.asarray(re_values)[order]
    c = np.asarray(coefficients)[order]
    if np.any(np.diff(x) <= 0) or not x[0] <= query <= x[-1]:
        raise ValueError("Need distinct case Re values and an in-range query")
    return np.asarray([np.interp(query, x, c[:, j]) for j in range(c.shape[1])])


def select_physical_rank(selection, limits, relative_tolerance=0.10):
    """All gates are validation-only and declared before candidates are evaluated."""
    eligible = selection.copy()
    for key, limit in limits.items():
        eligible = eligible[np.isfinite(eligible[key]) & (eligible[key] <= limit)]
    if eligible.empty:
        raise ValueError("No candidate passes the declared validation physics gates. Report failure; do not open test to adjust limits.")
    best = eligible["relative_L2_uv"].min()
    eligible = eligible[eligible["relative_L2_uv"] <= (1 + relative_tolerance) * best]
    return eligible.sort_values(["rank", "relative_L2_p"]).iloc[0]


def profile_comparison(reference_x, reference_y, model_x, model_y, core=(0.1, 0.9)):
    """Compare on digitized coordinates without extrapolation or smoothing."""
    x, y, mx, my = map(lambda a: np.asarray(a, dtype=float), (reference_x, reference_y, model_x, model_y))
    if not all(np.isfinite(a).all() for a in (x, y, mx, my)) or np.any(np.diff(mx) <= 0):
        raise ValueError("Finite values and strictly increasing model coordinates required")
    if np.any(x < mx[0]) or np.any(x > mx[-1]):
        raise ValueError("Reference extends outside sampled gas-cell support; report the unsupported corners separately")
    pred = np.interp(x, mx, my)
    reports = {}
    for name, mask in (("full_supported_lid", np.ones(len(x), dtype=bool)), ("interior_lid", (x >= core[0]) & (x <= core[1]))):
        if not mask.any():
            raise ValueError(f"No points in {name}")
        error = pred[mask] - y[mask]
        norm = np.linalg.norm(y[mask])
        reports[name] = {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(error**2))),
                         "max_abs": float(np.max(np.abs(error))),
                         "relative_l2": float(np.linalg.norm(error)/norm) if norm else None}
    return pred, reports
