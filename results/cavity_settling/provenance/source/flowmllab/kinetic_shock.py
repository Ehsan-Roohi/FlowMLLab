"""CPU kinetic-shock teaching tools, independently implemented for FlowMLLab.

Saved results are attributed, hash-bound extracts from PINNs-DVM.  No upstream
solver code or neural checkpoints are imported.  A spatially one-dimensional
monatomic gas still has three molecular velocity components; R=1 and gamma=5/3.
The small fresh exercise is a homogeneous collision/quadrature experiment, not
a newly computed shock solution or a training reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.polynomial.legendre import leggauss

GAMMA = 5.0 / 3.0
UPSTREAM_COMMIT = "add5388f7da5f7eace95fff97d34d6ed58b90169"
FIELDS = ("rho", "u", "T", "q", "sigma", "m", "R", "Delta", "S")
UNITS = "nondimensional_R1_gamma5over3"


def data_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "week10_2_kinetic_shock"


def _finite(a: Any, name: str) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite numbers")
    return out


def _x(x: Any) -> np.ndarray:
    out = _finite(x, "x")
    if out.ndim != 1 or len(out) < 2 or np.any(np.diff(out) <= 0):
        raise ValueError("x must be a strictly increasing vector with at least two nodes")
    return out


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _verified_file(root: Path, entry: dict[str, Any]) -> Path:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("data path escapes the extract directory")
    if not path.is_file():
        raise ValueError(f"missing data file: {entry['path']}")
    raw = path.read_bytes()
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"data hash/size mismatch: {entry['path']}")
    return path


def load_manifest(data_root: str | Path | None = None) -> dict[str, Any]:
    root = data_directory() if data_root is None else Path(data_root)
    try:
        result = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("missing or malformed kinetic-shock manifest") from exc
    if (result.get("schema_version") != 1 or result.get("upstream_commit") != UPSTREAM_COMMIT
            or result.get("units") != UNITS or not isinstance(result.get("cases"), dict)):
        raise ValueError("manifest schema, upstream commit, or units mismatch")
    return result


def available_cases(data_root: str | Path | None = None) -> tuple[str, ...]:
    return tuple(load_manifest(data_root)["cases"])


@dataclass(frozen=True)
class ProfileCase:
    """Profiles already on the recorded common grid; no implicit alignment."""

    x: np.ndarray
    columns: tuple[str, ...]
    profiles: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def field(self, method: str, field: str) -> np.ndarray:
        if method not in self.profiles or field not in self.columns:
            raise ValueError(f"unknown method/field: {method}/{field}")
        return self.profiles[method][:, self.columns.index(field)]


def load_case(case_id: str, data_root: str | Path | None = None) -> ProfileCase:
    """Verify SHA256, schema, units, case identity and grid before returning data.

    Transient CSVs retain their supplied comparison grid.  The supervised v2
    extract explicitly records linear interpolation onto the 399 of 401 existing
    equation-trained nodes within common coverage. Neither loader fits shifts,
    rescales amplitudes, smooths data, or extrapolates the two excluded endpoints.
    """
    root = data_directory() if data_root is None else Path(data_root)
    manifest = load_manifest(root)
    if case_id not in manifest["cases"]:
        raise ValueError(f"unknown kinetic-shock case: {case_id}")
    meta = manifest["cases"][case_id]
    if (meta.get("case_id") != case_id or meta.get("units") != UNITS
            or meta.get("problem") not in {"transient_shock_tube", "steady_normal_shock"}
            or meta.get("collision_model") not in {"BGK", "Shakhov"}):
        raise ValueError("case identity, collision model, problem, or units mismatch")
    if case_id.startswith("tube_") and (
        meta["problem"] != "transient_shock_tube"
        or meta["collision_model"].lower() != case_id.removeprefix("tube_")
        or meta.get("time") != 0.02 or meta.get("tau_law") != "constant_1"
    ):
        raise ValueError("shock-tube case/physics mismatch")
    if case_id.startswith("m2_") and (
        meta["problem"] != "steady_normal_shock" or meta["collision_model"] != "Shakhov"
        or meta.get("mach") != 2 or meta.get("tau_law") != "1/rho"
        or not np.isclose(meta.get("Pr", np.nan), 2 / 3)
    ):
        raise ValueError("stationary Mach-2 case/physics mismatch")
    if case_id.startswith("m2_") and case_id != f"m2_seed{meta.get('seed')}_{meta.get('stage')}":
        raise ValueError("stationary seed/stage identity mismatch")
    if case_id == "supervised_s0_current" and (
        meta["problem"] != "transient_shock_tube" or meta["collision_model"] != "BGK"
        or meta.get("tau_law") != "constant_1" or meta.get("time") != 0.02
        or meta.get("paper_admission") is not False or meta.get("independent_test_claim") is not False
    ):
        raise ValueError("supervised case physics/validation-scope mismatch")
    if "upstream_path" in meta and meta["path"] != "upstream/" + meta["upstream_path"]:
        raise ValueError("upstream file/case path mismatch")
    path = _verified_file(root, meta)
    columns = tuple(meta["columns"])
    if not columns or len(set(columns)) != len(columns) or any(c not in FIELDS for c in columns):
        raise ValueError("unknown or duplicated field columns")
    expected_columns = FIELDS[:5] if case_id.startswith("m2_") else FIELDS
    if columns != expected_columns:
        raise ValueError("case-specific field schema mismatch")
    profiles: dict[str, np.ndarray] = {}
    if meta["format"] == "matched_csv":
        values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
        names = ("x",) + tuple(f"{prefix}_{c}" for prefix in ("parent", "final", "DVM") for c in columns)
        if values.dtype.names != names:
            raise ValueError("CSV column schema mismatch")
        x = _x(values["x"])
        for method, prefix in (("parent", "parent"), ("final", "final"), ("reference", "DVM")):
            profiles[method] = np.column_stack([values[f"{prefix}_{c}"] for c in columns])
    elif meta["format"] in {"m2_core_npz", "supervised_comparison_npz"}:
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(meta["npz_keys"]):
                raise ValueError("NPZ key schema mismatch")
            x = _x(arrays["x"])
            if "columns" in arrays and tuple(arrays["columns"].tolist()) != columns:
                raise ValueError("NPZ field columns mismatch")
            if "case_id" in arrays and str(arrays["case_id"]) != case_id:
                raise ValueError("NPZ case identity mismatch")
            for method, key in meta["profile_keys"].items():
                profiles[method] = arrays[key].astype(np.float64, copy=True)
            if "flux_defect_percent" in arrays:
                defect = _finite(arrays["flux_defect_percent"], "saved flux defects")
                if defect.shape != (len(x), 3):
                    raise ValueError("saved flux-defect shape mismatch")
    else:
        raise ValueError("unknown case format")
    if (len(x) != meta["rows"] or not np.allclose(
        x[[0, -1]], meta["x_endpoints"], rtol=0, atol=1e-13
    )):
        raise ValueError("case coordinate coverage mismatch")
    coordinate_hash = hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()
    if coordinate_hash != meta["x_sha256"]:
        raise ValueError("case coordinate hash mismatch")
    if tuple(profiles) != tuple(meta["methods"]) or "reference" not in profiles:
        raise ValueError("case method identity mismatch")
    for method, values in profiles.items():
        _finite(values, method)
        if values.shape != (len(x), len(columns)):
            raise ValueError(f"profile shape mismatch for {method}")
        for positive_field in ("rho", "T"):
            if positive_field in columns and np.any(values[:, columns.index(positive_field)] <= 0):
                raise ValueError(f"nonpositive {positive_field} in {method}")
        values.setflags(write=False)
    x.setflags(write=False)
    return ProfileCase(x, columns, profiles, dict(meta))


def load_observation_locations(data_root: str | Path | None = None) -> dict[str, Any]:
    """Exact frozen data-assisted observation x locations, with their field roles.

    160 macro locations times 3 fields plus 42 moment locations times 5 fields
    give 690 scalar observations. Excluding these does not undo reference-based
    scales and acceptance selection. These arrays do not define a test split.
    """
    root = data_directory() if data_root is None else Path(data_root)
    meta = load_manifest(root)["cases"]["supervised_s0_current"]
    path = _verified_file(root, meta["observation_sets"])
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"macro_x", "moment_x", "original_macro_x", "original_moment_x",
                                "macro_columns", "moment_columns"}:
            raise ValueError("observation-location schema mismatch")
        out = {k: arrays[k].copy() for k in arrays.files}
    for key, count in (("macro_x", 160), ("moment_x", 42),
                       ("original_macro_x", 160), ("original_moment_x", 42)):
        value = _finite(out[key], key)
        if value.shape != (count,) or np.any(abs(value) > 0.5):
            raise ValueError("observation-location shape or domain mismatch")
        value.setflags(write=False)
        out[key] = value
    if (tuple(out["macro_columns"]) != ("rho", "u", "T")
            or tuple(out["moment_columns"]) != ("q", "sigma", "m", "R", "Delta")):
        raise ValueError("observation field-role mismatch")
    out.update(scalar_observations=690, independent_test_claim=False,
               reference_used_for_scales_and_acceptance=True,
               source_path=meta["observation_sets"]["upstream_path"])
    return out


def profile_metrics(x: Any, prediction: Any, reference: Any,
                    global_scale: Any | None = None) -> dict[str, Any]:
    """Spatial trapezoidal L2; signed errors scaled by a whole-profile amplitude.

    Rows are spatial samples.  Zero reference L2 or zero/negative global scales
    raise explicitly; no epsilon denominator or pointwise percentage is used.
    This spatial norm is not the paper's unweighted 401-point Euclidean norm.
    """
    x = _x(x)
    p, r = _finite(prediction, "prediction"), _finite(reference, "reference")
    if p.shape != r.shape or p.ndim not in (1, 2) or p.shape[0] != len(x):
        raise ValueError("matched prediction/reference shapes must be (Nx,) or (Nx,Nfields)")
    err = p - r
    reference_l2 = np.sqrt(np.trapezoid(r * r, x, axis=0))
    error_l2 = np.sqrt(np.trapezoid(err * err, x, axis=0))
    if np.any(reference_l2 <= 0):
        raise ValueError("relative L2 requires a nonzero reference norm for every field")
    scale = np.max(np.abs(r), axis=0) if global_scale is None else _finite(global_scale, "global_scale")
    if np.shape(scale) not in ((), r.shape[1:]) or np.any(scale <= 0):
        raise ValueError("global_scale must be positive and scalar or one value per field")
    return {"relative_l2_percent": 100 * error_l2 / reference_l2,
            "reference_l2": reference_l2, "error_l2": error_l2,
            "signed_error": err, "signed_error_percent": 100 * err / scale,
            "global_scale": scale, "max_abs_error": np.max(np.abs(err), axis=0),
            "norm": "spatial_trapezoidal_L2", "alignment": "none"}


def jump_states(mach: float) -> dict[str, Any]:
    """Monatomic normal-shock RH states in upstream rho1=T1=R=1 units."""
    mach = float(mach)
    if not np.isfinite(mach) or mach < 1:
        raise ValueError("normal-shock Mach must be finite and at least one")
    u1 = mach * np.sqrt(GAMMA)
    rho2 = 4 * mach**2 / (mach**2 + 3)
    return {"mach": mach, "gamma": GAMMA,
            "upstream": {"rho": 1.0, "u": u1, "T": 1.0},
            "downstream": {"rho": rho2, "u": u1 / rho2,
                           "T": (5 * mach**2 - 1) / (4 * rho2)}}


def stationary_fluxes(rho: Any, u: Any, T: Any, q: Any, sigma: Any) -> dict[str, np.ndarray]:
    """Signed kinetic stress: Pi=rho*u^2+rho*T+sigma, E=...+u*sigma+q."""
    rho, u, T, q, sigma = np.broadcast_arrays(*[_finite(v, n) for v, n in
                                               zip((rho, u, T, q, sigma), ("rho", "u", "T", "q", "sigma"))])
    if np.any(rho <= 0) or np.any(T <= 0):
        raise ValueError("density and temperature must be positive")
    return {"mass": rho * u,
            "momentum": rho * u**2 + rho * T + sigma,
            "energy": rho * u * (0.5 * u**2 + 2.5 * T) + u * sigma + q}


def stationary_flux_diagnostics(case: ProfileCase, method: str) -> dict[str, Any]:
    """Compare steady fluxes with analytic upstream RH values, never transient data."""
    if case.metadata.get("problem") != "steady_normal_shock":
        raise ValueError("constant-flux diagnostics require a steady normal shock")
    state = jump_states(case.metadata["mach"])["upstream"]
    target = stationary_fluxes(**state, q=0, sigma=0)
    flux = stationary_fluxes(*(case.field(method, name) for name in ("rho", "u", "T", "q", "sigma")))
    defect = {k: 100 * (value - target[k]) / abs(target[k]) for k, value in flux.items()}
    return {"fluxes": flux, "upstream_fluxes": target, "defect_percent": defect,
            "max_abs_defect_percent": {k: float(np.max(np.abs(v))) for k, v in defect.items()},
            "alignment": "none", "stress_sign": "kinetic_deviatoric_second_central_moment"}


def reduced_moments(vx: Any, weights: Any, g: Any, h: Any, k: Any | None = None) -> dict[str, float]:
    """Moments of g=int f dvperp, h=int r^2 f dvperp, k=int r^4 f dvperp.

    Arrays are one axial-velocity vector. R and Delta require k; S=R+Delta/3
    cancels k and is available from g,h. No positivity clipping is performed.
    """
    v, w, g, h = [_finite(a, n) for a, n in zip((vx, weights, g, h), ("vx", "weights", "g", "h"))]
    if v.ndim != 1 or len(v) < 2 or any(a.shape != v.shape for a in (w, g, h)) or np.any(w <= 0):
        raise ValueError("reduced quadrature needs matching vectors and positive weights")
    rho = float(w @ g)
    _positive(rho, "rho")
    u = float(w @ (v * g) / rho)
    c = v - u
    T = float(w @ (c**2 * g + h) / (3 * rho))
    _positive(T, "T")
    sigma = float(w @ ((2 * c**2 * g - h) / 3))
    out = {"rho": rho, "u": u, "T": T,
           "q": float(0.5 * (w @ (c * (c**2 * g + h)))), "sigma": sigma,
           "m": float(w @ ((2 * c**3 * g - 3 * c * h) / 5)),
           "S": float(w @ (c**4 * g + c**2 * h) - 7 * T * sigma - 5 * rho * T**2)}
    if k is not None:
        k = _finite(k, "k")
        if k.shape != v.shape:
            raise ValueError("k must match the axial quadrature")
        out["R"] = float(w @ ((2 * c**4 * g + c**2 * h - k) / 3) - 7 * T * sigma)
        out["Delta"] = float(w @ (c**4 * g + 2 * c**2 * h + k) - 15 * rho * T**2)
    return out


def transverse_counterexample(order: int = 64, cutoff: float = 9.0) -> dict[str, Any]:
    """Two positive distributions share g,h but have different transverse fourth moments.

    A=Gaussian(vx;0,1)*Gaussian2D(Tperp=1). B uses an equal transverse
    mixture Tperp=0.5 and 1.5. Analytically A:(R,Delta)=(0,0), B:(-2/3,2).
    """
    order = _order(order, maximum=160)
    cutoff = _positive(cutoff, "cutoff")
    z, w = leggauss(order)
    v, w = cutoff * z, cutoff * w
    g = np.exp(-0.5 * v**2) / np.sqrt(2 * np.pi)
    h, ka, kb = 2 * g, 8 * g, 10 * g
    return {"vx": v, "weights": w, "g": g, "h": h, "k_A": ka, "k_B": kb,
            "A": reduced_moments(v, w, g, h, ka), "B": reduced_moments(v, w, g, h, kb),
            "without_k": reduced_moments(v, w, g, h),
            "analytic_R_Delta": {"A": (0.0, 0.0), "B": (-2 / 3, 2.0)},
            "positive_distributions": True, "clipping_applied": False}


def _order(order: int, maximum: int = 48) -> int:
    if isinstance(order, bool) or int(order) != order or not 4 <= int(order) <= maximum:
        raise ValueError(f"quadrature order must be an integer between 4 and {maximum}")
    return int(order)


def velocity_quadrature(order: int = 20, cutoff: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre nodes/weights on [-cutoff,cutoff]^3, at most 48^3."""
    order, cutoff = _order(order), _positive(cutoff, "cutoff")
    if cutoff > 20:
        raise ValueError("teaching velocity cutoff must not exceed 20")
    z, w = leggauss(order)
    axes = [cutoff * z] * 3
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    weight = np.prod(np.stack(np.meshgrid(*([cutoff * w] * 3), indexing="ij"), axis=-1), axis=-1).ravel()
    return nodes, weight


def maxwellian(velocity: Any, rho: float = 1.0, u: float = 0.0, T: float = 1.0) -> np.ndarray:
    """Three-velocity Maxwellian with streamwise bulk velocity u."""
    v = _finite(velocity, "velocity")
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("velocity must have shape (Nv,3)")
    rho, T = _positive(rho, "rho"), _positive(T, "T")
    u = float(_finite(u, "u"))
    c = v - np.array([u, 0.0, 0.0])
    return rho / (2 * np.pi * T)**1.5 * np.exp(-np.sum(c * c, axis=1) / (2 * T))


def distribution_moments(velocity: Any, weights: Any, f: Any) -> dict[str, Any]:
    """Full 3D central moments; only planar symmetric distributions are admitted.

    Signed f is allowed to inspect Shakhov targets. The returned rho and T must
    remain positive. Transverse mean velocities are measured, not assumed away.
    """
    v, w, f = _finite(velocity, "velocity"), _finite(weights, "weights"), _finite(f, "f")
    if v.ndim != 2 or v.shape[1] != 3 or w.shape != (len(v),) or f.shape != w.shape or np.any(w <= 0):
        raise ValueError("full quadrature requires velocity(Nv,3), positive weights(Nv), f(Nv)")
    fw = f * w
    rho = _positive(float(fw.sum()), "rho")
    bulk = fw @ v / rho
    if np.max(np.abs(bulk[1:])) > 1e-10 * max(1.0, abs(bulk[0])):
        raise ValueError("teaching moments require zero transverse mean velocity")
    c = v - bulk
    e = np.sum(c * c, axis=1)
    cx = c[:, 0]
    T = _positive(float(fw @ e / (3 * rho)), "T")
    sigma = float(fw @ (cx**2 - e / 3))
    R = float(fw @ ((e - 7 * T) * (cx**2 - e / 3)))
    Delta = float(fw @ e**2 - 15 * rho * T**2)
    return {"rho": rho, "u": float(bulk[0]), "T": T, "q": float(0.5 * fw @ (cx * e)),
            "sigma": sigma, "m": float(fw @ (cx**3 - 0.6 * cx * e)),
            "R": R, "Delta": Delta, "S": R + Delta / 3,
            "transverse_mean": bulk[1:]}


def collision_targets(velocity: Any, moments: dict[str, Any], Pr: float = 2 / 3) -> dict[str, np.ndarray]:
    """Analytic Maxwellian and Shakhov targets on the supplied finite grid.

    Quadrature conservation is measured downstream, not forced by rescaling
    these targets. Negative Shakhov tails are returned without clipping.
    """
    Pr = _positive(Pr, "Pr")
    v = _finite(velocity, "velocity")
    rho, T = _positive(moments["rho"], "rho"), _positive(moments["T"], "T")
    u, q = float(_finite(moments["u"], "u")), float(_finite(moments["q"], "q"))
    eq = maxwellian(v, rho, u, T)
    c = v - np.array([u, 0.0, 0.0])
    correction = (1 - Pr) * q * c[:, 0] / (5 * rho * T**2) * (np.sum(c * c, axis=1) / T - 5)
    return {"BGK": eq, "Shakhov": eq * (1 + correction)}


def collision_experiment(order: int = 20, cutoff: float = 7.0,
                         Pr: float = 2 / 3, tau: float = 1.0) -> dict[str, Any]:
    """Fresh bounded collision test of a positive two-Maxwellian mixture.

    Mixture fractions .7/.3, velocities .8/-1.2, temperatures .7/1.4.
    Continuum rho=1, u=.2, T=1.19, sigma=.56, q=-1.071.
    q-rate is the collision projection in the current fixed peculiar-velocity
    frame. Conservation defects must tend to zero before interpreting this as
    the continuum central-moment rate. No time stepping, fitting, or training.
    """
    tau, Pr = _positive(tau, "tau"), _positive(Pr, "Pr")
    v, w = velocity_quadrature(order, cutoff)
    f = 0.7 * maxwellian(v, u=0.8, T=0.7) + 0.3 * maxwellian(v, u=-1.2, T=1.4)
    moments = distribution_moments(v, w, f)
    targets = collision_targets(v, moments, Pr)
    c = v - np.array([moments["u"], 0.0, 0.0])
    e = np.sum(c * c, axis=1)
    invariants = np.column_stack([np.ones(len(v)), v, 0.5 * np.sum(v * v, axis=1)])
    moment_kernel = 0.5 * c[:, 0] * e
    stress_kernel = c[:, 0]**2 - e / 3
    rows = {}
    for name, target in targets.items():
        collision = (target - f) / tau
        invariant_rate = (w * collision) @ invariants
        expected_q = -(1 if name == "BGK" else Pr) * moments["q"] / tau
        expected_sigma = -moments["sigma"] / tau
        q_rate = float(w @ (collision * moment_kernel))
        stress_rate = float(w @ (collision * stress_kernel))
        negative = target < 0
        rows[name] = {"invariant_rate": invariant_rate,
                      "invariant_names": ("mass", "momentum_x", "momentum_y", "momentum_z", "energy"),
                      "q_rate": q_rate, "expected_q_rate": expected_q,
                      "q_rate_absolute_error": abs(q_rate - expected_q),
                      "stress_rate": stress_rate, "expected_stress_rate": expected_sigma,
                      "stress_rate_absolute_error": abs(stress_rate - expected_sigma),
                      "minimum_target": float(target.min()), "negative_nodes": int(negative.sum()),
                      "negative_mass": float(-w[negative] @ target[negative]),
                      "negative_target_weight_fraction": float(w[negative].sum() / w.sum()),
                      "q_rate_ratio": tau * q_rate / moments["q"],
                      "sigma_rate_ratio": tau * stress_rate / moments["sigma"],
                      "clipping_applied": False, "conservation_correction_applied": False}
    return {"velocity": v, "weights": w, "f": f, "targets": targets, "moments": moments,
            "diagnostics": rows, "order": int(order), "cutoff": float(cutoff), "Pr": Pr, "tau": tau,
            "input_positive": bool(np.all(f > 0)), "clipping_applied": False,
            "conservation_correction_applied": False, "is_shock_solution": False,
            "reference": "analytic_continuum_collision_identities",
            "analytic_mixture_moments": {"rho": 1.0, "u": 0.2, "T": 1.19, "sigma": 0.56, "q": -1.071},
            "summary": {"order": int(order), "cutoff": float(cutoff), "Pr": Pr, "tau": tau,
                        **{k: moments[k] for k in ("rho", "u", "T", "q", "sigma")},
                        **{name: {"mass_defect": row["invariant_rate"][0],
                                  "momentum_defect": row["invariant_rate"][1],
                                  "energy_defect": row["invariant_rate"][4],
                                  **{k: row[k] for k in ("q_rate_ratio", "sigma_rate_ratio", "minimum_target",
                                                       "negative_target_weight_fraction", "negative_mass")}}
                           for name, row in rows.items()}},
            "tolerance_policy": "report measured finite-grid defects; refine order and cutoff separately"}
