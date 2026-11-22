"""Physical-time cavity startup snapshots using the existing MAC PISO/PIMPLE solver.

This teaching companion does not change the retained steady solver. Every frame
is an actual solution at its stated nondimensional time t* = t U_lid/L, not an
interpolation between steady states or pressure-correction iterations.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .pressure_velocity import coupling_step, derived_fields, divergence, _residual
from .pressure_velocity_experiments import ExperimentalConfig


DIAGNOSTIC_COLUMNS = (
    "step", "time", "dt", "divergence_linf", "time_step_momentum_defect_linf",
    "steady_momentum_defect_linf", "pressure_mean", "normal_wall_velocity_linf",
    "kinetic_energy", "advective_cfl", "pressure_solves_cumulative",
)


def observation_times(end_time=10.0, interval=0.1):
    """A monotone schedule including rest and the requested final time exactly."""
    if not (np.isfinite(end_time) and np.isfinite(interval) and end_time > 0 and interval > 0):
        raise ValueError("Finite positive end_time and frame interval are required")
    times = np.arange(int(np.floor(end_time / interval)) + 1, dtype=float) * interval
    if end_time - times[-1] > 1e-12 * max(1.0, end_time):
        times = np.r_[times, end_time]
    else:
        times[-1] = end_time
    return times


def _schedule(config, times):
    times = np.asarray(times, dtype=float)
    if config.method not in ("piso", "pimple"):
        raise ValueError("Startup movies require PISO or PIMPLE physical-time steps; SIMPLE is steady-only")
    if not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("A finite positive physical timestep is required")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("A finite one-dimensional observation schedule is required")
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Observation times must start at zero and increase strictly")
    if config.end_time is not None and not np.isclose(times[-1], config.end_time, rtol=0, atol=1e-12):
        raise ValueError("The last frame must equal the configured physical end_time")
    return times


def simulate_startup(config, times, progress=False, divergence_limit=1e-8):
    """Advance from rest, shortening a step only to land on a requested frame.

    Pressure is native transient MAC pressure, shifted to zero spatial mean.
    The finite-volume wall condition is an impulsively started unit-speed lid;
    the other walls are stationary. Lid-corner discontinuities are retained.
    A small divergence is necessary but does not establish time/grid accuracy.
    Nonlinear momentum defects are recorded separately for every actual step.
    """
    times = _schedule(config, times)
    n = config.cells
    h = 1.0 / n
    u = np.zeros((n, n + 1))
    v = np.zeros((n + 1, n))
    p = np.zeros((n, n))
    saved_u, saved_v, saved_p = [u.copy()], [v.copy()], [p.copy()]
    diagnostics = []
    now, step, pressure_solves = 0.0, 0, 0
    start = perf_counter()
    for target in times[1:]:
        while target - now > 1e-12 * max(1.0, target):
            if step >= config.max_iterations:
                raise RuntimeError("max_iterations exhausted before the requested final frame")
            dt = min(config.dt, float(target - now))
            old_u, old_v = u.copy(), v.copy()
            u, v, p, count = coupling_step(u, v, p, config, dt=dt)
            step += 1
            pressure_solves += count
            now += dt
            if not all(np.isfinite(a).all() for a in (u, v, p)):
                raise FloatingPointError("Non-finite startup field; no movie may be labelled complete")
            mass = float(np.max(np.abs(divergence(u, v, h))))
            wall = float(max(np.max(np.abs(u[:, [0, -1]])), np.max(np.abs(v[[0, -1], :]))))
            if mass > divergence_limit or wall > divergence_limit:
                raise RuntimeError(f"Mass/wall gate failed at t*={now:g}: divergence={mass:g}, wall={wall:g}")
            derived = derived_fields(u, v, h)
            cfl = dt / h * (float(np.max(np.abs(u))) + float(np.max(np.abs(v))))
            diagnostics.append([
                step, now, dt, mass, _residual(u, v, p, config, old_u, old_v, dt),
                _residual(u, v, p, config), float(p.mean()), wall,
                derived["kinetic_energy"], cfl, pressure_solves,
            ])
        now = float(target)
        saved_u.append(u.copy()); saved_v.append(v.copy()); saved_p.append(p.copy())
        if progress:
            print(f"Re={config.reynolds:g} N={n} t*={target:g} steps={step}", flush=True)
    result = {
        "config": asdict(config), "times": times.copy(),
        "u_faces": np.asarray(saved_u), "v_faces": np.asarray(saved_v),
        "p": np.asarray(saved_p), "diagnostics": np.asarray(diagnostics),
        "runtime_seconds": perf_counter() - start, "pressure_solves": pressure_solves,
        "initial_condition": "fluid at rest; impulsively started unit lid at t*=0",
        "pressure_gauge": "cell-area mean pressure is zero at each time",
        "time_integration": "backward Euler with PISO/PIMPLE pressure-velocity coupling",
        "qualification": "2D finite-time teaching trajectory; not a steady Ghia or 3D stability validation",
    }
    validate_trajectory(result)
    return result


def validate_trajectory(result):
    """Reject incomplete, mislabelled or non-finite scientific frame payloads."""
    config = ExperimentalConfig(**result["config"])
    times = _schedule(config, result["times"])
    n, frames = config.cells, len(times)
    for key, shape in (("u_faces", (frames, n, n + 1)), ("v_faces", (frames, n + 1, n)), ("p", (frames, n, n))):
        value = np.asarray(result[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} trajectory shape or values")
    if not all(np.count_nonzero(result[key][0]) == 0 for key in ("u_faces", "v_faces", "p")):
        raise ValueError("This startup route requires the recorded initial interior to be at rest")
    if np.max(np.abs(result["p"].mean(axis=(1, 2)))) > 1e-10:
        raise ValueError("Pressure frames do not share the declared zero-mean gauge")
    u, v = result["u_faces"], result["v_faces"]
    if max(np.max(np.abs(u[:, :, [0, -1]])), np.max(np.abs(v[:, [0, -1], :]))) > 1e-8:
        raise ValueError("Saved frames violate wall impermeability")
    mass = (np.diff(u, axis=2) + np.diff(v, axis=1)) * n
    if np.max(np.abs(mass)) > 1e-8:
        raise ValueError("Saved frames violate finite-volume continuity")
    diagnostics = np.asarray(result["diagnostics"])
    if diagnostics.ndim != 2 or diagnostics.shape[1] != len(DIAGNOSTIC_COLUMNS) or not np.isfinite(diagnostics).all():
        raise ValueError("Invalid actual-step diagnostics")
    if len(diagnostics) == 0 or not np.isclose(diagnostics[-1, 1], times[-1], rtol=0, atol=1e-11):
        raise ValueError("Actual steps did not reach the last recorded physical frame")
    if np.any(diagnostics[:, 2] <= 0) or not np.allclose(
            np.cumsum(diagnostics[:, 2]), diagnostics[:, 1], rtol=0, atol=1e-10):
        raise ValueError("Actual-step times and timestep sizes are inconsistent")
    if not np.array_equal(diagnostics[:, 0], np.arange(1, len(diagnostics) + 1)):
        raise ValueError("Actual-step identifiers must be consecutive positive integers")
    if np.any(diagnostics[:, 2] > config.dt + 1e-12):
        raise ValueError("An actual timestep exceeds the declared maximum dt")
    if any(not np.any(np.isclose(diagnostics[:, 1], time, rtol=0, atol=1e-10)) for time in times[1:]):
        raise ValueError("Every saved frame must match an actual timestep")
    return True


def frame_fields(result, index):
    """Compute cell-centered fields and nodal streamfunction from saved faces."""
    n = result["config"]["cells"]
    u, v = result["u_faces"][index], result["v_faces"][index]
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    fields = derived_fields(u, v, 1.0 / n)
    fields.update(x=(np.arange(n) + 0.5) / n, y=(np.arange(n) + 0.5) / n,
                  u=uc, v=vc, speed=np.hypot(uc, vc), p=result["p"][index])
    return fields


def save_trajectory(result, path):
    """Store compact native arrays and explicit metadata without pickle."""
    validate_trajectory(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: result[key] for key in ("times", "u_faces", "v_faces", "p", "diagnostics")}
    metadata = {key: value for key, value in result.items() if key not in arrays}
    metadata["schema"] = 1
    metadata["diagnostic_columns"] = list(DIAGNOSTIC_COLUMNS)
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        load_trajectory(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trajectory(path):
    with np.load(Path(path), allow_pickle=False) as data:
        result = json.loads(str(data["metadata_json"]))
        result.update({key: data[key].copy() for key in ("times", "u_faces", "v_faces", "p", "diagnostics")})
    if result.pop("schema", None) != 1 or result.pop("diagnostic_columns", None) != list(DIAGNOSTIC_COLUMNS):
        raise ValueError("Unknown startup data schema")
    validate_trajectory(result)
    return result


def compare_trajectories(candidate, reference):
    """Matched-time differences; fine-grid fields sampled at coarse centers.

    The reference is a numerical comparator, not truth. Cell-center sampling
    is used only for this explicitly reported grid-sensitivity calculation;
    rendering and saved scientific frames retain their native grids.
    """
    validate_trajectory(candidate)
    validate_trajectory(reference)
    if candidate["config"]["reynolds"] != reference["config"]["reynolds"]:
        raise ValueError("Sensitivity requires the same Reynolds number")
    if candidate["config"]["cells"] > reference["config"]["cells"]:
        raise ValueError("For grid sensitivity, supply the coarser case as candidate")
    if not np.array_equal(candidate["times"], reference["times"]):
        raise ValueError("Sensitivity requires identical physical observation times")
    rows = []
    for k, time in enumerate(candidate["times"]):
        a, b = frame_fields(candidate, k), frame_fields(reference, k)
        xx, yy = np.meshgrid(a["x"], a["y"])
        points = np.column_stack([yy.ravel(), xx.ravel()])
        sampled = {}
        for key in ("u", "v", "p", "omega"):
            sampled[key] = b[key] if a[key].shape == b[key].shape else RegularGridInterpolator(
                (b["y"], b["x"]), b[key], bounds_error=True)(points).reshape(a[key].shape)
        velocity_error = np.sqrt(np.mean((a["u"] - sampled["u"])**2 + (a["v"] - sampled["v"])**2))
        velocity_reference = np.sqrt(np.mean(sampled["u"]**2 + sampled["v"]**2))
        rows.append({"time": float(time), "velocity_rms_difference": float(velocity_error),
                     "velocity_relative_l2": float(velocity_error / velocity_reference) if velocity_reference else None,
                     "pressure_rms_difference": float(np.sqrt(np.mean((a["p"] - sampled["p"])**2))),
                     "vorticity_rms_difference": float(np.sqrt(np.mean((a["omega"] - sampled["omega"])**2))),
                     "kinetic_energy_difference": float(a["kinetic_energy"] - b["kinetic_energy"]),
                     "primary_psi_difference": float(a["psi_min"] - b["psi_min"])})
    return {"candidate": candidate["config"], "reference": reference["config"], "frames": rows,
            "meaning": "Observed matched-time numerical sensitivity, not an error bound or steady-state qualification"}


def case_summary(result):
    diagnostics = result["diagnostics"]
    return {"config": result["config"], "frames": len(result["times"]),
            "actual_steps": len(diagnostics), "end_time": float(result["times"][-1]),
            "runtime_seconds": result["runtime_seconds"],
            "max_divergence_linf": float(diagnostics[:, 3].max()),
            "max_time_step_momentum_defect_linf": float(diagnostics[:, 4].max()),
            "last_steady_momentum_defect_linf": float(diagnostics[-1, 5]),
            "max_pressure_mean_abs": float(np.abs(diagnostics[:, 6]).max()),
            "max_cfl": float(diagnostics[:, 9].max()),
            "qualification": result["qualification"]}


def common_field_bounds(results):
    """Full extrema over every stored frame and Reynolds number, without clipping."""
    results = list(results)
    if not results:
        raise ValueError("At least one trajectory is required")
    bounds = {name: [np.inf, -np.inf] for name in ("speed", "omega", "p")}
    for result in results:
        for k in range(len(result["times"])):
            fields = frame_fields(result, k)
            for key in bounds:
                bounds[key][0] = min(bounds[key][0], float(fields[key].min()))
                bounds[key][1] = max(bounds[key][1], float(fields[key].max()))
    return {key: {"data_min": lo, "data_max": hi,
                  "vmin": 0.0 if key == "speed" else -max(abs(lo), abs(hi), 1e-12),
                  "vmax": max(1.0, hi) if key == "speed" else max(abs(lo), abs(hi), 1e-12)}
            for key, (lo, hi) in bounds.items()}
