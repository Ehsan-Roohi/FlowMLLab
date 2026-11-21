"""Checkpointed physical-time continuation of verified cavity startup fields.

The original MAC momentum/coupling implementation is called without changes.
All criteria describe temporal stationarity of a fixed discrete configuration,
not spatial convergence, a Ghia validation, or stability of a three-dimensional
flow. Checkpoint state contains every window accumulator needed for a restart.
"""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import uuid

import numpy as np
import scipy

from .cavity_startup import load_trajectory
from .pressure_velocity import coupling_step, derived_fields, divergence, _residual
from .pressure_velocity_experiments import ExperimentalConfig


SCHEMA = "flowmllab.cavity.continuation.v1"
CORE_SOURCES = ("flowmllab/pressure_velocity.py", "flowmllab/pressure_velocity_experiments.py")
CONTINUATION_SOURCES = CORE_SOURCES + (
    "flowmllab/cavity_startup.py", "flowmllab/ghia.py",
    "flowmllab/cavity_startup_continuation.py", "qa/continue_cavity_startup.py",
)
METRICS = (
    "velocity_time_derivative_rms", "velocity_time_derivative_linf",
    "velocity_anchor_rms", "velocity_anchor_linf",
    "pressure_anchor_rms", "pressure_anchor_linf", "kinetic_energy_range",
    "psi_min_range", "psi_max_range", "steady_momentum_defect_linf",
    "time_step_momentum_defect_linf", "divergence_linf",
    "normal_wall_velocity_linf", "pressure_mean_abs",
    "velocity_window_mean_rms", "pressure_window_mean_rms",
)
PROBES = {"central": (.5, .5), "upper_left": (.05, .95),
          "lower_left": (.05, .05), "lower_right": (.95, .05)}
DIAGNOSTIC_COLUMNS = (
    "step", "time", "dt", "velocity_time_derivative_rms",
    "velocity_time_derivative_linf", "velocity_anchor_rms", "velocity_anchor_linf",
    "pressure_anchor_rms", "pressure_anchor_linf", "kinetic_energy", "psi_min", "psi_max",
    "steady_momentum_defect_linf", "time_step_momentum_defect_linf", "divergence_linf",
    "normal_wall_velocity_linf", "pressure_mean_abs", "advective_cfl", "pressure_solves",
) + tuple(f"{name}_{field}" for name in PROBES for field in ("u", "v", "p", "omega"))
ARRAY_NAMES = ("u_faces", "v_faces", "p", "anchor_u", "anchor_v", "anchor_p",
               "sum_u", "sum_v", "sum_p", "previous_mean_u", "previous_mean_v",
               "previous_mean_p", "diagnostics")
FIELD_GROUPS = (("u_faces", "v_faces", "p"), ("anchor_u", "anchor_v", "anchor_p"),
                ("previous_mean_u", "previous_mean_v", "previous_mean_p"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: sha256(root / name) for name in CONTINUATION_SOURCES}


def runtime_contract():
    """Restart admission, not a promise of cross-platform bitwise agreement."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "byteorder": sys.byteorder,
            "machine": platform.machine(), "system": platform.system(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def production_policy(reynolds):
    """Engineering gates frozen before continuation; not literature thresholds.

    The run driver requires a separately supplied policy JSON. This function
    exposes the agreed policy for review/freezing, never tunes it from results.
    Native MAC-component RMS uses all stored u/v degrees of freedom equally;
    L-infinity is the maximum absolute component, not a cell-interpolated norm.
    """
    if reynolds not in (100, 400, 1000, 3200, 5000, 7500, 10000):
        raise ValueError("Expected an established Ghia Reynolds number")
    thresholds = {key: 1e-7 for key in METRICS}
    for key in ("velocity_time_derivative_rms", "velocity_anchor_rms", "pressure_anchor_rms",
                "velocity_window_mean_rms", "pressure_window_mean_rms"):
        thresholds[key] = 1e-6
    for key in ("velocity_time_derivative_linf", "velocity_anchor_linf", "pressure_anchor_linf"):
        thresholds[key] = 1e-5
    thresholds.update(divergence_linf=1e-8, normal_wall_velocity_linf=1e-8, pressure_mean_abs=1e-10)
    return {"schema": SCHEMA, "reynolds": reynolds,
            "criterion_origin": "Engineering gates frozen on 2026-11-21 before continuation; not paper-quoted tolerances",
            "window_duration": 20. if reynolds <= 3200 else 50.,
            "consecutive_windows": 3, "diagnostic_interval": .05,
            "snapshot_interval": .5, "checkpoint_interval": 1.,
            "max_snapshots": 1000, "max_diagnostic_rows": 100000,
            "thresholds": thresholds}


def _ticks(duration, dt, name):
    if not (np.isfinite(duration) and duration >= 0 and np.isfinite(dt) and dt > 0):
        raise ValueError(f"{name} and dt must be finite, with positive dt")
    count = int(round(duration / dt))
    if not np.isclose(duration, count * dt, rtol=0, atol=1e-10 * max(1., duration)):
        raise ValueError(f"{name} must lie on the original fixed dt grid; no shortened-step trajectory changes")
    return count


def validate_policy(policy, config):
    expected = set(production_policy(config.reynolds))
    if set(policy) != expected or policy["schema"] != SCHEMA or policy["reynolds"] != config.reynolds:
        raise ValueError("Policy schema/fields/Re mismatch")
    if set(policy["thresholds"]) != set(METRICS):
        raise ValueError("Explicit thresholds for every stationarity metric are required")
    if any(not np.isfinite(value) or value <= 0 for value in policy["thresholds"].values()):
        raise ValueError("Thresholds must be finite positive numbers")
    for name in ("window_duration", "diagnostic_interval", "snapshot_interval", "checkpoint_interval"):
        if _ticks(policy[name], config.dt, name) < 1:
            raise ValueError(f"{name} must be at least one original dt")
    for name in ("consecutive_windows", "max_snapshots", "max_diagnostic_rows"):
        if type(policy[name]) is not int or policy[name] < (2 if name == "consecutive_windows" else 1):
            raise ValueError(f"Invalid integer policy field {name}")
    if not isinstance(policy["criterion_origin"], str) or not policy["criterion_origin"].strip():
        raise ValueError("Policy needs a documented criterion origin")
    canonical_json(policy)
    return True


def _admit_fields(u, v, p, n):
    for name, field, shape in (("u", u, (n, n + 1)), ("v", v, (n + 1, n)), ("p", p, (n, n))):
        if field.shape != shape or field.dtype != np.float64 or not np.isfinite(field).all():
            raise ValueError(f"Invalid native {name} field shape/dtype/values")
    mass = float(np.max(np.abs(divergence(u, v, 1. / n))))
    wall = float(max(np.max(np.abs(u[:, [0, -1]])), np.max(np.abs(v[[0, -1], :]))))
    gauge = float(abs(p.mean()))
    if mass > 1e-8 or wall > 1e-8 or gauge > 1e-10:
        raise ValueError(f"Native continuity/wall/gauge admission failed: {mass:g}, {wall:g}, {gauge:g}")
    return mass, wall, gauge


def _velocity_norm(u, v):
    return (float(np.sqrt((np.sum(u * u) + np.sum(v * v)) / (u.size + v.size))),
            float(max(np.max(np.abs(u)), np.max(np.abs(v)))))


def _scalar_values(u, v):
    fields = derived_fields(u, v, 1. / u.shape[0])
    return {"kinetic_energy": float(fields["kinetic_energy"]),
            "psi_min": float(fields["psi"].min()), "psi_max": float(fields["psi"].max())}, fields


def _new_window(state):
    values, _ = _scalar_values(state["u_faces"], state["v_faces"])
    for destination, source in zip(("anchor_u", "anchor_v", "anchor_p"), ("u_faces", "v_faces", "p")):
        state[destination] = state[source].copy()
    for destination, source in zip(("sum_u", "sum_v", "sum_p"), ("u_faces", "v_faces", "p")):
        state[destination] = np.zeros_like(state[source])
    window_ticks = _ticks(state["policy"]["window_duration"], state["config"]["dt"], "window_duration")
    absolute_tick = state["origin_tick"] + state["step"] - state["origin_step"]
    state["window"] = {"start_step": state["step"], "start_time": state["time"],
                       "partial_initial_window": absolute_tick % window_ticks != 0,
                       "samples": 0, "low": values.copy(), "high": values.copy(),
                       "maxima": {key: 0. for key in METRICS if not key.endswith("_range") and "window_mean" not in key}}


def initial_state(seed, receipt_path, policy):
    """Verify closed seed bytes, receipt, numerical source and native final state."""
    seed, receipt_path = Path(seed), Path(receipt_path)
    seed_stat, receipt_stat = seed.stat(), receipt_path.stat()
    receipt_hash = sha256(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = sha256(seed)
    if digest != receipt.get("trajectory_sha256") or receipt.get("trajectory") != seed.name:
        raise ValueError("Seed trajectory checksum/name disagrees with its receipt")
    source = source_hashes()
    if any(receipt.get("source_sha256", {}).get(name) != source[name] for name in CORE_SOURCES):
        raise ValueError("Seed numerical source hashes differ from the current unchanged MAC solver")
    case = load_trajectory(seed)
    config = ExperimentalConfig(**case["config"])
    if config.method not in ("piso", "pimple") or not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("Continuation requires finite positive fixed-dt PISO/PIMPLE physics")
    validate_policy(policy, config)
    if (case["config"] != receipt.get("config") or len(case["times"]) != receipt.get("frames")
            or len(case["diagnostics"]) != receipt.get("actual_steps")
            or float(case["times"][-1]) != receipt.get("end_time")):
        raise ValueError("Seed configuration/time/count metadata disagrees with receipt")
    if ((seed.stat().st_size, seed.stat().st_mtime_ns) != (seed_stat.st_size, seed_stat.st_mtime_ns)
            or (receipt_path.stat().st_size, receipt_path.stat().st_mtime_ns) != (receipt_stat.st_size, receipt_stat.st_mtime_ns)
            or sha256(receipt_path) != receipt_hash or sha256(seed) != digest):
        raise ValueError("Seed or receipt changed during admission; possible live writer")
    contract = {"seed_sha256": digest, "seed_receipt_sha256": receipt_hash,
                "seed_receipt": receipt, "continuation_source_sha256": source,
                "runtime": runtime_contract(), "policy_sha256": fingerprint(policy),
                "physics": "Unmodified coupling_step; native transient zero-area-mean pressure; no RNG; fixed original dt",
                "legacy_limits": "Seed config.end_time/max_iterations describe its original collector only; stage target is separate",
                "velocity_norm": "RMS over both native MAC-component DOF sets; Linf maximum absolute component",
                "window_means": "Arithmetic means of every completed physical step (fixed dt); no endpoint-only test",
                "qualification": "Fixed-grid temporal stationarity candidate only; tighter coupling/time/grid confirmation remains separate"}
    state = {"schema": SCHEMA, "config": asdict(config), "policy": policy,
             "contract": contract, "contract_sha256": fingerprint(contract),
             "origin_time": float(case["times"][-1]), "origin_step": len(case["diagnostics"]),
             "origin_tick": _ticks(float(case["times"][-1]), config.dt, "seed physical time"),
             "time": float(case["times"][-1]), "step": len(case["diagnostics"]),
             "pressure_solves": int(case["pressure_solves"]), "completed_windows": [],
             "consecutive_passes": 0, "candidate": False, "previous_mean_available": False,
             "snapshots": [], "last_diagnostics": None,
             "diagnostics": np.empty((0, len(DIAGNOSTIC_COLUMNS)), dtype=np.float64),
             "probe_locations": {name: {"requested": list(xy), "cell_index_yx": [min(config.cells - 1, int(xy[1] * config.cells)), min(config.cells - 1, int(xy[0] * config.cells))]}
                                 for name, xy in PROBES.items()}}
    for point in state["probe_locations"].values():
        j, i = point["cell_index_yx"]
        point["actual_xy"] = [(i + .5) / config.cells, (j + .5) / config.cells]
    for name in ("u_faces", "v_faces", "p"):
        state[name] = case[name][-1].copy()
    for name, current in zip(("previous_mean_u", "previous_mean_v", "previous_mean_p"), ("u_faces", "v_faces", "p")):
        state[name] = np.zeros_like(state[current])
    _new_window(state)
    validate_state(state)
    return state


def validate_state(state, expected_contract=None):
    """Admission for complete recovery, including unfinished-window accumulators."""
    if state.get("schema") != SCHEMA or fingerprint(state["contract"]) != state["contract_sha256"]:
        raise ValueError("Unknown restart schema or contract fingerprint mismatch")
    if expected_contract is not None and state["contract"] != expected_contract:
        raise ValueError("Restart seed/source/runtime/policy contract mismatch")
    config = ExperimentalConfig(**state["config"])
    validate_policy(state["policy"], config)
    if state["config"] != state["contract"]["seed_receipt"]["config"] or fingerprint(state["policy"]) != state["contract"]["policy_sha256"]:
        raise ValueError("Restart configuration/policy mismatch")
    receipt = state["contract"]["seed_receipt"]
    if state["origin_step"] != receipt["actual_steps"] or state["origin_time"] != receipt["end_time"]:
        raise ValueError("Restart origin differs from the verified source seed")
    if type(state["step"]) is not int or type(state["origin_step"]) is not int or state["step"] < state["origin_step"]:
        raise ValueError("Invalid absolute step counter")
    local = state["step"] - state["origin_step"]
    absolute_tick = state["origin_tick"] + local
    if state["origin_tick"] != _ticks(state["origin_time"], config.dt, "seed physical time"):
        raise ValueError("Seed absolute-time lattice mismatch")
    if not np.isclose(state["time"], state["origin_time"] + local * config.dt, rtol=0, atol=1e-10):
        raise ValueError("Absolute restart time/step mismatch")
    for names in FIELD_GROUPS:
        _admit_fields(*(state[name] for name in names), config.cells)
    for name, shape in (("sum_u", state["u_faces"].shape), ("sum_v", state["v_faces"].shape), ("sum_p", state["p"].shape)):
        if state[name].shape != shape or state[name].dtype != np.float64 or not np.isfinite(state[name]).all():
            raise ValueError("Invalid full-field window mean accumulator")
    window_ticks = _ticks(state["policy"]["window_duration"], config.dt, "window_duration")
    window = state["window"]
    window_start_tick = max(state["origin_tick"], (absolute_tick // window_ticks) * window_ticks)
    completed_count = absolute_tick // window_ticks - state["origin_tick"] // window_ticks
    if (window["start_step"] != state["origin_step"] + window_start_tick - state["origin_tick"]
            or window["samples"] != absolute_tick - window_start_tick
            or len(state["completed_windows"]) != completed_count
            or window["partial_initial_window"] != (window_start_tick % window_ticks != 0)):
        raise ValueError("Incomplete or inconsistent physical window recovery state")
    if not np.isclose(window["start_time"], state["origin_time"] + (window["start_step"] - state["origin_step"]) * config.dt, rtol=0, atol=1e-10):
        raise ValueError("Window anchor absolute time mismatch")
    maximum_keys = set(METRICS) - {"kinetic_energy_range", "psi_min_range", "psi_max_range", "velocity_window_mean_rms", "pressure_window_mean_rms"}
    if (set(window["maxima"]) != maximum_keys or any(value < 0 for value in window["maxima"].values())
            or set(window["low"]) != {"kinetic_energy", "psi_min", "psi_max"}
            or set(window["high"]) != set(window["low"])
            or any(window["low"][key] > window["high"][key] for key in window["low"])):
        raise ValueError("Invalid every-step window extrema/metrics")
    consecutive, previous_full = 0, False
    previous_end = state["origin_time"]
    for record in state["completed_windows"]:
        if set(record["metrics"]) != set(METRICS) or record["start_time"] != previous_end:
            raise ValueError("Completed window coverage/metric mismatch")
        expected_failed = _failed_metrics(record["metrics"], state["policy"])
        if record["partial_initial_window"]:
            expected_failed.append("partial_initial_window_excluded")
        consecutive = consecutive + 1 if not expected_failed else 0
        if (record["failed_metrics"] != expected_failed or record["passed"] != (not expected_failed)
                or record["consecutive_passes"] != consecutive
                or record["candidate"] != (consecutive >= state["policy"]["consecutive_windows"])
                or record["first_window_has_no_previous_mean"] != (not previous_full)):
            raise ValueError("Stored stationarity guard history disagrees with frozen thresholds")
        for name in ("velocity_window_mean_rms", "pressure_window_mean_rms"):
            if (record["metrics"][name] is None) != (not previous_full):
                raise ValueError("Missing or fabricated successive-window mean comparison")
        previous_full = not record["partial_initial_window"]
        previous_end = record["end_time"]
    expected_candidate = consecutive >= state["policy"]["consecutive_windows"] and not _active_window_failed(state)
    if (state["consecutive_passes"] != consecutive or state["candidate"] != expected_candidate
            or state["previous_mean_available"] != previous_full):
        raise ValueError("Restart candidate/consecutive-window status is inconsistent")
    diagnostics = state["diagnostics"]
    diagnostic_ticks = _ticks(state["policy"]["diagnostic_interval"], config.dt, "diagnostic_interval")
    diagnostic_count = absolute_tick // diagnostic_ticks - state["origin_tick"] // diagnostic_ticks
    if (diagnostics.shape != (diagnostic_count, len(DIAGNOSTIC_COLUMNS))
            or not np.isfinite(diagnostics).all()):
        raise ValueError("Incomplete or nonfinite compact diagnostic history")
    if len(diagnostics):
        first_tick = (state["origin_tick"] // diagnostic_ticks + 1) * diagnostic_ticks
        expected = state["origin_step"] + first_tick - state["origin_tick"] + np.arange(len(diagnostics)) * diagnostic_ticks
        if not np.array_equal(diagnostics[:, 0], expected) or not np.allclose(diagnostics[:, 1], state["origin_time"] + (expected - state["origin_step"]) * config.dt, rtol=0, atol=1e-10):
            raise ValueError("Diagnostic physical cadence/step mismatch")
        if not np.all(diagnostics[:, 2] == config.dt):
            raise ValueError("Diagnostic timestep differs from unchanged physical configuration")
    snapshot_ticks = _ticks(state["policy"]["snapshot_interval"], config.dt, "snapshot_interval")
    expected_snapshot_ticks = [state["origin_tick"]] + list(range((state["origin_tick"] // snapshot_ticks + 1) * snapshot_ticks, absolute_tick + 1, snapshot_ticks))
    # Initial in-memory admission occurs before its first snapshot is committed.
    if state["snapshots"]:
        if len(state["snapshots"]) != len(expected_snapshot_ticks):
            raise ValueError("Missing or extra native snapshot cadence records")
        for record, tick in zip(state["snapshots"], expected_snapshot_ticks):
            if record["step"] != state["origin_step"] + tick - state["origin_tick"] or not np.isclose(record["time"], tick * config.dt, rtol=0, atol=1e-10):
                raise ValueError("Native snapshot absolute-time cadence mismatch")
    if len(diagnostics) > state["policy"]["max_diagnostic_rows"] or len(state["snapshots"]) > state["policy"]["max_snapshots"]:
        raise ValueError("Recovery exceeds frozen storage caps")
    canonical_json({key: value for key, value in state.items() if key not in ARRAY_NAMES})
    return True


def _failed_metrics(metrics, policy):
    # JSON canonicalization sorts dictionary keys. A stable list order keeps
    # uninterrupted and resumed guard histories exactly identical.
    return sorted(key for key, value in metrics.items() if value is not None and value >= policy["thresholds"][key])


def _active_window_failed(state):
    metrics = state["window"]["maxima"].copy()
    for key in state["window"]["low"]:
        metrics[key + "_range"] = state["window"]["high"][key] - state["window"]["low"][key]
    return bool(_failed_metrics(metrics, state["policy"]))


def _finish_window(state):
    window = state["window"]
    metrics = window["maxima"].copy()
    for key in window["low"]:
        metrics[key + "_range"] = window["high"][key] - window["low"][key]
    means = [state[name] / window["samples"] for name in ("sum_u", "sum_v", "sum_p")]
    if state["previous_mean_available"]:
        metrics["velocity_window_mean_rms"] = _velocity_norm(means[0] - state["previous_mean_u"], means[1] - state["previous_mean_v"])[0]
        metrics["pressure_window_mean_rms"] = float(np.sqrt(np.mean((means[2] - state["previous_mean_p"]) ** 2)))
    else:
        metrics["velocity_window_mean_rms"] = metrics["pressure_window_mean_rms"] = None
    failed = _failed_metrics(metrics, state["policy"])
    if window["partial_initial_window"]:
        failed.append("partial_initial_window_excluded")
    passed = not failed
    state["consecutive_passes"] = state["consecutive_passes"] + 1 if passed else 0
    state["candidate"] = state["consecutive_passes"] >= state["policy"]["consecutive_windows"]
    state["completed_windows"].append({"start_time": window["start_time"], "end_time": state["time"],
                                       "actual_steps": window["samples"], "metrics": metrics,
                                       "first_window_has_no_previous_mean": not state["previous_mean_available"],
                                       "partial_initial_window": window["partial_initial_window"],
                                       "failed_metrics": failed, "passed": passed,
                                       "consecutive_passes": state["consecutive_passes"],
                                       "candidate": state["candidate"]})
    for name, value in zip(("previous_mean_u", "previous_mean_v", "previous_mean_p"), means):
        state[name] = value
    state["previous_mean_available"] = not window["partial_initial_window"]
    _new_window(state)


def advance_one_step(state):
    """Only commit a completed, admitted physical step to the in-memory state."""
    config = ExperimentalConfig(**state["config"])
    old_u, old_v = state["u_faces"], state["v_faces"]
    # coupling_step mutates its guesses; copies keep the preceding recovery intact.
    u, v, p, count = coupling_step(old_u.copy(), old_v.copy(), state["p"].copy(), config, dt=config.dt)
    mass, wall, gauge = _admit_fields(u, v, p, config.cells)
    values, fields = _scalar_values(u, v)
    rate_rms, rate_linf = _velocity_norm((u - old_u) / config.dt, (v - old_v) / config.dt)
    anchor_rms, anchor_linf = _velocity_norm(u - state["anchor_u"], v - state["anchor_v"])
    pressure_delta = p - state["anchor_p"]
    metrics = {"velocity_time_derivative_rms": rate_rms, "velocity_time_derivative_linf": rate_linf,
               "velocity_anchor_rms": anchor_rms, "velocity_anchor_linf": anchor_linf,
               "pressure_anchor_rms": float(np.sqrt(np.mean(pressure_delta ** 2))),
               "pressure_anchor_linf": float(np.max(np.abs(pressure_delta))),
               "steady_momentum_defect_linf": _residual(u, v, p, config),
               "time_step_momentum_defect_linf": _residual(u, v, p, config, old_u, old_v, config.dt),
               "divergence_linf": mass, "normal_wall_velocity_linf": wall, "pressure_mean_abs": gauge}
    if not np.isfinite(list(metrics.values())).all():
        raise FloatingPointError("Nonfinite step metrics; preceding checkpoint remains the recovery")
    state.update(u_faces=u, v_faces=v, p=p, step=state["step"] + 1, pressure_solves=state["pressure_solves"] + count)
    local = state["step"] - state["origin_step"]
    state["time"] = state["origin_time"] + local * config.dt
    window = state["window"]
    window["samples"] += 1
    for key, value in metrics.items():
        window["maxima"][key] = max(window["maxima"][key], value)
    for key, value in values.items():
        window["low"][key] = min(window["low"][key], value)
        window["high"][key] = max(window["high"][key], value)
    for key, value in zip(("sum_u", "sum_v", "sum_p"), (u, v, p)):
        state[key] += value
    if _active_window_failed(state):
        state["candidate"] = False
    data = {"step": state["step"], "time": state["time"], "dt": config.dt, **metrics, **values,
            "advective_cfl": config.dt * config.cells * (float(np.abs(u).max()) + float(np.abs(v).max())),
            "pressure_solves": state["pressure_solves"]}
    uc, vc = .5 * (u[:, :-1] + u[:, 1:]), .5 * (v[:-1, :] + v[1:, :])
    for name, point in state["probe_locations"].items():
        j, i = point["cell_index_yx"]
        data.update({f"{name}_{key}": float(array[j, i]) for key, array in (("u", uc), ("v", vc), ("p", p), ("omega", fields["omega"]))})
    state["last_diagnostics"] = [data[name] for name in DIAGNOSTIC_COLUMNS]
    absolute_tick = state["origin_tick"] + local
    if absolute_tick % _ticks(state["policy"]["diagnostic_interval"], config.dt, "diagnostic_interval") == 0:
        state["diagnostics"] = np.vstack([state["diagnostics"], state["last_diagnostics"]])
    if absolute_tick % _ticks(state["policy"]["window_duration"], config.dt, "window_duration") == 0:
        _finish_window(state)


def _fsync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_npz(path, arrays, metadata):
    """Write a new immutable payload; never replace an existing scientific file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable scientific payload: {path}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays, metadata_json=np.asarray(canonical_json(metadata)))
            stream.flush(); os.fsync(stream.fileno())
        read_arrays, read_metadata = _load_npz(temporary)
        if read_metadata != metadata or any(not np.array_equal(read_arrays[key], value) for key, value in arrays.items()):
            raise ValueError("Written payload failed read-back verification")
        # The exclusive case lock prevents competing writers between check/rename.
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        arrays = {name: data[name].copy() for name in data.files if name != "metadata_json"}
    return arrays, metadata


def _safe_member(directory, relative):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()) or Path(relative).is_absolute():
        raise ValueError("Manifest member escapes its case directory")
    return path


def load_checkpoint(directory, record, expected_contract=None):
    path = _safe_member(directory, record["file"])
    if sha256(path) != record["sha256"]:
        raise ValueError("Checkpoint checksum mismatch; no automatic fallback or overwrite")
    arrays, metadata = _load_npz(path)
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError("Checkpoint does not contain every native recovery/accumulator array")
    state = {**metadata, **arrays}
    validate_state(state, expected_contract)
    if state["step"] != record["step"] or state["time"] != record["time"]:
        raise ValueError("Checkpoint manifest time/step mismatch")
    return state


def _snapshot_metadata(state):
    return {"schema": SCHEMA + ".snapshot", "contract_sha256": state["contract_sha256"],
            "config": state["config"], "step": state["step"], "time": state["time"],
            "pressure_gauge": "native cell-area mean is zero",
            "field_location": "u/v native MAC faces, p cell centers; no temporal interpolation"}


def save_snapshot(directory, state):
    relative = f"snapshots/frame_{state['step']:012d}.npz"
    path = _safe_member(directory, relative)
    path.parent.mkdir(exist_ok=True)
    arrays = {key: state[key] for key in ("u_faces", "v_faces", "p")}
    metadata = _snapshot_metadata(state)
    if path.exists():
        # A killed writer may have saved this frame before committing restart.json.
        # Deterministic replay may reuse it only after exact scientific verification.
        retained, retained_metadata = _load_npz(path)
        if retained_metadata != metadata or set(retained) != set(arrays) or any(not np.array_equal(retained[key], arrays[key]) for key in arrays):
            raise ValueError("Existing snapshot disagrees with deterministic replay; preserving both recovery and evidence")
    else:
        _save_npz(path, arrays, metadata)
    record = {"file": relative, "sha256": sha256(path), "step": state["step"], "time": state["time"]}
    if not any(row["step"] == state["step"] for row in state["snapshots"]):
        state["snapshots"].append(record)
    return record


def verify_snapshots(directory, state):
    for record in state["snapshots"]:
        path = _safe_member(directory, record["file"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Retained native snapshot checksum mismatch: {record['file']}")
        arrays, metadata = _load_npz(path)
        if metadata["contract_sha256"] != state["contract_sha256"] or metadata["step"] != record["step"] or metadata["time"] != record["time"]:
            raise ValueError("Snapshot contract/time mismatch")
        _admit_fields(arrays["u_faces"], arrays["v_faces"], arrays["p"], state["config"]["cells"])


def commit_checkpoint(directory, state):
    """Keep two verified generations; remove only the superseded closed restart.

    Immutable native snapshots are never deleted. A crash before the atomic
    manifest replacement leaves the previous current/previous pair usable.
    Deletion is only after a verified replacement, and records actual bytes.
    """
    directory = Path(directory)
    validate_state(state)
    manifest_path = directory / "restart.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if old:
        load_checkpoint(directory, old["current"], state["contract"])
        if old.get("previous"):
            load_checkpoint(directory, old["previous"], state["contract"])
        old = _finish_pending_retirement(directory, old, state["contract"])
        if old["current"]["step"] == state["step"]:
            # Idempotent stage-end checkpoint; do not rotate away older recovery.
            current = load_checkpoint(directory, old["current"], state["contract"])
            if canonical_json({k: v for k, v in current.items() if k not in ARRAY_NAMES}) != canonical_json({k: v for k, v in state.items() if k not in ARRAY_NAMES}) or any(not np.array_equal(current[k], state[k]) for k in ARRAY_NAMES):
                raise ValueError("Same-step checkpoint differs from retained complete recovery")
            return old
        if old["current"]["step"] > state["step"]:
            raise ValueError("Refusing to overwrite a later recovery with an older step")
    restart_dir = directory / "restarts"
    restart_dir.mkdir(exist_ok=True)
    relative = f"restarts/checkpoint_{state['step']:012d}_{uuid.uuid4().hex}.npz"
    path = _safe_member(directory, relative)
    _save_npz(path, {key: state[key] for key in ARRAY_NAMES}, {key: value for key, value in state.items() if key not in ARRAY_NAMES})
    record = {"file": relative, "sha256": sha256(path), "step": state["step"], "time": state["time"]}
    load_checkpoint(directory, record, state["contract"])
    deletion_log = [] if old is None else old.get("actual_restart_deletions", [])
    obsolete = old.get("previous") if old else None
    manifest = {"schema": SCHEMA, "contract_sha256": state["contract_sha256"],
                "current": record, "previous": old["current"] if old else None,
                "pending_superseded_restart": obsolete, "actual_restart_deletions": deletion_log,
                "qualification": state["contract"]["qualification"]}
    atomic_json(manifest_path, manifest)
    confirmed = json.loads(manifest_path.read_text(encoding="utf-8"))
    load_checkpoint(directory, confirmed["current"], state["contract"])
    if confirmed["previous"]:
        load_checkpoint(directory, confirmed["previous"], state["contract"])
    manifest = _finish_pending_retirement(directory, manifest, state["contract"])
    manifest = _retire_closed_orphans(directory, manifest, state["contract"])
    return manifest


def _finish_pending_retirement(directory, manifest, contract):
    """Recover a kill between verified manifest commit and retirement receipt."""
    obsolete = manifest.get("pending_superseded_restart")
    if not obsolete:
        return manifest
    for key in ("current", "previous"):
        if manifest.get(key):
            load_checkpoint(directory, manifest[key], contract)
            if manifest[key]["file"] == obsolete["file"]:
                raise ValueError("Retirement points at retained recovery; refusing deletion")
    path = _safe_member(directory, obsolete["file"])
    if path.exists():
        if sha256(path) != obsolete["sha256"]:
            raise ValueError("Superseded restart changed; retained without deletion")
        size = path.stat().st_size
        path.unlink()
        if path.exists():
            raise OSError("Superseded checkpoint deletion did not complete")
        manifest["actual_restart_deletions"].append({"file": obsolete["file"], "bytes": size,
                                                    "replaced_at_step": manifest["current"]["step"]})
    else:
        # A kill may occur after unlink but before this small receipt is written.
        # Do not invent freed-byte evidence for a deletion this process did not see.
        manifest.setdefault("retirement_recovery_notes", []).append({"file": obsolete["file"],
            "status": "already absent at resume; no additional removed bytes claimed"})
    manifest["pending_superseded_restart"] = None
    atomic_json(Path(directory) / "restart.json", manifest)
    return manifest


def _retire_closed_orphans(directory, manifest, contract):
    """Retire a complete uncommitted restart only after a newer verified pair.

    An atomic-manifest failure can leave a new immutable checkpoint outside the
    old manifest. Keep it while it is newer than either retained recovery. Once
    deterministic continuation has passed it and both recoveries are verified,
    its full matching contract/state identifies it as a superseded restart.
    Unknown files and incomplete .part payloads are not inferred expendable.
    """
    if not manifest.get("previous"):
        return manifest
    for key in ("current", "previous"):
        load_checkpoint(directory, manifest[key], contract)
    retained = {manifest[key]["file"] for key in ("current", "previous")}
    for path in sorted((Path(directory) / "restarts").glob("checkpoint_*.npz")):
        relative = path.relative_to(directory).as_posix()
        if relative in retained:
            continue
        arrays, metadata = _load_npz(path)
        if set(arrays) != set(ARRAY_NAMES):
            raise ValueError("Unknown orphan restart payload; preserved for inspection")
        orphan = {**metadata, **arrays}
        validate_state(orphan, contract)
        if orphan["step"] >= manifest["previous"]["step"]:
            continue
        size, digest = path.stat().st_size, sha256(path)
        path.unlink()
        if path.exists():
            raise OSError("Superseded orphan deletion did not complete")
        manifest["actual_restart_deletions"].append({"file": relative, "bytes": size,
            "sha256": digest, "replaced_at_step": manifest["current"]["step"], "kind": "verified superseded uncommitted recovery"})
    atomic_json(Path(directory) / "restart.json", manifest)
    return manifest


@contextmanager
def writer_lock(directory):
    """Cross-process exclusive guard; never guess whether a remote owner died."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "writer.lock"
    token = uuid.uuid4().hex
    owner = {"token": token, "host": socket.gethostname(), "pid": os.getpid(),
             "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "schema": SCHEMA}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Case has an existing writer lock: {path}; verify scheduler/owner before explicit stale-lock recovery") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(owner)); stream.flush(); os.fsync(stream.fileno())
        yield owner
    finally:
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("token") == token:
            path.unlink()


def report(state, reason):
    return {"schema": SCHEMA, "time": state["time"], "step": state["step"], "reason": reason,
            "candidate": state["candidate"], "consecutive_passes": state["consecutive_passes"],
            "completed_windows": len(state["completed_windows"]),
            "unfinished_window_steps": state["window"]["samples"],
            "last_complete_window": state["completed_windows"][-1] if state["completed_windows"] else None,
            "last_diagnostics": dict(zip(DIAGNOSTIC_COLUMNS, state["last_diagnostics"])) if state["last_diagnostics"] else None,
            "diagnostic_rows": len(state["diagnostics"]), "native_snapshots": len(state["snapshots"]),
            "contract_sha256": state["contract_sha256"], "policy": state["policy"],
            "config": state["config"], "qualification": state["contract"]["qualification"]}


def export_compact_history(directory, state, reason):
    """Convenient current views; the verified checkpoint is authoritative."""
    directory = Path(directory)
    atomic_json(directory / "status.json", report(state, reason))
    atomic_json(directory / "windows.json", {"contract_sha256": state["contract_sha256"], "windows": state["completed_windows"]})
    temporary = directory / ("diagnostics." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(stream, state["diagnostics"], delimiter=",", header=",".join(DIAGNOSTIC_COLUMNS), comments="", fmt="%.17g")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, directory / "diagnostics.csv")
    finally:
        if temporary.exists():
            temporary.unlink()


def continue_case(seed, seed_receipt, policy, output, target_time, stop_requested=None,
                  stop_on_candidate=False, progress=False):
    """Advance to an absolute physical stage target, or safely checkpoint a stop.

    Targets/cadences must lie on the original dt grid. No duplicate trajectory
    is launched if a case lock exists. Policy/runtime/source changes and bad
    current recovery checksums are refusals, never silent fresh starts.
    """
    output = Path(output)
    with writer_lock(output):
        initial = initial_state(seed, seed_receipt, policy)
        config = ExperimentalConfig(**initial["config"])
        target_ticks = _ticks(float(target_time) - initial["origin_time"], config.dt, "target_time minus seed time")
        target_step = initial["origin_step"] + target_ticks
        manifest_path = output / "restart.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state = load_checkpoint(output, manifest["current"], initial["contract"])
            if manifest.get("previous"):
                load_checkpoint(output, manifest["previous"], initial["contract"])
            verify_snapshots(output, state)
            _finish_pending_retirement(output, manifest, initial["contract"])
        else:
            unexpected = [path.name for path in output.iterdir() if path.name != "writer.lock"]
            if unexpected:
                raise FileExistsError("Output has no committed recovery but is not empty; inspect instead of overwriting: " + ", ".join(unexpected))
            state = initial
            save_snapshot(output, state)
            commit_checkpoint(output, state)
        if target_step < state["step"]:
            raise ValueError("Absolute target precedes the retained recovery")
        cadence = {name: _ticks(policy[name + "_interval"], config.dt, name + "_interval") for name in ("diagnostic", "snapshot", "checkpoint")}
        reason = "target_reached"
        while state["step"] < target_step:
            if stop_requested is not None and stop_requested():
                reason = "stop_requested_after_complete_step"; break
            if stop_on_candidate and state["candidate"]:
                reason = "fixed_grid_stationarity_candidate"; break
            next_tick = state["origin_tick"] + state["step"] - state["origin_step"] + 1
            if (next_tick % cadence["snapshot"] == 0 and len(state["snapshots"]) >= policy["max_snapshots"]
                    or next_tick % cadence["diagnostic"] == 0 and len(state["diagnostics"]) >= policy["max_diagnostic_rows"]):
                reason = "frozen_storage_cap_reached_no_scientific_data_deleted"; break
            advance_one_step(state)
            if next_tick % cadence["snapshot"] == 0:
                save_snapshot(output, state)
            if next_tick % cadence["checkpoint"] == 0:
                commit_checkpoint(output, state)
                export_compact_history(output, state, "running")
                if progress:
                    print(json.dumps({"Re": config.reynolds, "time": state["time"], "candidate": state["candidate"], "windows": len(state["completed_windows"])}), flush=True)
        commit_checkpoint(output, state)
        export_compact_history(output, state, reason)
        return report(state, reason)
