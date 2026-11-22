"""Read-only, exact-time analysis of the frozen cavity continuation campaign.

Run on Unity or a verified local mirror; no solver, writer lock or checkpoint
mutation is used. Only atomic-manifest-committed recoveries and their immutable
native snapshots are admitted. Missing cases are explicit partial coverage.

Example::

    python qa/analyze_cavity_settling.py --campaign campaign.json \
        --data-root /path/to/campaign --seed-directory /path/to/fine128 \
        --output analysis/current.json

Fine fields are bilinearly sampled at coarse cell centers solely for spatial
difference norms, matching cavity_startup.compare_trajectories. Profiles retain
each native grid's cell-center coverage and interpolate only to x/y=0.5.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from zipfile import BadZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from flowmllab import cavity_startup as startup
from flowmllab import cavity_startup_continuation as continuation

SCHEMA = "flowmllab.cavity.settling.analysis.v1"
MEANING = "Observed matched-time numerical sensitivity; the reference is a numerical comparator, not truth or an error bound."
QUALIFICATION = "Frozen windows identify fixed-grid temporal stationarity candidates only; time, coupling and mesh confirmation remain separate. No steady Ghia, 3D stability or turbulence qualification is inferred."
HISTORY_COLUMNS = ["time", "kinetic_energy", "psi_min", "psi_max", "steady_momentum_defect_linf"] + [
    f"{name}_{component}" for name in continuation.PROBES for component in ("u", "v", "p", "omega")]
DIAGNOSTIC_COLUMNS = ["time", "steady_momentum_defect_linf", "time_step_momentum_defect_linf",
                      "velocity_time_derivative_rms", "velocity_time_derivative_linf",
                      "divergence_linf", "advective_cfl", "kinetic_energy", "psi_min", "psi_max"] + [
    f"{name}_{component}" for name in continuation.PROBES for component in ("u", "v", "p", "omega")]
EVIDENCE_ERRORS = (OSError, ValueError, KeyError, AssertionError, BadZipFile)


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_verified(path, expected=None):
    payload = Path(path).read_bytes()
    if expected is not None and sha_bytes(payload) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return payload


def member(directory, relative):
    directory = Path(directory).resolve()
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory):
        raise ValueError(f"Manifest member leaves its directory: {relative}")
    return path


def time_key(time, dt):
    """Exact physical-step-lattice key; only binary-float serialization is tolerated."""
    if not np.isfinite(time) or time < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid physical time/dt")
    tick = round(float(time) / float(dt))
    key = tick * Fraction(str(dt))
    if not np.isclose(float(key), time, rtol=0, atol=1e-10 * max(1., float(time))):
        raise ValueError("Frame is not on its unchanged physical timestep lattice")
    return key


def unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    return arrays, metadata


@lru_cache(maxsize=2)
def read_seed(path, expected_sha):
    arrays, metadata = unpack(read_verified(path, expected_sha))
    if metadata.pop("schema", None) != 1 or metadata.pop("diagnostic_columns", None) != list(startup.DIAGNOSTIC_COLUMNS):
        raise ValueError("Unknown seed trajectory schema")
    result = {**metadata, **arrays}
    startup.validate_trajectory(result)
    return result


def expected_config(case, config):
    expected = {"cells": case["cells"], "reynolds": case["reynolds"], "dt": case["dt"],
                "outer_correctors": case["outer"], "pressure_correctors": case["pressure"],
                "method": "pimple", "convection": "central"}
    if any(config.get(name) != value for name, value in expected.items()):
        raise ValueError("Actual numerical configuration differs from frozen campaign case")


@dataclass
class Frame:
    time: float
    config: dict
    path: Path | None = None
    digest: str | None = None
    seed_index: int | None = None
    contract: str | None = None
    step: int | None = None
    arrays: dict | None = None

    def native(self):
        if self.arrays is not None:
            values = self.arrays
        elif self.seed_index is not None:
            seed = read_seed(str(self.path), self.digest)
            values = {key: seed[key][self.seed_index] for key in ("u_faces", "v_faces", "p")}
        else:
            values, metadata = unpack(read_verified(self.path, self.digest))
            if (set(values) != {"u_faces", "v_faces", "p"}
                    or metadata.get("schema") != continuation.SCHEMA + ".snapshot"
                    or metadata.get("contract_sha256") != self.contract
                    or metadata.get("config") != self.config or metadata.get("step") != self.step
                    or metadata.get("time") != self.time):
                raise ValueError(f"Native snapshot metadata mismatch: {self.path}")
        continuation._admit_fields(values["u_faces"], values["v_faces"], values["p"], self.config["cells"])
        return values

    def fields(self):
        values = self.native()
        return startup.frame_fields({"config": self.config,
            **{key: value[None, ...] for key, value in values.items()}}, 0)

    def provenance(self):
        return {"time": self.time, "file": str(self.path) if self.path is not None else None,
                "sha256": self.digest, "seed_index": self.seed_index,
                "step": self.step, "contract_sha256": self.contract,
                "origin": "committed_checkpoint" if self.arrays is not None else (
                    "original_rest_start_prefix" if self.seed_index is not None else "immutable_snapshot")}


@dataclass
class Series:
    case: dict
    config: dict | None = None
    frames: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def add_frame(self, frame):
        key = time_key(frame.time, frame.config["dt"])
        if key in self.frames:
            first, second = self.frames[key].native(), frame.native()
            if any(not np.array_equal(first[name], second[name]) for name in first):
                raise ValueError("Conflicting native fields at one physical time; refusing to choose a version")
        else:
            self.frames[key] = frame


def admit_sources(campaign):
    expected = campaign["source_sha256"]
    actual = {}
    for name in continuation.CONTINUATION_SOURCES:
        if name not in expected:
            raise ValueError(f"Missing frozen numerical-source identity: {name}")
        read_verified(ROOT / name, expected[name])
        actual[name] = expected[name]
    return actual


def checkpoint_state(directory, record):
    arrays, metadata = unpack(read_verified(member(directory, record["file"]), record["sha256"]))
    if set(arrays) != set(continuation.ARRAY_NAMES):
        raise ValueError("Incomplete checkpoint arrays")
    state = {**metadata, **arrays}
    continuation.validate_state(state)
    if state["step"] != record["step"] or state["time"] != record["time"]:
        raise ValueError("Checkpoint time disagrees with atomic manifest")
    return state


def admit_checkpoint(directory, case, campaign, policy_path):
    """Short stable-manifest read. Later immutable snapshots are independent of writer progress."""
    manifest_path = directory / "restart.json"
    for attempt in range(3):
        before = manifest_path.read_bytes()
        try:
            manifest = json.loads(before)
            state = checkpoint_state(directory, manifest["current"])
            if manifest.get("previous"):
                previous = checkpoint_state(directory, manifest["previous"])
                if previous["contract"] != state["contract"] or previous["step"] >= state["step"]:
                    raise ValueError("Previous/current recovery pair mismatch")
            if before != manifest_path.read_bytes():
                continue
        except EVIDENCE_ERRORS:
            if before != manifest_path.read_bytes():
                continue
            raise
        expected_config(case, state["config"])
        contract = state["contract"]
        if manifest.get("schema") != continuation.SCHEMA or manifest.get("contract_sha256") != state["contract_sha256"]:
            raise ValueError("Manifest contract identity mismatch")
        if contract["continuation_source_sha256"] != {name: campaign["source_sha256"][name] for name in continuation.CONTINUATION_SOURCES}:
            raise ValueError("Checkpoint numerical source contract differs from frozen campaign")
        policy = json.loads(read_verified(policy_path, case["policy_file_sha256"]))
        if state["policy"] != policy:
            raise ValueError("Checkpoint policy differs from frozen campaign policy")
        if case["seed_mode"] == "existing_native_t10":
            if (contract["seed_sha256"] != case["seed_sha256"]
                    or contract["seed_receipt_sha256"] != case["seed_receipt_sha256"]
                    or state["origin_time"] != 10.):
                raise ValueError("Baseline recovery is not the frozen native t=10 seed")
        elif case["seed_mode"] == "fresh_rest_start":
            if state["origin_time"] != .1:
                raise ValueError("Fresh control does not start from the declared rest-start seed")
        else:
            raise ValueError("Unknown seed mode")
        return state, {"restart_manifest_sha256": sha_bytes(before), "current_checkpoint": manifest["current"],
                       "previous_checkpoint": manifest.get("previous"), "writer_lock_seen": (directory / "writer.lock").exists()}
    raise RuntimeError("Atomic manifest changed during all three admission attempts; retry later")


def seed_paths(case, case_root, seed_directory):
    if case["seed_mode"] == "fresh_rest_start":
        path = case_root / "seed" / f"re{case['reynolds']}.npz"
        return path, path.with_suffix(".json")
    if seed_directory is not None:
        path = Path(seed_directory) / f"re{case['reynolds']}.npz"
        return path, path.with_suffix(".json")
    remote = Path(case["seed_path"])
    if remote.is_file():
        return remote, Path(case["seed_receipt"])
    path = ROOT / "results/cavity_startup/fine128" / f"re{case['reynolds']}.npz"
    return path, path.with_suffix(".json")


def add_seed(series, seed_path, receipt_path, campaign, state=None):
    if not seed_path.exists() or not receipt_path.exists():
        series.summary["seed_prefix"] = {"status": "unavailable", "reason": "Native prefix/receipt not present; no earlier frames fabricated"}
        return
    case = series.case
    receipt_bytes = read_verified(receipt_path, case.get("seed_receipt_sha256"))
    receipt = json.loads(receipt_bytes)
    if receipt.get("trajectory") != seed_path.name:
        raise ValueError("Seed receipt names another trajectory")
    if case.get("seed_sha256") not in (None, receipt["trajectory_sha256"]):
        raise ValueError("Baseline prefix differs from campaign seed")
    for name in continuation.CORE_SOURCES:
        if receipt["source_sha256"].get(name) != campaign["source_sha256"][name]:
            raise ValueError("Seed prefix used different MAC kernels")
    # A later analysis may reuse decoded immutable data, but must freshly admit
    # the file identity even when this interpreter has a populated decode cache.
    read_verified(seed_path, receipt["trajectory_sha256"])
    seed = read_seed(str(seed_path), receipt["trajectory_sha256"])
    expected_config(case, seed["config"])
    if (receipt.get("config") != seed["config"] or receipt.get("frames") != len(seed["times"])
            or receipt.get("actual_steps") != len(seed["diagnostics"])
            or receipt.get("end_time") != float(seed["times"][-1])):
        raise ValueError("Seed receipt configuration/time/count mismatch")
    if case["seed_mode"] == "existing_native_t10" and seed["times"][-1] != 10.:
        raise ValueError("Expected the original t=0..10 baseline")
    if case["seed_mode"] == "fresh_rest_start" and seed["times"][-1] != .1:
        raise ValueError("Expected the fresh t=0..0.1 control seed")
    if state is not None and (state["contract"]["seed_sha256"] != receipt["trajectory_sha256"]
                              or state["contract"]["seed_receipt_sha256"] != sha_bytes(receipt_bytes)):
        raise ValueError("Prefix and continuation have different seed identities")
    series.config = seed["config"]
    for index, time in enumerate(seed["times"]):
        series.add_frame(Frame(float(time), seed["config"], seed_path, receipt["trajectory_sha256"], index,
                               step=round(float(time) / seed["config"]["dt"])))
    for row in seed["diagnostics"]:
        values = dict(zip(startup.DIAGNOSTIC_COLUMNS, map(float, row)))
        series.diagnostics[time_key(values["time"], seed["config"]["dt"])] = values
    series.summary["seed_prefix"] = {"status": "sha_verified", "trajectory_sha256": receipt["trajectory_sha256"],
                                    "receipt_sha256": sha_bytes(receipt_bytes), "frames": len(seed["times"])}


def load_case(case, campaign, data_root, seed_directory, policy_root):
    if Path(case["id"]).name != case["id"]:
        raise ValueError("Case ID must be a plain directory name")
    series = Series(case, summary={"case": case, "admission": "missing", "qualification": QUALIFICATION})
    case_root = member(data_root / "runs", case["id"])
    directory = case_root / "continuation"
    state = None
    if (directory / "restart.json").is_file():
        state, proof = admit_checkpoint(directory, case, campaign, policy_root / f"re{case['reynolds']}.json")
        series.config = state["config"]
        series.summary.update(admission="verified_checkpoint", provenance=proof,
            stationarity={"candidate": state["candidate"], "consecutive_passes": state["consecutive_passes"],
                "completed_windows": state["completed_windows"],
                "last_complete_window": state["completed_windows"][-1] if state["completed_windows"] else None,
                "active_window": state["window"], "policy": state["policy"],
                "qualification": state["contract"]["qualification"]})
    add_seed(series, *seed_paths(case, case_root, seed_directory), campaign, state)
    if state is not None:
        for record in state["snapshots"]:
            series.add_frame(Frame(record["time"], state["config"], member(directory, record["file"]),
                record["sha256"], contract=state["contract_sha256"], step=record["step"]))
        # A complete checkpoint can supply an actual final frame between movie snapshots.
        current = proof["current_checkpoint"]
        series.add_frame(Frame(state["time"], state["config"], member(directory, current["file"]),
            current["sha256"], contract=state["contract_sha256"], step=state["step"],
            arrays={key: state[key] for key in ("u_faces", "v_faces", "p")}))
        for row in state["diagnostics"]:
            values = dict(zip(continuation.DIAGNOSTIC_COLUMNS, map(float, row)))
            series.diagnostics[time_key(values["time"], state["config"]["dt"])] = values
        if state["last_diagnostics"] is not None:
            values = dict(zip(continuation.DIAGNOSTIC_COLUMNS, state["last_diagnostics"]))
            series.diagnostics[time_key(values["time"], state["config"]["dt"])] = values
    elif series.frames:
        series.summary["admission"] = "seed_only"
    return series


def compare_fields(a, b):
    """Same cell-centered norm and spatial sampling as compare_trajectories."""
    xx, yy = np.meshgrid(a["x"], a["y"])
    points = np.column_stack([yy.ravel(), xx.ravel()])
    sampled = {name: b[name] if a[name].shape == b[name].shape else RegularGridInterpolator(
        (b["y"], b["x"]), b[name], bounds_error=True)(points).reshape(a[name].shape)
        for name in ("u", "v", "p", "omega")}
    error = float(np.sqrt(np.mean((a["u"] - sampled["u"]) ** 2 + (a["v"] - sampled["v"]) ** 2)))
    norm = float(np.sqrt(np.mean(sampled["u"] ** 2 + sampled["v"] ** 2)))
    return {"velocity_rms_difference": error, "velocity_relative_l2": error / norm if norm else None,
            "pressure_rms_difference": float(np.sqrt(np.mean((a["p"] - sampled["p"]) ** 2))),
            "vorticity_rms_difference": float(np.sqrt(np.mean((a["omega"] - sampled["omega"]) ** 2))),
            "kinetic_energy_difference": float(a["kinetic_energy"] - b["kinetic_energy"]),
            "primary_psi_difference": float(a["psi_min"] - b["psi_min"])}


def profiles(fields, time):
    x, y = fields["x"], fields["y"]
    vertical, horizontal = {"y": y.tolist()}, {"x": x.tolist()}
    for name in ("u", "v", "p", "omega"):
        interpolator = RegularGridInterpolator((y, x), fields[name], bounds_error=True)
        vertical[name] = interpolator(np.column_stack([y, np.full(len(y), .5)])).tolist()
        horizontal[name] = interpolator(np.column_stack([np.full(len(x), .5), x])).tolist()
    return {"time": time, "vertical": vertical, "horizontal": horizontal}


def plan_pairs(cases):
    def locate(re, n, dt, outer=3, pressure=2):
        found = [c["id"] for c in cases if (c["reynolds"], c["cells"], c["dt"], c["outer"], c["pressure"]) == (re, n, dt, outer, pressure)]
        if len(found) > 1:
            raise ValueError("Ambiguous duplicate numerical configuration in campaign")
        return found[0] if found else None
    pairs = []
    for re in (1000, 7500, 10000):
        for first, second in ((.01, .005), (.005, .0025), (.01, .0025)):
            pairs.append({"id": f"time_Re{re}_{first:g}_vs_{second:g}", "kind": "time",
                          "candidate_id": locate(re, 128, first), "reference_id": locate(re, 128, second)})
    for re in (3200, 5000, 7500, 10000):
        pairs.append({"id": f"grid_Re{re}_128_vs_256", "kind": "grid",
                      "candidate_id": locate(re, 128, .0025), "reference_id": locate(re, 256, .0025)})
    for re in (1000, 7500, 10000):
        pairs.append({"id": f"coupling_Re{re}_3x2_vs_6x4", "kind": "coupling",
                      "candidate_id": locate(re, 128, .01), "reference_id": locate(re, 128, .01, 6, 4)})
    return pairs


def check_pair(kind, candidate, reference):
    allowed = {"time": {"dt"}, "grid": {"cells"}, "coupling": {"outer_correctors", "pressure_correctors"}}[kind]
    for name in ("reynolds", "method", "convection", "cells", "dt", "outer_correctors", "pressure_correctors", "velocity_relaxation", "pressure_relaxation"):
        if name not in allowed and candidate[name] != reference[name]:
            raise ValueError(f"Unmatched {kind} control: {name}")
    if candidate["cells"] > reference["cells"]:
        raise ValueError("Grid comparison must sample the finer reference onto the coarse candidate")


def describe_series(series, profile_keys, history_stride):
    result = dict(series.summary)
    result["config"] = series.config
    result["profiles"] = []
    rows = []
    keys = sorted(series.frames)
    for key in keys:
        values = series.frames[key].fields()  # SHA and native-shape/gauge/continuity admission for every frame.
        diagnostic = series.diagnostics.get(key, {})
        record = {"time": float(key), "kinetic_energy": float(values["kinetic_energy"]),
                  "psi_min": float(values["psi"].min()), "psi_max": float(values["psi"].max()),
                  "steady_momentum_defect_linf": diagnostic.get("steady_momentum_defect_linf")}
        n = series.config["cells"]
        for name, (x, y) in continuation.PROBES.items():
            j, i = min(n - 1, int(y * n)), min(n - 1, int(x * n))
            for component in ("u", "v", "p", "omega"):
                record[f"{name}_{component}"] = float(values[component][j, i])
        rows.append([record[name] for name in HISTORY_COLUMNS])
        if key in profile_keys:
            result["profiles"].append(profiles(values, float(key)))
    diagnostic_keys = sorted(series.diagnostics)
    selected = diagnostic_keys[::history_stride]
    if diagnostic_keys and diagnostic_keys[-1] not in selected:
        selected.append(diagnostic_keys[-1])
    result["frame_history"] = {"columns": HISTORY_COLUMNS, "rows": rows,
        "scope": "Actual stored native frames only; not a reconstruction of missing physical times"}
    result["frame_sources"] = [series.frames[key].provenance() for key in keys]
    result["diagnostics"] = {"columns": DIAGNOSTIC_COLUMNS,
        "rows": [[series.diagnostics[k].get(name) for name in DIAGNOSTIC_COLUMNS] for k in selected],
        "stride": history_stride, "available_rows": len(diagnostic_keys),
        "continuation_interval": result.get("stationarity", {}).get("policy", {}).get("diagnostic_interval"),
        "prefix_interval": series.config["dt"] if series.config and result.get("seed_prefix", {}).get("status") == "sha_verified" else None,
        "missing_values": "null means this quantity was not recorded at that actual time; no zero filling",
        "scope": "Recorded actual-step diagnostics; thinning is explicit and does not affect frozen window gates"}
    result["coverage"] = {"native_times": [float(k) for k in keys], "frames": len(keys),
                          "first_time": float(keys[0]) if keys else None, "last_time": float(keys[-1]) if keys else None}
    if series.config:
        n = series.config["cells"]
        result["probe_coordinates"] = {name: {"requested_xy": list(point),
            "actual_xy": [(min(n - 1, int(value * n)) + .5) / n for value in point]}
            for name, point in continuation.PROBES.items()}
        result["probe_locations"] = result["probe_coordinates"]
    return result


def analyze(campaign_path, data_root=None, seed_directory=None, profile_times=(10., 25., 50., 100., 200., 400.), history_stride=1):
    if history_stride < 1 or any(not np.isfinite(t) or t < 0 for t in profile_times):
        raise ValueError("Positive history stride and finite nonnegative requested profile times are required")
    campaign_path = Path(campaign_path).resolve()
    payload = campaign_path.read_bytes(); campaign = json.loads(payload)
    source_proof = admit_sources(campaign)
    if len({case["id"] for case in campaign["cases"]}) != len(campaign["cases"]):
        raise ValueError("Duplicate campaign case IDs")
    if data_root is None:
        native = Path(campaign["campaign_directory"])
        data_root = native if native.is_dir() else campaign_path.parent
    data_root = Path(data_root).resolve()
    policy_root = data_root / "policies"
    if not policy_root.is_dir():
        policy_root = campaign_path.parent / "policies"
    report = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_sha256": sha_bytes(payload), "campaign_directory_recorded": campaign["campaign_directory"],
        "analyzed_data_root": str(data_root), "analysis_source_sha256": continuation.sha256(Path(__file__)),
        "numerical_source_sha256": source_proof, "qualification": QUALIFICATION, "cases": {}, "comparisons": [],
        "time_matching": "Intersection of exact original-dt physical-step lattice keys. Recorded floating-point times must agree within serialization tolerance; no temporal interpolation, extrapolation or nearest-frame substitution.",
        "spatial_comparison": "Bilinear fine-to-coarse cell-center sampling only for difference norms. Each pressure field has zero area mean on its original grid; no additional gauge fit. Profiles use native cell-center coverage and linear interpolation only to x/y=0.5.",
        "units": {"time": "t U_lid/L", "velocity": "u/U_lid, v/U_lid", "pressure": "(p-mean(p))/(rho U_lid^2)", "vorticity": "omega L/U_lid", "streamfunction": "psi/(U_lid L)"},
        "requested_profile_times": list(profile_times), "errors": []}
    series = {}
    for case in campaign["cases"]:
        try:
            series[case["id"]] = load_case(case, campaign, data_root, seed_directory, policy_root)
        except EVIDENCE_ERRORS + (RuntimeError,) as exc:
            status = "unstable_manifest" if isinstance(exc, RuntimeError) else "invalid"
            series[case["id"]] = Series(case, summary={"case": case, "admission": status, "error": str(exc), "qualification": QUALIFICATION})
            report["errors"].append({"case": case["id"], "error": str(exc)})
    requests = {Fraction(str(t)) for t in profile_times}
    needed = {name: requests.intersection(item.frames) | ({max(item.frames)} if item.frames else set()) for name, item in series.items()}
    plans = plan_pairs(campaign["cases"])
    for pair in plans:
        a, b = series.get(pair["candidate_id"]), series.get(pair["reference_id"])
        common = sorted(set(a.frames).intersection(b.frames)) if a is not None and b is not None else []
        pair["profile_times"] = [float(t) for t in sorted(requests.intersection(common) | ({common[-1]} if common else set()))]
        if common:
            needed[pair["candidate_id"]].update(Fraction(str(t)) for t in pair["profile_times"])
            needed[pair["reference_id"]].update(Fraction(str(t)) for t in pair["profile_times"])
    # Admit every referenced native frame before making any pairwise claim.
    for name, item in series.items():
        try:
            report["cases"][name] = describe_series(item, needed[name], history_stride)
            report["cases"][name]["unavailable_requested_profile_times"] = [float(t) for t in sorted(requests - set(item.frames))]
        except EVIDENCE_ERRORS as exc:
            item.frames.clear(); item.config = None
            item.summary.update(admission="invalid", error=str(exc))
            item.summary.pop("profiles", None)
            report["cases"][name] = item.summary
            report["errors"].append({"case": name, "error": str(exc)})
    for pair in plans:
        pair.update(meaning=MEANING, frames=[])
        a, b = series.get(pair["candidate_id"]), series.get(pair["reference_id"])
        if a is None or b is None or not a.frames or not b.frames:
            pair.update(status="missing_cases", profile_times=[])
        else:
            try:
                check_pair(pair["kind"], a.config, b.config)
                common = sorted(set(a.frames).intersection(b.frames))
                pair.update(status="matched" if common else "no_common_frames",
                    unavailable_requested_profile_times=[float(t) for t in sorted(requests - set(common))],
                    unmatched_candidate_times=[float(t) for t in sorted(set(a.frames) - set(b.frames))],
                    unmatched_reference_times=[float(t) for t in sorted(set(b.frames) - set(a.frames))])
                for key in common:
                    pair["frames"].append({"time": float(key),
                        "candidate_recorded_time": a.frames[key].time, "reference_recorded_time": b.frames[key].time,
                        **compare_fields(a.frames[key].fields(), b.frames[key].fields())})
            except EVIDENCE_ERRORS as exc:
                pair.update(status="invalid", frames=[], profile_times=[], error=str(exc))
                report["errors"].append({"comparison": pair["id"], "error": str(exc)})
        report["comparisons"].append(pair)
    report["coverage"] = {status: sum(item["admission"] == status for item in report["cases"].values())
        for status in ("verified_checkpoint", "seed_only", "missing", "invalid", "unstable_manifest")}
    report["status"] = "invalid_evidence" if report["errors"] else (
        "committed_campaign_snapshot" if report["coverage"]["verified_checkpoint"] == len(series) else "partial_campaign")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="Campaign/mirror root containing runs/ and policies/")
    parser.add_argument("--seed-directory", type=Path, help="Original fine128 t=0..10 baseline NPZ/JSON directory")
    parser.add_argument("--profile-times", type=float, nargs="+", default=[10., 25., 50., 100., 200., 400.])
    parser.add_argument("--history-stride", type=int, default=1)
    parser.add_argument("--require-all", action="store_true", help="Return exit3 if any case lacks a committed recovery")
    parser.add_argument("--output", type=Path, help="Compact current JSON report; omitted prints JSON to stdout")
    args = parser.parse_args(argv)
    report = analyze(args.campaign, args.data_root, args.seed_directory, args.profile_times, args.history_stride)
    if args.output:
        output = args.output.resolve()
        root = Path(report["analyzed_data_root"])
        protected = [root / "runs", root / "policies", args.campaign.resolve().parent / "policies",
                     (args.seed_directory or ROOT / "results/cavity_startup/fine128").resolve()]
        if (output.suffix.lower() != ".json" or output == args.campaign.resolve()
                or any(output.is_relative_to(path) for path in protected)):
            raise ValueError("Analysis output must be a JSON report outside case/recovery/seed/policy inputs")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".{os.getpid()}.part")
        try:
            temporary.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(json.dumps({"output": str(output), "sha256": continuation.sha256(output),
                          "status": report["status"], "coverage": report["coverage"], "errors": report["errors"]}, indent=2))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["errors"] else (3 if args.require_all and report["status"] == "partial_campaign" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
