"""Confirm a closed cavity candidate with an explicit native 3/2 -> 6/4 fork.

No rest-start history is invented. The imported state is the parent's actual
native MAC field at its verified final physical time. A new, honestly typed
admission receipt documents the operator-configuration fork. The unchanged
continuation engine's compatibility field named ``seed_receipt`` contains this
derived receipt, not a modified copy of the parent's original source receipt.

Only one complete additional physical window is evaluated. The engine's own
three-window candidate flag is not reused as the confirmation decision: the
separate confirmation.json also compares the new window means with the parent's
final candidate-window means. This is independent tighter-coupling confirmation
on the same grid/dt, never mesh or three-dimensional validation.

Example::

    python qa/confirm_cavity_stationarity.py --parent CASE/continuation \
        --output DISTINCT_CONFIRMATION_DIRECTORY --progress

An optional absolute --target-time permits a bounded partial stage. Repeating
the command resumes exact accumulated state. Parent and fork source hashes,
policies and parent checkpoint identities must remain unchanged on resume.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from flowmllab import cavity_startup_continuation as engine
from flowmllab.pressure_velocity import _residual
from flowmllab.pressure_velocity_experiments import ExperimentalConfig


SCHEMA = "flowmllab.cavity.stationarity.confirmation.v1"
SOURCE_KIND = "native_stationarity_confirmation"
ORIGIN_ARRAYS = ("u_faces", "v_faces", "p", "parent_mean_u", "parent_mean_v", "parent_mean_p")


def confirmation_sources():
    return {"qa/confirm_cavity_stationarity.py": engine.sha256(__file__)}


def _closed_parent(parent):
    if (parent / "writer.lock").exists() or (parent.parent / ".launch.lock").exists():
        raise FileExistsError("Parent has an active or unresolved writer/launcher lock; do not fork live work")


def _validated_recovery_pair(directory, manifest, expected_contract=None, *, require_previous=False, role="Confirmation"):
    """Admit manifest identity/order before any retirement or scientific write.

    A valid checksum proves payload integrity, not that a manifest has named
    its newest recovery ``current``. Enforce both identities and chronology,
    including an outstanding retirement record, before calling engine helpers
    that may mutate the manifest or remove a superseded checkpoint.
    """
    if not isinstance(manifest, dict) or manifest.get("schema") != engine.SCHEMA:
        raise ValueError(f"{role} recovery manifest schema is invalid")
    digest = manifest.get("contract_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{role} recovery manifest contract identity is invalid")
    if expected_contract is not None and digest != engine.fingerprint(expected_contract):
        raise ValueError(f"{role} recovery manifest contract identity does not match the admitted fork")

    def record(name, required=True):
        value = manifest.get(name)
        if value is None and not required:
            return None, None
        if not isinstance(value, dict) or not {"file", "sha256", "step", "time"} <= set(value):
            raise ValueError(f"{role} recovery manifest {name} record is incomplete")
        if (not isinstance(value["file"], str) or not value["file"]
                or not isinstance(value["sha256"], str) or len(value["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in value["sha256"])
                or type(value["step"]) is not int or value["step"] < 0
                or type(value["time"]) not in (int, float) or not np.isfinite(value["time"]) or value["time"] < 0):
            raise ValueError(f"{role} recovery manifest {name} identity is invalid")
        # The engine retirement helper accepts any in-case relative path, so
        # admit only this engine's immutable restart namespace before calling
        # it. An origin or snapshot is never a disposable restart, including
        # when a malicious/malformed record points at an already absent file.
        filename = re.fullmatch(r"restarts/checkpoint_(\d{12})_([0-9a-f]{32})\.npz", value["file"])
        if filename is None or int(filename.group(1)) != value["step"]:
            raise ValueError(f"{role} recovery manifest {name} has an invalid restart namespace or filename")
        managed_restarts = Path(directory).resolve() / "restarts"
        member = engine._safe_member(directory, value["file"])
        expected = managed_restarts / Path(value["file"]).name
        if managed_restarts.resolve() != managed_restarts or member != expected:
            raise ValueError(f"{role} recovery manifest {name} redirects outside its managed restart namespace")
        return value, member

    current_record, current_path = record("current")
    previous_record, previous_path = record("previous", required=require_previous)
    if previous_record is not None:
        same_file = current_path == previous_path
        if current_path.exists() and previous_path.exists():
            same_file = same_file or os.path.samefile(current_path, previous_path)
        if same_file or current_record["sha256"] == previous_record["sha256"]:
            raise ValueError(f"{role} current and previous recovery identities must be distinct")
        if previous_record["step"] >= current_record["step"] or previous_record["time"] >= current_record["time"]:
            raise ValueError(f"{role} recovery pair must be ordered: previous strictly precedes current")
    current = engine.load_checkpoint(directory, current_record, expected_contract)
    if current["contract_sha256"] != digest:
        raise ValueError(f"{role} recovery manifest contract identity disagrees with its current payload")
    previous = None
    if previous_record is not None:
        previous = engine.load_checkpoint(directory, previous_record, current["contract"])
        if (previous["contract_sha256"] != digest or previous["config"] != current["config"]
                or previous["step"] >= current["step"] or previous["time"] >= current["time"]):
            raise ValueError(f"{role} loaded recovery pair is inconsistent")
    elif current["step"] != current["origin_step"] or current["time"] != current["origin_time"]:
        raise ValueError(f"{role} advanced recovery is missing its previous complete checkpoint")
    pending, pending_path = record("pending_superseded_restart", required=False)
    if pending is not None:
        if previous_record is None:
            raise ValueError(f"{role} retirement requires two verified retained recoveries")
        for retained, path in ((current_record, current_path), (previous_record, previous_path)):
            same_file = pending_path == path
            if pending_path.exists() and path.exists():
                same_file = same_file or os.path.samefile(pending_path, path)
            if same_file or pending["sha256"] == retained["sha256"]:
                raise ValueError(f"{role} pending retirement is not distinct from retained recovery")
        if pending["step"] >= previous_record["step"] or pending["time"] >= previous_record["time"]:
            raise ValueError(f"{role} pending retirement must be older than both retained recoveries")
        # Missing means a prior process may already have unlinked it before
        # committing the retirement receipt; the frozen engine records that
        # situation without claiming an additional deletion.
        if pending_path.exists():
            engine.load_checkpoint(directory, pending, current["contract"])
    return current, previous


def admit_parent(parent, expected_policy=None):
    """Acquire a brief read-admission guard; never change parent scientific data.

    ``expected_policy`` is explicit dependency injection for labelled tiny test
    fixtures. The production CLI always requires the exact frozen engineering
    policy returned by the unchanged continuation module.
    """
    parent = Path(parent).resolve()
    if not parent.is_dir():
        raise FileNotFoundError("Parent continuation directory does not exist")
    _closed_parent(parent)
    # Using the existing exclusive guard prevents a competing continuation from
    # writing between checksum checks. It writes/removes only its own lock token.
    with engine.writer_lock(parent):
        if (parent.parent / ".launch.lock").exists():
            raise FileExistsError("Parent launcher became active during read admission")
        path = parent / "restart.json"
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
        current, previous = _validated_recovery_pair(parent, manifest, require_previous=True, role="Parent")
        if manifest.get("pending_superseded_restart"):
            raise ValueError("Parent restart retirement is unfinished; resolve it before confirmation")
        if current["contract"]["continuation_source_sha256"] != engine.source_hashes():
            raise ValueError("Parent numerical/continuation source integrity does not match the unchanged local engine")
        if current["contract"]["runtime"] != engine.runtime_contract():
            raise ValueError("Parent numerical runtime differs; explicit compatible admission is required")
        config = ExperimentalConfig(**current["config"])
        policy = engine.production_policy(config.reynolds) if expected_policy is None else expected_policy
        if current["policy"] != policy or current["contract"]["policy_sha256"] != engine.fingerprint(policy):
            raise ValueError("Parent policy is not the explicitly expected frozen policy")
        if config.method != "pimple" or (config.outer_correctors, config.pressure_correctors) != (3, 2):
            raise ValueError("This independent confirmation requires a PIMPLE 3/2 parent and forks only to 6/4")
        windows = current["completed_windows"]
        if (not current["candidate"] or current["consecutive_passes"] < 3
                or len(windows) < 3 or not all(row["passed"] and not row["partial_initial_window"] for row in windows[-3:])):
            raise ValueError("Parent has not passed three complete frozen candidate windows")
        if (current["window"]["samples"] != 0 or windows[-1]["end_time"] != current["time"]
                or not current["previous_mean_available"]):
            raise ValueError("Fork requires a completed parent window boundary, not an unfinished candidate window")
        if not windows[-1]["candidate"]:
            raise ValueError("Parent final complete window does not carry the verified candidate flag")
        last = dict(zip(engine.DIAGNOSTIC_COLUMNS, current["last_diagnostics"] or []))
        if last.get("step") != current["step"] or last.get("time") != current["time"]:
            raise ValueError("Parent last-step diagnostic identity is incomplete")
        steady = _residual(current["u_faces"], current["v_faces"], current["p"], config)
        if steady >= policy["thresholds"]["steady_momentum_defect_linf"]:
            raise ValueError("Independently recomputed parent steady momentum defect fails its frozen gate")
        for key, value in last.items():
            if key in policy["thresholds"] and value >= policy["thresholds"][key]:
                raise ValueError(f"Parent final actual-step {key} fails its frozen gate")
        # Repeat the closed-byte checks while our exclusive admission guard holds.
        if path.read_bytes() != manifest_bytes:
            raise ValueError("Parent manifest changed during admission")
        for record in (manifest["current"], manifest["previous"]):
            if engine.sha256(engine._safe_member(parent, record["file"])) != record["sha256"]:
                raise ValueError("Parent checkpoint changed during admission")
        if (parent.parent / ".launch.lock").exists():
            raise FileExistsError("Parent launcher became active during read admission")
        identity = {"schema": SCHEMA + ".parent", "manifest_sha256": engine.sha256(path),
                    "current": deepcopy(manifest["current"]), "previous": deepcopy(manifest["previous"]),
                    "actual_time": current["time"], "actual_step": current["step"],
                    "original_config": deepcopy(current["config"]),
                    "original_contract": deepcopy(current["contract"]),
                    "original_contract_sha256": current["contract_sha256"],
                    "policy_sha256": engine.fingerprint(current["policy"]),
                    "consecutive_passes": current["consecutive_passes"],
                    "last_complete_window": deepcopy(windows[-1]),
                    "recomputed_steady_momentum_defect_linf": steady}
    return current, identity


def _target_config(parent):
    original = ExperimentalConfig(**parent["config"])
    configured = asdict(replace(original, outer_correctors=6, pressure_correctors=4))
    changed = {key for key in configured if configured[key] != parent["config"][key]}
    if changed != {"outer_correctors", "pressure_correctors"}:
        raise ValueError("Only the declared coupling counts may change")
    return configured


def _origin_metadata(parent, identity, configured):
    return {"schema": SCHEMA + ".native_origin", "source_kind": SOURCE_KIND,
            "parent_identity": identity, "original_config": parent["config"], "config": configured,
            "actual_time": parent["time"], "actual_step": parent["step"],
            "initial_condition": "Imported actual parent MAC state; NOT a rest start and NOT a new t*=0 history",
            "parent_mean_definition": "Final complete candidate-window means of every physical step, from verified parent recovery",
            "pressure_gauge": "Native cell-area mean pressure equals zero; no pressure reconstruction or re-gauging"}


def create_origin(output, parent, parent_path, identity, configured):
    arrays = {name: parent[name].copy() for name in ("u_faces", "v_faces", "p")}
    for destination, source in zip(("parent_mean_u", "parent_mean_v", "parent_mean_p"),
                                   ("previous_mean_u", "previous_mean_v", "previous_mean_p")):
        arrays[destination] = parent[source].copy()
    origin = output / "fork_origin.npz"
    engine._save_npz(origin, arrays, _origin_metadata(parent, identity, configured))
    receipt = {"schema": SCHEMA + ".native_fork_admission", "source_kind": SOURCE_KIND,
               "trajectory": origin.name, "trajectory_sha256": engine.sha256(origin), "frames": 1,
               "actual_steps": parent["step"], "end_time": parent["time"], "config": configured,
               "original_config": parent["config"], "parent_identity": identity,
               "parent_source_location": str(Path(parent_path).resolve()),
               "configuration_role": "The config applies to future fork steps only; imported fields were generated by original_config",
               "initial_condition": "One imported native parent state at actual parent time; no rest-start trajectory is claimed",
               "continuation_source_sha256": engine.source_hashes(),
               "confirmation_source_sha256": confirmation_sources()}
    engine.atomic_json(output / "fork_admission.json", receipt)
    return arrays, receipt


def read_origin(output, parent, identity, configured):
    receipt_path = output / "fork_admission.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != SCHEMA + ".native_fork_admission" or receipt.get("source_kind") != SOURCE_KIND
            or receipt.get("parent_identity") != identity or receipt.get("config") != configured
            or receipt.get("original_config") != parent["config"]
            or receipt.get("continuation_source_sha256") != engine.source_hashes()
            or receipt.get("confirmation_source_sha256") != confirmation_sources()):
        raise ValueError("Native-fork parent/source/configuration admission changed; refusing resume")
    origin = engine._safe_member(output, receipt["trajectory"])
    if engine.sha256(origin) != receipt["trajectory_sha256"]:
        raise ValueError("Native fork-origin checksum mismatch")
    arrays, metadata = engine._load_npz(origin)
    if set(arrays) != set(ORIGIN_ARRAYS) or metadata != _origin_metadata(parent, identity, configured):
        raise ValueError("Native fork-origin payload or lineage is incomplete")
    for name in ("u_faces", "v_faces", "p"):
        if not np.array_equal(arrays[name], parent[name]):
            raise ValueError("Imported native field differs from the verified actual parent state")
    for name, source in zip(("parent_mean_u", "parent_mean_v", "parent_mean_p"),
                             ("previous_mean_u", "previous_mean_v", "previous_mean_p")):
        if not np.array_equal(arrays[name], parent[source]):
            raise ValueError("Imported parent final-window mean differs from verified recovery")
    return arrays, receipt


def fork_state(parent, output, arrays, receipt):
    """New history with honest typed admission, using the frozen engine schema."""
    config = receipt["config"]
    contract = {"source_kind": SOURCE_KIND, "seed_sha256": receipt["trajectory_sha256"],
                "seed_receipt_sha256": engine.sha256(output / "fork_admission.json"),
                "seed_receipt": receipt, "continuation_source_sha256": engine.source_hashes(),
                "confirmation_source_sha256": confirmation_sources(), "runtime": engine.runtime_contract(),
                "policy_sha256": engine.fingerprint(parent["policy"]),
                "physics": "Same native parent state, grid, dt, Reynolds number, BC, convection and momentum operators; only PIMPLE3/2 becomes6/4",
                "compatibility_field_semantics": "seed_receipt is a new typed native-fork admission, not a rest-start solver receipt or modified parent receipt",
                "velocity_norm": parent["contract"]["velocity_norm"],
                "window_means": parent["contract"]["window_means"],
                "qualification": "Independent tighter-coupling confirmation on the same fixed grid/dt only; not mesh convergence, a new steady-candidate campaign, or 3D validation"}
    state = {"schema": engine.SCHEMA, "config": deepcopy(config), "policy": deepcopy(parent["policy"]),
             "contract": contract, "contract_sha256": engine.fingerprint(contract),
             "origin_time": parent["time"], "origin_step": parent["step"],
             "origin_tick": engine._ticks(parent["time"], config["dt"], "native parent time"),
             "time": parent["time"], "step": parent["step"], "pressure_solves": parent["pressure_solves"],
             "completed_windows": [], "consecutive_passes": 0, "candidate": False,
             "previous_mean_available": False, "snapshots": [], "last_diagnostics": None,
             "diagnostics": np.empty((0, len(engine.DIAGNOSTIC_COLUMNS)), dtype=np.float64),
             "probe_locations": deepcopy(parent["probe_locations"])}
    for name in ("u_faces", "v_faces", "p"):
        state[name] = arrays[name].copy()
    for name, source in zip(("previous_mean_u", "previous_mean_v", "previous_mean_p"), ("u_faces", "v_faces", "p")):
        state[name] = np.zeros_like(state[source])
    engine._new_window(state)
    engine.validate_state(state)
    return state


def confirmation_report(state, origin, reason):
    duration = state["policy"]["window_duration"]
    finish = state["origin_time"] + duration
    complete = np.isclose(state["time"], finish, rtol=0, atol=1e-10)
    metrics, failed, passed = None, [], None
    if complete:
        windows = state["completed_windows"]
        if len(windows) != 1 or windows[0]["partial_initial_window"] or state["window"]["samples"]:
            raise ValueError("Confirmation did not accumulate exactly one additional complete window")
        metrics = deepcopy(windows[0]["metrics"])
        metrics["velocity_window_mean_rms"] = engine._velocity_norm(
            state["previous_mean_u"] - origin["parent_mean_u"],
            state["previous_mean_v"] - origin["parent_mean_v"])[0]
        metrics["pressure_window_mean_rms"] = float(np.sqrt(np.mean((state["previous_mean_p"] - origin["parent_mean_p"]) ** 2)))
        if set(metrics) != set(engine.METRICS) or not np.isfinite(list(metrics.values())).all():
            raise ValueError("Independent confirmation metric set is incomplete")
        failed = engine._failed_metrics(metrics, state["policy"])
        passed = not failed
    return {"schema": SCHEMA, "source_kind": SOURCE_KIND, "complete": bool(complete),
            "independent_confirmation_passed": passed,
            "status": ("confirmed_tighter_coupling" if passed else "not_confirmed") if complete else "in_progress",
            "stop_reason": reason, "parent_time": state["origin_time"], "actual_time": state["time"],
            "required_end_time": finish, "required_window_duration": duration,
            "actual_step": state["step"], "new_physical_steps": state["step"] - state["origin_step"],
            "original_config": state["contract"]["seed_receipt"]["original_config"],
            "confirmation_config": state["config"],
            "metrics": metrics, "failed_metrics": failed, "thresholds": state["policy"]["thresholds"],
            "mean_comparison": "New full-window means versus the immutable parent's final complete candidate-window means",
            "engine_candidate": state["candidate"], "engine_candidate_note": "Expected false: this fork tests one independent window, not three new candidate windows",
            "parent_identity": state["contract"]["seed_receipt"]["parent_identity"],
            "contract_sha256": state["contract_sha256"], "qualification": state["contract"]["qualification"]}


def run_confirmation(parent_path, output, target_time=None, *, stop_requested=None,
                     progress=False, expected_policy=None):
    parent_path, output = Path(parent_path).resolve(), Path(output).resolve()
    if output == parent_path or output.is_relative_to(parent_path) or parent_path.is_relative_to(output):
        raise ValueError("Confirmation output must be distinct from, and outside, the closed parent directory")
    with engine.writer_lock(output):
        parent, identity = admit_parent(parent_path, expected_policy)
        configured = _target_config(parent)
        end = parent["time"] + parent["policy"]["window_duration"]
        target = end if target_time is None else float(target_time)
        target_ticks = engine._ticks(target - parent["time"], configured["dt"], "absolute confirmation target minus parent time")
        if target > end + 1e-10:
            raise ValueError("This confirmation is bounded to one additional complete physical window")
        manifest_path = output / "restart.json"
        if manifest_path.exists():
            arrays, receipt = read_origin(output, parent, identity, configured)
            expected = fork_state(parent, output, arrays, receipt)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            state, _ = _validated_recovery_pair(output, manifest, expected["contract"], role="Confirmation")
            engine.verify_snapshots(output, state)
            if state["origin_step"] + target_ticks < state["step"]:
                raise ValueError("Requested absolute stage target precedes retained confirmation recovery")
            engine._finish_pending_retirement(output, manifest, expected["contract"])
        else:
            unexpected = [path.name for path in output.iterdir() if path.name != "writer.lock"]
            if unexpected:
                raise FileExistsError("Uncommitted fork files are preserved for inspection; never overwrite: " + ", ".join(unexpected))
            arrays, receipt = create_origin(output, parent, parent_path, identity, configured)
            # Verify the full immutable native payload again before first recovery.
            arrays, receipt = read_origin(output, parent, identity, configured)
            state = fork_state(parent, output, arrays, receipt)
            engine.save_snapshot(output, state)
            engine.commit_checkpoint(output, state)
        target_step = state["origin_step"] + target_ticks
        if target_step < state["step"]:
            raise ValueError("Requested absolute stage target precedes retained confirmation recovery")
        cadence = {name: engine._ticks(state["policy"][name + "_interval"], configured["dt"], name + "_interval")
                   for name in ("diagnostic", "snapshot", "checkpoint")}
        reason = "stage_target_reached"
        while state["step"] < target_step:
            if stop_requested is not None and stop_requested():
                reason = "signal_stop_after_complete_step"; break
            tick = state["origin_tick"] + state["step"] - state["origin_step"] + 1
            if (tick % cadence["snapshot"] == 0 and len(state["snapshots"]) >= state["policy"]["max_snapshots"]
                    or tick % cadence["diagnostic"] == 0 and len(state["diagnostics"]) >= state["policy"]["max_diagnostic_rows"]):
                reason = "frozen_storage_cap_reached_no_scientific_data_deleted"; break
            engine.advance_one_step(state)
            if tick % cadence["snapshot"] == 0:
                engine.save_snapshot(output, state)
            if tick % cadence["checkpoint"] == 0:
                engine.commit_checkpoint(output, state)
                engine.export_compact_history(output, state, "independent_confirmation_running")
                engine.atomic_json(output / "confirmation.json", confirmation_report(state, arrays, "running"))
                if progress:
                    print(json.dumps({"source_kind": SOURCE_KIND, "Re": configured["reynolds"], "time": state["time"], "required_end_time": end}), flush=True)
        engine.commit_checkpoint(output, state)
        engine.export_compact_history(output, state, reason)
        report = confirmation_report(state, arrays, reason)
        engine.atomic_json(output / "confirmation.json", report)
        return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-time", type=float)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    stopping = {"signal": None}
    def stop(signum, _frame):
        stopping["signal"] = signum
    previous = {}
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            sig = getattr(signal, name)
            previous[sig] = signal.signal(sig, stop)
    try:
        result = run_confirmation(args.parent, args.output, args.target_time,
                                  stop_requested=lambda: stopping["signal"] is not None, progress=args.progress)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if stopping["signal"] is not None:
        return 75
    if result["stop_reason"].startswith("frozen_storage_cap"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
