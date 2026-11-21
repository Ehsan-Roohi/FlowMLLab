"""Native tighter-coupling fork tests; tiny real MAC solves, labelled loose gates."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pytest

from flowmllab import cavity_startup_continuation as engine
from flowmllab.cavity_startup import case_summary, observation_times, save_trajectory, simulate_startup
from flowmllab.pressure_velocity import coupling_step
from flowmllab.pressure_velocity_experiments import ExperimentalConfig
from qa import confirm_cavity_stationarity as confirmation


@pytest.fixture(scope="module")
def parent_fixture(tmp_path_factory):
    base = tmp_path_factory.mktemp("native_confirmation_parent")
    config = ExperimentalConfig(method="pimple", reynolds=100, cells=6, dt=.01,
                                end_time=.1, outer_correctors=3, pressure_correctors=2)
    case = simulate_startup(config, observation_times(.1, .02))
    seed = base / "seed.npz"
    digest = save_trajectory(case, seed)
    receipt = {**case_summary(case), "trajectory": seed.name, "trajectory_sha256": digest,
               "source_sha256": engine.source_hashes()}
    receipt_path = base / "seed.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8", newline="\n")
    policy = engine.production_policy(100)
    policy.update(window_duration=.04, diagnostic_interval=.01, snapshot_interval=.02,
                  checkpoint_interval=.02, criterion_origin="Explicit loose6-cell test fixture, never production")
    policy["thresholds"] = {name: 100. for name in engine.METRICS}
    policy["thresholds"].update(divergence_linf=1e-8, normal_wall_velocity_linf=1e-8, pressure_mean_abs=1e-10)
    parent = base / "parent"
    engine.continue_case(seed, receipt_path, policy, parent, .24)
    return {"parent": parent, "policy": policy, "seed": seed, "seed_receipt": receipt_path}


def current(directory):
    manifest = json.loads((directory / "restart.json").read_text(encoding="utf-8"))
    return engine.load_checkpoint(directory, manifest["current"])


def persistent_hashes(directory):
    """Hash every persistent file after bounded filesystem-settling checks.

    NFS may briefly expose a renamed open-file entry while an unlinked test
    writer lock is closing. Exclude nothing: require two identical complete
    inventories, retry a disappearing entry, and fail if the directory never
    settles. Stable unknown files and changed scientific payloads remain part
    of the exact before/after comparison.
    """
    deadline = time.monotonic() + 2.
    previous = None
    while True:
        try:
            inventory = {path.relative_to(directory).as_posix(): engine.sha256(path)
                         for path in directory.rglob("*") if path.is_file()}
        except FileNotFoundError:
            previous = None
        else:
            if previous is not None and inventory == previous:
                return inventory
            previous = inventory
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("Full directory hash inventory did not become stable within 2 seconds")
        time.sleep(min(.075, remaining))


def test_persistent_hashes_retries_disappearance_but_excludes_no_files(tmp_path, monkeypatch):
    checkpoint = tmp_path / "scientific_checkpoint.npz"
    unknown = tmp_path / ".nfs_unknown_payload"
    checkpoint.write_bytes(b"unique scientific state")
    unknown.write_bytes(b"unknown evidence must remain included")
    real_hash = engine.sha256
    attempts = {"count": 0}
    def transient_missing(path):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FileNotFoundError("Test-only disappearing directory entry")
        return real_hash(path)
    monkeypatch.setattr(engine, "sha256", transient_missing)
    before = persistent_hashes(tmp_path)
    assert before == {path.name: real_hash(path) for path in (checkpoint, unknown)}
    assert attempts["count"] >= 5
    checkpoint.write_bytes(b"changed scientific state")
    after = persistent_hashes(tmp_path)
    assert after[checkpoint.name] != before[checkpoint.name]
    assert after[unknown.name] == before[unknown.name]


def test_native_first_step_is_exact_six_four_solver_and_parent_unchanged(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    before = persistent_hashes(parent)
    parent_state = current(parent)
    target = parent_state["time"] + parent_state["config"]["dt"]
    output = tmp_path / "fork"
    result = confirmation.run_confirmation(parent, output, target, expected_policy=policy)
    state = current(output)
    assert result["status"] == "in_progress" and result["independent_confirmation_passed"] is None
    assert state["origin_time"] == parent_state["time"]
    assert state["origin_step"] == parent_state["step"]
    assert not state["candidate"] and len(state["completed_windows"]) == 0
    configured = replace(ExperimentalConfig(**parent_state["config"]), outer_correctors=6, pressure_correctors=4)
    u, v, p, count = coupling_step(parent_state["u_faces"].copy(), parent_state["v_faces"].copy(),
                                   parent_state["p"].copy(), configured, dt=configured.dt)
    for name, values in zip(("u_faces", "v_faces", "p"), (u, v, p)):
        np.testing.assert_array_equal(state[name], values)
    assert state["pressure_solves"] == parent_state["pressure_solves"] + count
    assert {key for key in state["config"] if state["config"][key] != parent_state["config"][key]} == {"outer_correctors", "pressure_correctors"}
    origin, metadata = engine._load_npz(output / "fork_origin.npz")
    assert np.count_nonzero(origin["u_faces"]) > 0
    for name in ("u_faces", "v_faces", "p"):
        np.testing.assert_array_equal(origin[name], parent_state[name])
    assert metadata["actual_time"] == parent_state["time"] and "NOT a rest start" in metadata["initial_condition"]
    assert state["contract"]["source_kind"] == confirmation.SOURCE_KIND
    assert state["contract"]["seed_receipt"]["schema"].endswith("native_fork_admission")
    assert state["contract"]["seed_receipt"]["parent_identity"]["original_contract"] == parent_state["contract"]
    assert persistent_hashes(parent) == before


def test_interrupted_confirmation_exact_resume_and_parent_mean_gate(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    direct, resumed = tmp_path / "direct", tmp_path / "resumed"
    complete = confirmation.run_confirmation(parent, direct, expected_policy=policy)
    calls = {"count": 0}
    def stop():
        calls["count"] += 1
        return calls["count"] > 1
    interrupted = confirmation.run_confirmation(parent, resumed, expected_policy=policy, stop_requested=stop)
    assert interrupted["stop_reason"] == "signal_stop_after_complete_step"
    assert interrupted["new_physical_steps"] == 1
    restarted = confirmation.run_confirmation(parent, resumed, expected_policy=policy)
    a, b = current(direct), current(resumed)
    for name in engine.ARRAY_NAMES:
        np.testing.assert_array_equal(a[name], b[name])
    assert a["completed_windows"] == b["completed_windows"]
    assert complete["metrics"] == restarted["metrics"]
    assert complete["complete"] and complete["independent_confirmation_passed"]
    assert complete["new_physical_steps"] == 4 and len(a["completed_windows"]) == 1
    assert complete["engine_candidate"] is False
    assert all(value is not None for value in complete["metrics"].values())
    assert len(list((resumed / "restarts").glob("*.npz"))) == 2
    assert not (resumed / "writer.lock").exists()
    # This deliberately perturbed in-memory test comparator proves the parent
    # mean gate is applied even when every within-fork window gate has passed.
    origin, _ = engine._load_npz(direct / "fork_origin.npz")
    pattern = np.arange(36, dtype=float).reshape(6, 6)
    origin["parent_mean_p"] = origin["parent_mean_p"] + 1000 * (pattern - pattern.mean())
    rejected = confirmation.confirmation_report(a, origin, "test_only_parent_mean_perturbation")
    assert not rejected["independent_confirmation_passed"]
    assert "pressure_window_mean_rms" in rejected["failed_metrics"]


def test_noncandidate_active_parent_partial_window_and_fixture_policy_refusals(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    # The CLI path does not silently adopt the deliberately loose fixture policy.
    with pytest.raises(ValueError, match="expected frozen policy"):
        confirmation.run_confirmation(parent, tmp_path / "production_rejects_fixture")
    with engine.writer_lock(parent):
        with pytest.raises(FileExistsError, match="writer/launcher lock"):
            confirmation.run_confirmation(parent, tmp_path / "active_writer", expected_policy=policy)
    launch = parent.parent / ".launch.lock"
    launch.write_text("test active launcher", encoding="utf-8", newline="\n")
    try:
        with pytest.raises(FileExistsError, match="writer/launcher lock"):
            confirmation.run_confirmation(parent, tmp_path / "active_launcher", expected_policy=policy)
    finally:
        launch.unlink()
    noncandidate = tmp_path / "not_yet_candidate"
    engine.continue_case(parent_fixture["seed"], parent_fixture["seed_receipt"], policy, noncandidate, .20)
    with pytest.raises(ValueError, match="three complete"):
        confirmation.run_confirmation(noncandidate, tmp_path / "no_candidate_fork", expected_policy=policy)
    partial = tmp_path / "partial_parent"
    shutil.copytree(parent, partial)
    engine.continue_case(parent_fixture["seed"], parent_fixture["seed_receipt"], policy, partial, .25)
    with pytest.raises(ValueError, match="window boundary"):
        confirmation.run_confirmation(partial, tmp_path / "partial_fork", expected_policy=policy)


def test_parent_changed_or_corrupt_and_child_origin_corruption_refuse_resume(parent_fixture, tmp_path):
    policy = parent_fixture["policy"]
    parent = tmp_path / "parent"
    shutil.copytree(parent_fixture["parent"], parent)
    fork = tmp_path / "fork"
    parent_time = current(parent)["time"]
    confirmation.run_confirmation(parent, fork, parent_time + .01, expected_policy=policy)
    before = (fork / "restart.json").read_bytes()
    manifest = json.loads((parent / "restart.json").read_text(encoding="utf-8"))
    checkpoint = parent / manifest["current"]["file"]
    saved = checkpoint.read_bytes()
    checkpoint.write_bytes(saved + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        confirmation.run_confirmation(parent, fork, expected_policy=policy)
    checkpoint.write_bytes(saved)
    origin = fork / "fork_origin.npz"
    origin.write_bytes(origin.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="origin checksum"):
        confirmation.run_confirmation(parent, fork, expected_policy=policy)
    assert (fork / "restart.json").read_bytes() == before
    recovery = json.loads(before)
    engine.load_checkpoint(fork, recovery["previous"])


def test_fork_duplicate_guard_scope_and_one_window_bound(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    output = tmp_path / "fork"
    with engine.writer_lock(output):
        with pytest.raises(FileExistsError, match="existing writer lock"):
            confirmation.run_confirmation(parent, output, expected_policy=policy)
    with pytest.raises(ValueError, match="distinct"):
        confirmation.run_confirmation(parent, parent / "forbidden_fork", expected_policy=policy)
    with pytest.raises(ValueError, match="bounded to one"):
        confirmation.run_confirmation(parent, tmp_path / "too_long", 1., expected_policy=policy)
    assert not (tmp_path / "too_long" / "fork_origin.npz").exists()


def test_tampered_child_manifest_refuses_before_any_retirement_or_mutation(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    fork = tmp_path / "fork"
    origin_time = current(parent)["time"]
    confirmation.run_confirmation(parent, fork, origin_time + .02, expected_policy=policy)
    path = fork / "restart.json"
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    # Both genuine payloads are SHA-valid before the malicious role swap.
    engine.load_checkpoint(fork, original["current"])
    engine.load_checkpoint(fork, original["previous"])
    snapshot = next((fork / "snapshots").glob("frame_*.npz"))
    for mutation in ("swapped", "schema", "contract", "duplicate", "missing_previous", "retire_newer",
                     "retire_origin", "retire_snapshot", "retire_absent_outside", "retire_invalid_name"):
        tampered = deepcopy(original)
        if mutation == "swapped":
            tampered["current"], tampered["previous"] = tampered["previous"], tampered["current"]
        elif mutation == "schema":
            tampered["schema"] = "wrong_schema"
        elif mutation == "contract":
            tampered["contract_sha256"] = "0" * 64
        elif mutation == "duplicate":
            tampered["previous"] = deepcopy(tampered["current"])
        elif mutation == "missing_previous":
            tampered["previous"] = None
        elif mutation == "retire_newer":
            tampered["pending_superseded_restart"] = deepcopy(tampered["current"])
        else:
            protected = fork / "fork_origin.npz" if mutation == "retire_origin" else snapshot
            relative = protected.relative_to(fork).as_posix()
            if mutation == "retire_absent_outside":
                relative = "snapshots/absent_restart.npz"
            elif mutation == "retire_invalid_name":
                relative = "restarts/fork_origin.npz"
            tampered["pending_superseded_restart"] = {
                "file": relative, "sha256": engine.sha256(protected),
                "step": original["previous"]["step"] - 1,
                "time": original["previous"]["time"] - .01}
        path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8", newline="\n")
        before = persistent_hashes(fork)
        parent_before = persistent_hashes(parent)
        with pytest.raises(ValueError, match="manifest|distinct|ordered|previous complete"):
            confirmation.run_confirmation(parent, fork, expected_policy=policy)
        assert persistent_hashes(fork) == before, mutation
        assert persistent_hashes(parent) == parent_before, mutation
        assert path.read_bytes() == json.dumps(tampered, sort_keys=True).encode("utf-8")
        assert (fork / original["current"]["file"]).is_file()
        assert (fork / original["previous"]["file"]).is_file()
        path.write_bytes(original_bytes)
    # A legitimate checkpoint at the fork origin has no previous fork yet.
    initial = tmp_path / "initial_only"
    confirmation.run_confirmation(parent, initial, origin_time, expected_policy=policy)
    result = confirmation.run_confirmation(parent, initial, expected_policy=policy)
    assert result["complete"]


def test_already_absent_genuine_retirement_has_no_invented_removed_bytes(parent_fixture, tmp_path):
    parent, policy = parent_fixture["parent"], parent_fixture["policy"]
    fork = tmp_path / "fork"
    origin_time = current(parent)["time"]
    confirmation.run_confirmation(parent, fork, origin_time + .02, expected_policy=policy)
    path = fork / "restart.json"
    formerly_retained = deepcopy(json.loads(path.read_text(encoding="utf-8"))["previous"])
    # Finishing the window legitimately supersedes and retires the origin
    # checkpoint; the immutable fork_origin.npz remains separate and retained.
    confirmation.run_confirmation(parent, fork, expected_policy=policy)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert not (fork / formerly_retained["file"]).exists()
    assert formerly_retained["step"] < manifest["previous"]["step"]
    before_log = deepcopy(manifest["actual_restart_deletions"])
    retained = {record["file"]: engine.sha256(fork / record["file"])
                for record in (manifest["current"], manifest["previous"])}
    manifest["pending_superseded_restart"] = formerly_retained
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8", newline="\n")
    result = confirmation.run_confirmation(parent, fork, expected_policy=policy)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert result["complete"]
    assert after["pending_superseded_restart"] is None
    assert after["actual_restart_deletions"] == before_log
    assert after["retirement_recovery_notes"][-1] == {
        "file": formerly_retained["file"],
        "status": "already absent at resume; no additional removed bytes claimed"}
    assert {name: engine.sha256(fork / name) for name in retained} == retained
    assert (fork / "fork_origin.npz").is_file()


def test_tampered_parent_pair_identity_and_order_preserve_all_parent_files(parent_fixture, tmp_path):
    parent, policy = tmp_path / "parent", parent_fixture["policy"]
    shutil.copytree(parent_fixture["parent"], parent)
    path = parent / "restart.json"
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    engine.load_checkpoint(parent, original["current"])
    engine.load_checkpoint(parent, original["previous"])
    for mutation in ("swapped", "schema", "contract", "duplicate"):
        tampered = deepcopy(original)
        if mutation == "swapped":
            tampered["current"], tampered["previous"] = tampered["previous"], tampered["current"]
        elif mutation == "schema":
            tampered["schema"] = "wrong_schema"
        elif mutation == "contract":
            tampered["contract_sha256"] = "0" * 64
        else:
            tampered["previous"] = deepcopy(tampered["current"])
        path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8", newline="\n")
        before = persistent_hashes(parent)
        output = tmp_path / ("rejected_" + mutation)
        with pytest.raises(ValueError, match="manifest|distinct|ordered"):
            confirmation.run_confirmation(parent, output, expected_policy=policy)
        assert persistent_hashes(parent) == before, mutation
        assert path.read_bytes() == json.dumps(tampered, sort_keys=True).encode("utf-8")
        assert not (output / "fork_origin.npz").exists()
        path.write_bytes(original_bytes)
