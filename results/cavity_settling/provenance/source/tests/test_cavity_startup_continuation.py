"""Independent continuation/recovery admission tests on small real MAC solves."""
from copy import deepcopy
import json

import numpy as np
import pytest

import flowmllab.cavity_startup_continuation as continuation
from flowmllab.cavity_startup import case_summary, observation_times, save_trajectory, simulate_startup
from flowmllab.pressure_velocity import coupling_step
from flowmllab.pressure_velocity_experiments import ExperimentalConfig


@pytest.fixture
def seed(tmp_path):
    config = ExperimentalConfig(method="pimple", reynolds=100, cells=6, dt=.01,
                                end_time=.1, outer_correctors=2, pressure_correctors=2)
    case = simulate_startup(config, observation_times(.1, .02))
    path = tmp_path / "seed.npz"
    digest = save_trajectory(case, path)
    receipt = {**case_summary(case), "trajectory": path.name, "trajectory_sha256": digest,
               "source_sha256": continuation.source_hashes()}
    receipt_path = tmp_path / "seed.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8", newline="\n")
    return path, receipt_path, case


def fixture_policy():
    policy = continuation.production_policy(100)
    policy.update(window_duration=.04, diagnostic_interval=.02, snapshot_interval=.04,
                  checkpoint_interval=.04, criterion_origin="Explicit loose fixture gates; never production")
    policy["thresholds"] = {key: 100. for key in continuation.METRICS}
    policy["thresholds"].update(divergence_linf=1e-8, normal_wall_velocity_linf=1e-8, pressure_mean_abs=1e-10)
    return policy


def state_at(directory):
    manifest = json.loads((directory / "restart.json").read_text(encoding="utf-8"))
    return continuation.load_checkpoint(directory, manifest["current"])


def test_interrupted_mid_window_resume_matches_uninterrupted_real_solver(seed, tmp_path):
    path, receipt, case = seed
    policy = fixture_policy()
    direct, split = tmp_path / "direct", tmp_path / "split"
    continuation.continue_case(path, receipt, policy, direct, .26)
    first = continuation.continue_case(path, receipt, policy, split, .17)
    assert first["unfinished_window_steps"] == 1
    continuation.continue_case(path, receipt, policy, split, .26)
    a, b = state_at(direct), state_at(split)
    for name in continuation.ARRAY_NAMES:
        np.testing.assert_array_equal(a[name], b[name])
    for name in ("window", "completed_windows", "consecutive_passes", "candidate", "last_diagnostics"):
        assert a[name] == b[name]
    # Independent direct calls verify that the wrapper did not alter solver physics.
    u, v, p = [case[name][-1].copy() for name in ("u_faces", "v_faces", "p")]
    config = ExperimentalConfig(**case["config"])
    for _ in range(16):
        u, v, p, _ = coupling_step(u, v, p, config, dt=.01)
    for value, key in zip((u, v, p), ("u_faces", "v_faces", "p")):
        np.testing.assert_array_equal(a[key], value)
    assert a["step"] == 26 and a["time"] == .26
    assert a["completed_windows"][0]["partial_initial_window"]
    assert not a["completed_windows"][0]["passed"]
    assert a["candidate"] and a["consecutive_passes"] == 3
    # Absolute, rather than seed-relative, cadences with no fictitious earlier frames.
    np.testing.assert_allclose(a["diagnostics"][:, 1], np.arange(.12, .261, .02), atol=1e-15)
    np.testing.assert_allclose([item["time"] for item in a["snapshots"]], [.1, .12, .16, .20, .24], atol=1e-15)
    assert len(list((split / "restarts").glob("*.npz"))) == 2
    assert len(list((split / "snapshots").glob("*.npz"))) == 5
    continuation.verify_snapshots(split, b)


def test_seed_mismatch_and_bad_policy_fail_before_initial_checkpoint(seed, tmp_path):
    path, receipt, _ = seed
    policy = fixture_policy()
    bad = deepcopy(policy)
    bad["thresholds"].pop("pressure_anchor_rms")
    with pytest.raises(ValueError, match="every stationarity"):
        continuation.continue_case(path, receipt, bad, tmp_path / "bad_policy", .2)
    doc = json.loads(receipt.read_text(encoding="utf-8"))
    doc["trajectory_sha256"] = "0" * 64
    receipt.write_text(json.dumps(doc), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="checksum"):
        continuation.continue_case(path, receipt, policy, tmp_path / "bad_seed", .2)
    assert not (tmp_path / "bad_seed" / "restart.json").exists()


def test_resume_policy_source_and_corruption_refused_without_losing_previous(seed, tmp_path, monkeypatch):
    path, receipt, _ = seed
    directory = tmp_path / "case"
    policy = fixture_policy()
    continuation.continue_case(path, receipt, policy, directory, .18)
    manifest_path = directory / "restart.json"
    original_manifest = manifest_path.read_bytes()
    changed = deepcopy(policy); changed["thresholds"]["pressure_anchor_rms"] *= .5
    with pytest.raises(ValueError, match="contract mismatch"):
        continuation.continue_case(path, receipt, changed, directory, .2)
    source = continuation.source_hashes()
    source["qa/continue_cavity_startup.py"] = "e" * 64
    with monkeypatch.context() as local:
        local.setattr(continuation, "source_hashes", lambda: source)
        with pytest.raises(ValueError, match="contract mismatch"):
            continuation.continue_case(path, receipt, policy, directory, .2)
    manifest = json.loads(original_manifest)
    current = directory / manifest["current"]["file"]
    current.write_bytes(current.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        continuation.continue_case(path, receipt, policy, directory, .2)
    assert manifest_path.read_bytes() == original_manifest
    assert continuation.load_checkpoint(directory, manifest["previous"])["step"] == manifest["previous"]["step"]


def test_duplicate_writer_and_unknown_directory_are_never_overwritten(seed, tmp_path):
    path, receipt, _ = seed
    directory = tmp_path / "locked"
    with continuation.writer_lock(directory):
        before = (directory / "writer.lock").read_bytes()
        with pytest.raises(FileExistsError, match="existing writer lock"):
            continuation.continue_case(path, receipt, fixture_policy(), directory, .2)
        assert (directory / "writer.lock").read_bytes() == before
    unrelated = tmp_path / "unknown"
    unrelated.mkdir(); (unrelated / "do_not_touch.txt").write_text("retained", encoding="utf-8", newline="\n")
    with pytest.raises(FileExistsError, match="not empty"):
        continuation.continue_case(path, receipt, fixture_policy(), unrelated, .2)
    assert (unrelated / "do_not_touch.txt").read_text(encoding="utf-8") == "retained"


def test_checkpoint_failure_retains_last_verified_recovery(seed, tmp_path, monkeypatch):
    path, receipt, _ = seed
    directory = tmp_path / "case"
    policy = fixture_policy()
    continuation.continue_case(path, receipt, policy, directory, .16)
    state = state_at(directory)
    continuation.advance_one_step(state)
    original_manifest = (directory / "restart.json").read_bytes()
    real_write = continuation.atomic_json
    def fail_manifest(destination, document):
        if destination.name == "restart.json":
            raise OSError("simulated atomic manifest failure")
        return real_write(destination, document)
    with monkeypatch.context() as local:
        local.setattr(continuation, "atomic_json", fail_manifest)
        with pytest.raises(OSError, match="simulated"):
            continuation.commit_checkpoint(directory, state)
    assert (directory / "restart.json").read_bytes() == original_manifest
    assert state_at(directory)["time"] == .16
    manifest = json.loads(original_manifest)
    assert continuation.load_checkpoint(directory, manifest["previous"])["time"] == pytest.approx(.12)
    continuation.continue_case(path, receipt, policy, directory, .24)
    assert len(list((directory / "restarts").glob("*.npz"))) == 2


def test_pending_retirement_recovery_keeps_both_verified_generations(seed, tmp_path, monkeypatch):
    path, receipt, _ = seed
    policy = fixture_policy()
    directory = tmp_path / "case"
    continuation.continue_case(path, receipt, policy, directory, .16)
    with monkeypatch.context() as local:
        def fail_retirement(directory, manifest, contract):
            if manifest.get("pending_superseded_restart"):
                raise OSError("simulated kill before superseded restart retirement")
            return manifest
        local.setattr(continuation, "_finish_pending_retirement", fail_retirement)
        with pytest.raises(OSError, match="simulated kill"):
            continuation.continue_case(path, receipt, policy, directory, .20)
    assert state_at(directory)["time"] == .20
    assert len(list((directory / "restarts").glob("*.npz"))) == 3
    continuation.continue_case(path, receipt, policy, directory, .20)
    assert len(list((directory / "restarts").glob("*.npz"))) == 2
    manifest = json.loads((directory / "restart.json").read_text(encoding="utf-8"))
    assert manifest["pending_superseded_restart"] is None
    assert manifest["actual_restart_deletions"][-1]["bytes"] > 0


def test_signal_style_stop_restarts_midwindow_and_fixed_caps_preserve_evidence(seed, tmp_path):
    path, receipt, _ = seed
    policy = fixture_policy()
    checks = {"count": 0}
    def stop():
        checks["count"] += 1
        return checks["count"] > 3
    directory = tmp_path / "stopped"
    result = continuation.continue_case(path, receipt, policy, directory, .26, stop_requested=stop)
    assert result["reason"] == "stop_requested_after_complete_step"
    assert result["step"] == 13
    assert not (directory / "writer.lock").exists()
    continuation.continue_case(path, receipt, policy, directory, .26)
    assert state_at(directory)["candidate"]
    capped = deepcopy(policy); capped["max_snapshots"] = 2
    result = continuation.continue_case(path, receipt, capped, tmp_path / "capped", .26)
    assert result["reason"].startswith("frozen_storage_cap") and result["step"] == 15
    assert len(list((tmp_path / "capped" / "snapshots").glob("*.npz"))) == 2


def test_window_excursions_detect_returning_flow_and_strict_gates_are_not_relaxed(seed):
    path, receipt, _ = seed
    state = continuation.initial_state(path, receipt, fixture_policy())
    invented = deepcopy(state)
    invented.update(candidate=True, consecutive_passes=3)
    with pytest.raises(ValueError, match="status is inconsistent"):
        continuation.validate_state(invented)
    for _ in range(2):
        continuation.advance_one_step(state)
    assert state["completed_windows"][0]["partial_initial_window"]
    # Simulate a measured excursion: the final field may return to its anchor,
    # but the window's already observed maximum must still fail its gate.
    state["window"]["maxima"]["velocity_anchor_linf"] = 101.
    for _ in range(4):
        continuation.advance_one_step(state)
    assert "velocity_anchor_linf" in state["completed_windows"][-1]["failed_metrics"]
    for reynolds in (100, 3200, 5000, 10000):
        policy = continuation.production_policy(reynolds)
        assert policy["thresholds"]["steady_momentum_defect_linf"] == 1e-7
        assert policy["thresholds"]["time_step_momentum_defect_linf"] == 1e-7
        assert policy["window_duration"] == (20. if reynolds <= 3200 else 50.)
    with pytest.raises(ValueError, match="original fixed dt grid"):
        continuation._ticks(.105, .01, "target")


def test_strict_failed_window_roundtrip_preserves_guard_history_order(seed, tmp_path):
    path, receipt, _ = seed
    policy = fixture_policy()
    policy["thresholds"] = continuation.production_policy(100)["thresholds"]
    directory = tmp_path / "strict"
    continuation.continue_case(path, receipt, policy, directory, .17)
    assert not state_at(directory)["candidate"]
    continuation.continue_case(path, receipt, policy, directory, .22)
    state = state_at(directory)
    assert not state["candidate"] and state["consecutive_passes"] == 0
    assert len(state["completed_windows"][-1]["failed_metrics"]) > 1
