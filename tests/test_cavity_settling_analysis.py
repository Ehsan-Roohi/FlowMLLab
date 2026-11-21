"""Small real-solver evidence tests for the read-only continuation analyzer."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from flowmllab import cavity_startup as startup
from flowmllab import cavity_startup_continuation as continuation
from flowmllab.pressure_velocity_experiments import ExperimentalConfig
from qa import analyze_cavity_settling as analysis


def trajectory(n=6, dt=.01, end=.1):
    config = ExperimentalConfig(method="pimple", reynolds=100, cells=n, dt=dt,
        end_time=end, outer_correctors=3, pressure_correctors=2)
    return startup.simulate_startup(config, startup.observation_times(end, .02))


def file_hashes(directory):
    return {str(p.relative_to(directory)): continuation.sha256(p)
            for p in directory.rglob("*") if p.is_file()}


def fixture_policy():
    policy = continuation.production_policy(100)
    policy.update(window_duration=.04, diagnostic_interval=.02, snapshot_interval=.04,
        checkpoint_interval=.04, criterion_origin="Explicit loose test gates; never production")
    policy["thresholds"] = {key: 100. for key in continuation.METRICS}
    policy["thresholds"].update(divergence_linf=1e-8, normal_wall_velocity_linf=1e-8, pressure_mean_abs=1e-10)
    return policy


def write_case(directory, name="test_case", dt=.01, target=.26, policy=None):
    policy = policy or fixture_policy()
    seed_directory = directory / "runs" / name / "seed"
    seed_directory.mkdir(parents=True)
    path = seed_directory / "re100.npz"
    seed = trajectory(dt=dt)
    digest = startup.save_trajectory(seed, path)
    receipt_path = path.with_suffix(".json")
    receipt = {**startup.case_summary(seed), "trajectory": path.name, "trajectory_sha256": digest,
               "source_sha256": continuation.source_hashes()}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    policy_path = directory / "policies" / "re100.json"
    policy_path.parent.mkdir(exist_ok=True)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    output = directory / "runs" / name / "continuation"
    continuation.continue_case(path, receipt_path, policy, output, target)
    return {"id": name, "purpose": "time_control", "cells": 6, "reynolds": 100,
        "dt": dt, "outer": 3, "pressure": 2, "seed_mode": "fresh_rest_start",
        "policy_file_sha256": continuation.sha256(policy_path)}, output, path, receipt_path


@pytest.fixture
def campaign(tmp_path):
    case, output, seed, receipt = write_case(tmp_path)
    document = {"campaign_directory": str(tmp_path), "source_sha256": continuation.source_hashes(),
                "cases": [case]}
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document, case, output, seed, receipt


def test_native_norms_agree_with_retained_trajectory_comparator():
    coarse, fine = trajectory(6), trajectory(8)
    retained = startup.compare_trajectories(coarse, fine)
    for index, expected in enumerate(retained["frames"]):
        actual = analysis.compare_fields(startup.frame_fields(coarse, index), startup.frame_fields(fine, index))
        for key, value in actual.items():
            assert value == pytest.approx(expected[key]) if value is not None else expected[key] is None


def test_time_keys_control_matching_and_signed_profiles():
    assert analysis.time_key(.12000000000000001, .01) == analysis.time_key(.12, .005) == Fraction(3, 25)
    with pytest.raises(ValueError, match="lattice"):
        analysis.time_key(.12001, .01)
    one = trajectory()["config"]
    other = {**one, "dt": .005}
    analysis.check_pair("time", one, other)
    with pytest.raises(ValueError, match="reynolds"):
        analysis.check_pair("time", one, {**other, "reynolds": 400})
    with pytest.raises(ValueError, match="dt"):
        analysis.check_pair("grid", one, other)
    xy = (np.arange(6) + .5) / 6
    xx, yy = np.meshgrid(xy, xy)
    fields = {"x": xy, "y": xy, "u": yy-.5, "v": xx-.5,
              "p": xx+2*yy-1.5, "omega": xx-yy}
    profile = analysis.profiles(fields, 1.)
    np.testing.assert_allclose(profile["vertical"]["p"], 2*xy-1.)
    np.testing.assert_allclose(profile["horizontal"]["p"], xy-.5)
    assert 0 < profile["vertical"]["y"][0] < profile["vertical"]["y"][-1] < 1
    assert min(profile["vertical"]["p"]) < 0 < max(profile["vertical"]["p"])


def test_committed_windows_snapshots_and_actual_diagnostics_are_read_only(campaign):
    path, document, case, output, _, _ = campaign
    before = file_hashes(path.parent)
    report = analysis.analyze(path, profile_times=(.1, .16, .25, .26))
    entry = report["cases"][case["id"]]
    assert report["status"] == "committed_campaign_snapshot" and not report["errors"]
    assert entry["admission"] == "verified_checkpoint"
    assert entry["stationarity"]["candidate"]
    assert entry["stationarity"]["consecutive_passes"] == 3
    assert entry["stationarity"]["completed_windows"][0]["partial_initial_window"]
    assert "candidates only" in report["qualification"]
    assert entry["unavailable_requested_profile_times"] == [.25]
    assert entry["frame_sources"][-1]["origin"] == "committed_checkpoint"
    assert entry["frame_sources"][-1]["sha256"]
    assert entry["coverage"]["last_time"] == .26
    columns, rows = entry["diagnostics"]["columns"], entry["diagnostics"]["rows"]
    omega = columns.index("upper_left_omega")
    assert rows[0][omega] is None and rows[-1][omega] is not None
    assert entry["probe_locations"]["upper_left"]["actual_xy"] == pytest.approx([1/12, 11/12])
    assert entry["frame_history"]["rows"][0][4] is None  # no invented initial residual
    json.dumps(report, allow_nan=False)
    assert file_hashes(path.parent) == before


def test_corrupted_snapshot_and_missing_case_are_explicit(campaign):
    path, document, case, output, _, _ = campaign
    missing = {**case, "id": "not_started", "dt": .005}
    document["cases"].append(missing)
    path.write_text(json.dumps(document), encoding="utf-8")
    partial = analysis.analyze(path)
    assert partial["status"] == "partial_campaign"
    assert partial["cases"]["not_started"]["admission"] == "missing"
    manifest = json.loads((output / "restart.json").read_text())
    state = continuation.load_checkpoint(output, manifest["current"])
    native = output / state["snapshots"][-1]["file"]
    native.write_bytes(native.read_bytes() + b"deliberate test corruption")
    before = file_hashes(path.parent)
    invalid = analysis.analyze(path)
    assert invalid["status"] == "invalid_evidence"
    assert invalid["cases"][case["id"]]["admission"] == "invalid"
    assert "SHA-256 mismatch" in invalid["cases"][case["id"]]["error"]
    assert "profiles" not in invalid["cases"][case["id"]]
    assert file_hashes(path.parent) == before


def test_atomic_manifest_change_is_retried_not_mixed(campaign, monkeypatch):
    path, document, case, output, seed, receipt = campaign
    real = analysis.checkpoint_state
    changed = False
    def checkpoint_then_writer(directory, record):
        nonlocal changed
        state = real(directory, record)
        if not changed:
            changed = True
            continuation.continue_case(seed, receipt, fixture_policy(), output, .28)
        return state
    monkeypatch.setattr(analysis, "checkpoint_state", checkpoint_then_writer)
    series = analysis.load_case(case, document, path.parent, None, path.parent / "policies")
    assert changed and max(series.frames) == Fraction(7, 25)
    assert series.summary["provenance"]["restart_manifest_sha256"] == continuation.sha256(output / "restart.json")


def test_comparison_intersects_actual_frames_without_temporal_interpolation(campaign, monkeypatch):
    path, document, case, output, seed, receipt = campaign
    control, *_ = write_case(path.parent, "smaller_dt", .005, .30)
    document["cases"].append(control)
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(analysis, "plan_pairs", lambda _: [{"id": "small_grid_time_test", "kind": "time",
        "candidate_id": case["id"], "reference_id": control["id"]}])
    result = analysis.analyze(path, profile_times=(.20, .25, .30))
    pair = result["comparisons"][0]
    assert pair["status"] == "matched"
    times = [row["time"] for row in pair["frames"]]
    assert .2 in times and max(times) == .24
    assert .25 not in times and .26 not in times and .3 not in times
    assert .26 in pair["unmatched_candidate_times"] and .3 in pair["unmatched_reference_times"]
    assert pair["profile_times"] == [.2, .24]
    assert pair["unavailable_requested_profile_times"] == [.25, .3]
    assert pair["frames"][0]["velocity_relative_l2"] is None
    assert all(row["velocity_relative_l2"] is not None for row in pair["frames"][1:])


def test_source_contract_and_path_escape_are_rejected(campaign):
    path, document, case, output, _, _ = campaign
    bad = deepcopy(document)
    bad["source_sha256"][continuation.CORE_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        analysis.admit_sources(bad)
    with pytest.raises(ValueError, match="leaves"):
        analysis.member(output, "../seed/re100.npz")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="outside"):
        analysis.main(["--campaign", str(path), "--output", str(path)])
    assert path.read_bytes() == original


def test_repeat_analysis_does_not_trust_a_previously_cached_but_corrupted_seed(campaign):
    path, document, case, output, seed, receipt = campaign
    assert analysis.analyze(path)["cases"][case["id"]]["admission"] == "verified_checkpoint"
    seed.write_bytes(seed.read_bytes() + b"deliberate cached-prefix corruption")
    report = analysis.analyze(path)
    assert report["cases"][case["id"]]["admission"] == "invalid"
    assert "SHA-256 mismatch" in report["cases"][case["id"]]["error"]


def test_frozen_production_plan_contains_all_16_matched_controls():
    path = analysis.ROOT / "tmp/cavity_settling/campaign.json"
    if not path.exists():
        pytest.skip("The local submission manifest is intentionally not a distributable test dependency")
    cases = json.loads(path.read_text())["cases"]
    pairs = analysis.plan_pairs(cases)
    assert len(pairs) == 16
    assert all(pair["candidate_id"] and pair["reference_id"] for pair in pairs)
    assert {kind: sum(pair["kind"] == kind for pair in pairs) for kind in ("time", "grid", "coupling")} == {
        "time": 9, "grid": 4, "coupling": 3}
