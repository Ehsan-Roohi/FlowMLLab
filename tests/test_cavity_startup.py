"""Physical-time and provenance checks for the additive cavity movie lab."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from flowmllab.cavity_startup import (
    observation_times, simulate_startup, validate_trajectory, save_trajectory,
    load_trajectory, compare_trajectories, common_field_bounds, frame_fields,
)
from flowmllab.pressure_velocity import coupling_step, run_cavity, divergence
from flowmllab.pressure_velocity_experiments import ExperimentalConfig


@pytest.fixture(scope="module")
def small_case():
    config = ExperimentalConfig(method="pimple", reynolds=100, cells=8,
        dt=.02, end_time=.1, outer_correctors=2, pressure_correctors=2)
    return simulate_startup(config, observation_times(.1, .02))


def test_frames_are_exact_physical_steps_and_preserve_existing_solver(small_case):
    config = ExperimentalConfig(**small_case["config"])
    retained = run_cavity(config)
    for name in ("u_faces", "v_faces", "p"):
        np.testing.assert_allclose(small_case[name][-1], retained[name], atol=2e-14, rtol=2e-13)
    np.testing.assert_array_equal(small_case["times"], observation_times(.1, .02))
    assert len(small_case["diagnostics"]) == 5
    assert not np.shares_memory(small_case["u_faces"][0], small_case["u_faces"][-1])


def test_noncommensurate_frame_schedule_shortens_real_steps():
    config = ExperimentalConfig(method="piso", reynolds=400, cells=6,
                                dt=.03, end_time=.08)
    times = np.array([0., .05, .08])
    result = simulate_startup(config, times)
    np.testing.assert_allclose(result["diagnostics"][:, 2], [.03, .02, .03], atol=1e-15)
    u, v, p = np.zeros((6, 7)), np.zeros((7, 6)), np.zeros((6, 6))
    for dt in (.03, .02, .03):
        u, v, p, _ = coupling_step(u, v, p, config, dt=dt)
    np.testing.assert_allclose(result["u_faces"][-1], u, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(result["p"][-1], p, rtol=1e-13, atol=1e-15)


def test_native_gauge_continuity_and_instantaneous_streamfunction(small_case):
    for k in range(len(small_case["times"])):
        u, v = small_case["u_faces"][k], small_case["v_faces"][k]
        assert np.max(np.abs(divergence(u, v, 1 / 8))) < 1e-10
        assert abs(small_case["p"][k].mean()) < 1e-12
        fields = frame_fields(small_case, k)
        np.testing.assert_allclose(np.diff(fields["psi"], axis=0) * 8, u, atol=1e-13)
        np.testing.assert_allclose(-np.diff(fields["psi"], axis=1) * 8, v, atol=1e-13)


def test_roundtrip_and_corrupted_field_rejection(tmp_path, small_case):
    path = tmp_path / "startup.npz"
    digest = save_trajectory(small_case, path)
    loaded = load_trajectory(path)
    assert len(digest) == 64
    for name in ("times", "u_faces", "v_faces", "p", "diagnostics"):
        np.testing.assert_array_equal(loaded[name], small_case[name])
    damaged = deepcopy(loaded)
    damaged["p"][-1] += .1
    with pytest.raises(ValueError, match="gauge"):
        validate_trajectory(damaged)
    damaged = deepcopy(loaded)
    damaged["u_faces"][-1, 1, 1] += .01
    with pytest.raises(ValueError, match="continuity"):
        validate_trajectory(damaged)
    damaged = deepcopy(loaded)
    damaged["times"][1:-1] += 1e-4
    with pytest.raises(ValueError, match="actual timestep"):
        validate_trajectory(damaged)
    damaged = deepcopy(loaded)
    damaged["diagnostics"][0, 0] = 0
    with pytest.raises(ValueError, match="identifiers"):
        validate_trajectory(damaged)


def test_schedule_rejects_iteration_movies_and_incomplete_end_times():
    with pytest.raises(ValueError, match="SIMPLE"):
        simulate_startup(ExperimentalConfig(method="simple", reynolds=100), [0, 1])
    config = ExperimentalConfig(method="pimple", reynolds=100, end_time=1)
    with pytest.raises(ValueError, match="increase"):
        simulate_startup(config, [0, .2, .2, 1])
    with pytest.raises(ValueError, match="last frame"):
        simulate_startup(config, [0, .2])
    with pytest.raises(ValueError, match="positive"):
        observation_times(1, 0)
    np.testing.assert_allclose(observation_times(.25, .1), [0, .1, .2, .25])


def test_matched_sensitivity_and_full_range_bounds(small_case):
    same = compare_trajectories(small_case, small_case)
    assert same["frames"][0]["velocity_relative_l2"] is None
    assert all(row["velocity_rms_difference"] == 0 for row in same["frames"])
    fine = simulate_startup(replace(ExperimentalConfig(**small_case["config"]), cells=12), small_case["times"])
    report = compare_trajectories(small_case, fine)
    assert report["frames"][-1]["velocity_rms_difference"] > 0
    with pytest.raises(ValueError, match="coarser"):
        compare_trajectories(fine, small_case)
    bounds = common_field_bounds([small_case, fine])
    for case in (small_case, fine):
        for k in range(len(case["times"])):
            fields = frame_fields(case, k)
            for name, limits in bounds.items():
                assert fields[name].min() >= limits["vmin"]
                assert fields[name].max() <= limits["vmax"]
    with pytest.raises(ValueError, match="At least one"):
        common_field_bounds([])


def test_solve_command_preserves_existing_scientific_result(tmp_path):
    from argparse import Namespace
    from qa.build_cavity_startup import solve
    path = tmp_path / "re100.npz"
    path.write_bytes(b"retained complete evidence")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        solve(Namespace(output=tmp_path, reynolds=100))
    assert path.read_bytes() == b"retained complete evidence"
