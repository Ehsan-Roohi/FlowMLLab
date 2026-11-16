"""Physics, provenance and schema contracts for the bounded CPU shock lesson."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pytest

from flowmllab.kinetic_shock import (
    FIELDS, available_cases, collision_experiment, collision_targets,
    data_directory, distribution_moments, jump_states, load_case,
    load_manifest, load_observation_locations, maxwellian, profile_metrics,
    reduced_moments, stationary_flux_diagnostics, stationary_fluxes,
    transverse_counterexample, velocity_quadrature,
)


@contextmanager
def copied_case(case_id="m2_seed141001_fixed500"):
    """All temporary test artifacts stay in this owned external-drive dataset."""
    with tempfile.TemporaryDirectory(prefix=".schema-test-", dir=data_directory()) as tmp:
        root = Path(tmp)
        manifest = load_manifest()
        entry = manifest["cases"][case_id]
        target = root / entry["path"]
        target.parent.mkdir(parents=True)
        shutil.copyfile(data_directory() / entry["path"], target)
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        yield root, manifest, target


def rewrite_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def rebind_local_fixture(manifest, cid, path):
    raw = path.read_bytes()
    manifest["cases"][cid].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def test_rankine_hugoniot_conserves_all_three_fluxes():
    for mach in (1.0, 2.0, 5.0, 8.0):
        states = jump_states(mach)
        left = stationary_fluxes(**states["upstream"], q=0, sigma=0)
        right = stationary_fluxes(**states["downstream"], q=0, sigma=0)
        for name in left:
            np.testing.assert_allclose(left[name], right[name], rtol=5e-15)
        assert states["downstream"]["rho"] >= states["upstream"]["rho"]
    np.testing.assert_allclose(list(jump_states(1)["upstream"].values()),
                               list(jump_states(1)["downstream"].values()))
    for bad in (0, 0.9, np.nan, np.inf):
        with pytest.raises(ValueError):
            jump_states(bad)


def test_flux_uses_kinetic_stress_sign_and_energy_work():
    flux = stationary_fluxes(rho=2, u=3, T=4, q=-5, sigma=7)
    assert flux == {"mass": 6.0, "momentum": 33.0, "energy": 103.0}
    with pytest.raises(ValueError, match="positive"):
        stationary_fluxes(0, 1, 1, 0, 0)
    with pytest.raises(ValueError, match="steady"):
        stationary_flux_diagnostics(load_case("tube_bgk"), "final")


def test_spatial_metrics_are_signed_integrals_on_nonuniform_grid():
    x = np.array([0.0, 1.0, 3.0])
    reference = np.array([1.0, 2.0, 4.0])
    result = profile_metrics(x, reference + 1, reference)
    assert result["relative_l2_percent"] == pytest.approx(100 * np.sqrt(3 / 22.5))
    np.testing.assert_array_equal(result["signed_error"], [1, 1, 1])
    np.testing.assert_array_equal(result["signed_error_percent"], [25, 25, 25])
    assert result["global_scale"] == 4
    negative = profile_metrics(x, reference - 1, reference)
    np.testing.assert_array_equal(negative["signed_error"], [-1, -1, -1])
    with pytest.raises(ValueError, match="nonzero"):
        profile_metrics(x, np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        profile_metrics(x, reference, reference, global_scale=0)
    with pytest.raises(ValueError, match="increasing"):
        profile_metrics(x[::-1], reference, reference)
    with pytest.raises(ValueError, match="matched"):
        profile_metrics(x, reference[:-1], reference)


def test_equilibrium_full_velocity_moments():
    v, w = velocity_quadrature(40, 9)
    f = maxwellian(v, rho=1.2, u=0.3, T=1.0)
    m = distribution_moments(v, w, f)
    for field, expected in (("rho", 1.2), ("u", 0.3), ("T", 1.0)):
        assert m[field] == pytest.approx(expected, abs=2e-10)
    for field in ("q", "sigma", "m", "R", "Delta", "S"):
        assert abs(m[field]) < 2e-8
    targets = collision_targets(v, m, Pr=1)
    np.testing.assert_array_equal(targets["BGK"], targets["Shakhov"])


def test_positive_mixture_conservation_and_relaxation_identities():
    result = collision_experiment(order=40, cutoff=9, Pr=2 / 3, tau=1.7)
    assert result["input_positive"] and not result["is_shock_solution"]
    assert not result["clipping_applied"]
    for field, expected in result["analytic_mixture_moments"].items():
        assert result["moments"][field] == pytest.approx(expected, abs=7e-9)
    for name, q_ratio in (("BGK", -1), ("Shakhov", -2 / 3)):
        row = result["diagnostics"][name]
        assert max(abs(row["invariant_rate"])) < 1e-10
        assert row["q_rate_ratio"] == pytest.approx(q_ratio, abs=1e-10)
        assert row["sigma_rate_ratio"] == pytest.approx(-1, abs=1e-10)
        assert row["q_rate_absolute_error"] < 1e-10
        assert row["stress_rate_absolute_error"] < 1e-10
        assert not row["clipping_applied"] and not row["conservation_correction_applied"]
    assert result["diagnostics"]["BGK"]["negative_nodes"] == 0
    shakhov = result["diagnostics"]["Shakhov"]
    assert shakhov["minimum_target"] < 0
    assert shakhov["negative_nodes"] > 0 and shakhov["negative_mass"] > 0
    assert 0 < shakhov["negative_target_weight_fraction"] < 1


def test_order_and_velocity_cutoff_are_distinct_errors():
    coarse = collision_experiment(12, 7)
    resolved = collision_experiment(32, 7)
    truncated = collision_experiment(32, 5)
    assert abs(coarse["moments"]["q"] + 1.071) > 0.01
    assert abs(resolved["moments"]["q"] + 1.071) < 3e-5
    assert abs(truncated["moments"]["q"] + 1.071) > 0.01
    assert (resolved["diagnostics"]["Shakhov"]["q_rate_absolute_error"]
            < coarse["diagnostics"]["Shakhov"]["q_rate_absolute_error"] / 1000)
    unity_pr = collision_experiment(16, 7, Pr=1)
    np.testing.assert_array_equal(unity_pr["targets"]["BGK"], unity_pr["targets"]["Shakhov"])
    for order in (3, 49, 4.5):
        with pytest.raises(ValueError, match="order"):
            velocity_quadrature(order)
    with pytest.raises(ValueError, match="cutoff"):
        velocity_quadrature(10, 21)
    with pytest.raises(ValueError, match="Pr"):
        collision_experiment(12, 7, Pr=0)


def test_g_h_do_not_identify_R_and_Delta_but_do_identify_S():
    result = transverse_counterexample()
    a, b = result["A"], result["B"]
    assert "R" not in result["without_k"] and "Delta" not in result["without_k"]
    for field in ("rho", "u", "T", "q", "sigma", "m", "S"):
        assert a[field] == pytest.approx(b[field], abs=3e-14)
    assert a["R"] == pytest.approx(0, abs=1e-12)
    assert a["Delta"] == pytest.approx(0, abs=1e-12)
    assert b["R"] == pytest.approx(-2 / 3, abs=1e-12)
    assert b["Delta"] == pytest.approx(2, abs=1e-12)
    for moments in (a, b):
        assert moments["S"] == pytest.approx(moments["R"] + moments["Delta"] / 3, abs=1e-13)
    with pytest.raises(ValueError, match="k must match"):
        reduced_moments(result["vx"], result["weights"], result["g"], result["h"], np.ones(2))


def test_reduced_moments_match_full_three_velocity_integrals():
    n, cutoff = 20, 7
    v, w = velocity_quadrature(n, cutoff)
    f = 0.7 * maxwellian(v, u=0.8, T=0.7) + 0.3 * maxwellian(v, u=-1.2, T=1.4)
    z, wa = np.polynomial.legendre.leggauss(n)
    vx, wa = cutoff * z, cutoff * wa
    transverse_weights = wa[:, None] * wa[None, :]
    radial2 = vx[:, None]**2 + vx[None, :]**2
    ft = f.reshape(n, n, n)
    g = np.sum(ft * transverse_weights, axis=(1, 2))
    h = np.sum(ft * transverse_weights * radial2, axis=(1, 2))
    k = np.sum(ft * transverse_weights * radial2**2, axis=(1, 2))
    reduced = reduced_moments(vx, wa, g, h, k)
    full = distribution_moments(v, w, f)
    for field in FIELDS:
        assert reduced[field] == pytest.approx(full[field], abs=2e-12)


def test_all_retained_cases_have_exact_grids_fields_and_positive_bulk_state():
    cases = available_cases()
    assert len(cases) == 12
    for cid in cases:
        case = load_case(cid)
        assert np.all(np.diff(case.x) > 0)
        assert case.metadata["clipping_applied"] is False
        for method in case.methods:
            assert case.profiles[method].shape == (len(case.x), len(case.columns))
            assert np.all(case.field(method, "rho") > 0)
            assert not case.profiles[method].flags.writeable
        if "S" in case.columns:
            for method in case.methods:
                np.testing.assert_allclose(case.field(method, "S"),
                                           case.field(method, "R") + case.field(method, "Delta") / 3,
                                           rtol=2e-10, atol=2e-10)
    supervised = load_case("supervised_s0_current")
    assert len(supervised.x) == 399
    assert "five_field_S0" in supervised.methods
    assert supervised.metadata["five_field_reference_grid_and_values_exactly_match_eight_moment_assessment"]
    assert supervised.metadata["dropped_equation_nodes_outside_assessment_coverage"] == 2
    assert supervised.metadata["paper_admission"] is False
    assert supervised.metadata["independent_test_claim"] is False
    assert all(supervised.metadata["contamination"].values())


def test_independent_m2_same_reference_and_stage_metrics_match_saved_audit():
    root = data_directory()
    rec = json.loads((root / "upstream/data/independent_m2/evaluation/RESULT.json").read_text())
    baseline = load_case("m2_seed141001_adam20000")
    for seed in (141001, 141002, 141003):
        for stage in ("adam20000", "ssb500", "fixed500"):
            case = load_case(f"m2_seed{seed}_{stage}")
            np.testing.assert_array_equal(case.x, baseline.x)
            np.testing.assert_array_equal(case.profiles["reference"], baseline.profiles["reference"])
            record = next(r for r in rec["stage_records"] if r["seed"] == seed and r["stage"] == stage)
            for field in ("q", "sigma"):
                error = profile_metrics(case.x, case.field("prediction", field), case.field("reference", field))
                expected = record["windows"]["core"]["errors"][field]["relative_L2_percent"]
                assert error["relative_l2_percent"] == pytest.approx(expected, rel=1e-10)
            flux = stationary_flux_diagnostics(case, "prediction")
            with np.load(root / case.metadata["path"], allow_pickle=False) as a:
                np.testing.assert_allclose(np.column_stack(list(flux["defect_percent"].values())),
                                           a["flux_defect_percent"], atol=2e-10)


def test_exact_observations_and_no_independent_test_claim():
    obs = load_observation_locations()
    assert len(obs["macro_x"]) == 160 and len(obs["moment_x"]) == 42
    assert obs["scalar_observations"] == 160 * 3 + 42 * 5
    assert obs["independent_test_claim"] is False
    assert obs["reference_used_for_scales_and_acceptance"] is True


def test_hash_corruption_is_rejected_before_npz_read():
    with copied_case() as (root, manifest, path):
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="hash/size mismatch"):
            load_case("m2_seed141001_fixed500", root)


@pytest.mark.parametrize("change,error", [
    ("units", "units"), ("case_id", "identity"), ("stage", "seed/stage"),
    ("x_endpoints", "coverage"), ("x_sha256", "coordinate hash"), ("columns", "schema"),
])
def test_malformed_case_metadata_rejected(change, error):
    cid = "m2_seed141001_fixed500"
    with copied_case(cid) as (root, manifest, _):
        entry = manifest["cases"][cid]
        entry[change] = {"units": "dimensional", "case_id": "tube_bgk", "stage": "adam20000",
                         "x_endpoints": [-1, 1], "x_sha256": "0" * 64,
                         "columns": ["rho", "u", "T", "sigma", "q"]}[change]
        rewrite_manifest(root, manifest)
        with pytest.raises(ValueError, match=error):
            load_case(cid, root)


def test_malformed_profile_shape_rejected_even_with_rebound_fixture_hash():
    cid = "m2_seed141001_fixed500"
    with copied_case(cid) as (root, manifest, path):
        with np.load(path, allow_pickle=False) as a:
            arrays = {key: a[key].copy() for key in a.files}
        arrays["fields"] = arrays["fields"][:-1]
        np.savez(path, **arrays)
        rebind_local_fixture(manifest, cid, path)
        rewrite_manifest(root, manifest)
        with pytest.raises(ValueError, match="shape mismatch"):
            load_case(cid, root)


def test_malformed_manifest_and_unknown_case_are_rejected():
    with copied_case() as (root, manifest, _):
        manifest["upstream_commit"] = "unverified"
        rewrite_manifest(root, manifest)
        with pytest.raises(ValueError, match="upstream commit"):
            load_manifest(root)
    with pytest.raises(ValueError, match="unknown"):
        load_case("mach2_transient")
