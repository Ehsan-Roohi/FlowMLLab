"""Tests of the scientific contracts used in the scalar/field teaching lab."""
import numpy as np
import pytest

from flowmllab.cavity_lab2 import case_split, interpolate_rows, field_metrics, point_samples


def fixture_data():
    re = np.array([100, 150, 175, 200, 225, 250, 275, 300, 350, 375, 400.])
    split = np.array(["train", "train", "test", "train", "train", "train", "test", "train", "train", "test", "train"])
    x = np.linspace(0, 1, 9)
    xx, yy = np.meshgrid(x, x)
    return {"Re": re, "split": split, "x": x, "y": x,
            "u": np.stack([r*xx for r in re]), "v": np.stack([r*yy for r in re])}


def test_case_split_preserves_test_and_requires_bracketing():
    d = fixture_data()
    fit, val, test = case_split(d)
    np.testing.assert_array_equal(d["Re"][test], [175, 275, 375])
    np.testing.assert_array_equal(d["Re"][val], [225, 350])
    assert not np.any(fit & (val | test))
    with pytest.raises(ValueError, match="training"):
        case_split(d, (175, 350))
    with pytest.raises(ValueError, match="minimum"):
        case_split(d, (100, 350))


def test_interpolation_retains_endpoints_and_never_extrapolates():
    re = np.array([100., 200., 400.])
    fields = np.stack([np.full((3, 3), r/100) for r in re])
    np.testing.assert_allclose(interpolate_rows(100, re, fields), 1)
    np.testing.assert_allclose(interpolate_rows(400, re, fields), 4)
    np.testing.assert_allclose(interpolate_rows(250, re, fields), 2.5)
    with pytest.raises(ValueError, match="bracketing"):
        interpolate_rows(401, re, fields)


def test_sampler_never_includes_a_test_case_and_keeps_wall_adjacent_nodes():
    d = fixture_data()
    fit, _, _ = case_split(d)
    features, values = point_samples(d, fit, 100, 400, stride=4)
    expected = 2*(d["Re"][fit]-100)/300-1
    np.testing.assert_allclose(np.unique(features[:, 0]), expected)
    assert 1/8 in features[:, 1] and 7/8 in features[:, 2]
    recovered_re = (features[:, 0]+1)*150+100
    np.testing.assert_allclose(values[:, 0], recovered_re*features[:, 1])
    np.testing.assert_allclose(values[:, 1], recovered_re*features[:, 2])


def test_metrics_measure_wall_and_interior_separately_without_signed_cancellation():
    x = np.linspace(0, 1, 9)
    xx, yy = np.meshgrid(x, x)
    u, v = xx, -yy  # exactly divergence-free linear field
    exact = field_metrics(x, x, u, v, u, v)
    assert exact["vector_relative_l2"] == 0
    assert exact["divergence_core_rms"] < 1e-14
    up = u.copy(); up[-1] += .2
    changed = field_metrics(x, x, u, v, up, v)
    assert changed["wall_max"] == pytest.approx(.2)
    assert changed["vector_rms"] > 0
    assert changed["divergence_core_rms"] < 1e-14
