import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "common/deeponet_data_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_variable_query_grids_pass_but_first_case_reuse_fails():
    ids = np.repeat(["a", "b"], 3)
    x = np.array([0., .4, 1., 0., .45, 1.])
    assert audit.audit_pairs(ids, ids, x, x)["rows"] == 6
    with pytest.raises(ValueError, match="source target"):
        audit.audit_pairs(ids, ids, np.tile(x[:3], 2), x)


def test_correct_shapes_do_not_protect_against_case_permutation():
    ids = np.repeat(["a", "b"], 3)
    xy = np.tile([[0., 0.], [.5, .5], [1., 1.]], (2, 1))
    assert audit.audit_pairs(ids, ids, xy, xy)["rows"] == 6
    with pytest.raises(ValueError, match="case IDs"):
        audit.audit_pairs(ids[::-1], ids, xy, xy)


@pytest.mark.parametrize("bad", [np.array([0., np.nan]), np.array([0., np.inf])])
def test_nonfinite_coordinates_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        audit.audit_pairs([0, 0], [0, 0], bad, bad)


def test_broadcasting_cannot_hide_shape_error():
    with pytest.raises(ValueError, match="dimensions"):
        audit.audit_pairs([0, 0], [0, 0], np.zeros((2, 1)), np.zeros(2))


def test_fixed_sensor_contract():
    assert audit.audit_fixed_sensors(np.tile(np.linspace(0, 1, 41), (2, 1)))
    with pytest.raises(ValueError, match="sensor"):
        audit.audit_fixed_sensors([[0., .4, 1.], [0., .45, 1.]])


def test_split_keys_represent_cases():
    assert audit.audit_case_splits([100, 200], [225], [175])
    with pytest.raises(ValueError, match="overlap"):
        audit.audit_case_splits([100, 175], [225], [175])
