"""Tests for the Week 7.4 trajectory-split protocol."""
from pathlib import Path

import numpy as np
import pytest

from flowmllab import diverse_wake_pretraining as dw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/week07_4_wakes"


def test_retained_split_is_disjoint_and_complete():
    manifest = dw.load_manifest(DATA)
    split = manifest["split_contract"]
    assert len(split["development"]) == 11
    assert split["validation"] == [105]
    assert split["test"] == [95, 115, 125, 135]


def test_case_shapes_and_finite_values():
    if not (DATA / "re095.npz").is_file():
        pytest.skip(
            "Week 7.4 data are optional; download release week07-4-wakes-v1 "
            "or run dw.ensure_release_data(DATA)."
        )
    fields, lift, times = dw.load_case(DATA, 95)
    assert fields.shape == (251, 32, 78) and lift.shape == times.shape == (251,)
    assert np.all(np.diff(times) > 0)


def test_normalized_rmse_and_trajectory_summary():
    reference = np.array([-1.0, 1.0])
    assert np.isclose(dw.normalized_rmse(reference + 0.5, reference), 50.0)
    records = []
    for reynolds, base in ((95, 1.0), (115, 3.0)):
        for seed in (0, 1):
            records.append({"method": "m", "k": 2, "reynolds": reynolds,
                            "seed": seed, "nrmse": base + seed})
    summary = dw.summarize_trajectory_records(records)["m"][2]
    assert summary["mean"] == 2.5 and summary["n_trajectories"] == 2
    assert np.isclose(summary["std_across_trajectories"], np.sqrt(2.0))


def test_overlap_is_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"split_contract":{"development":[1],"validation":[1],"test":[]},"cases":[{},{}]}')
    with pytest.raises(ValueError):
        dw.load_manifest(tmp_path)
