"""Native CFD evidence must reproduce the retained solver/reference metrics."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from flowmllab import cavity_validation as cv

ROOT = Path(__file__).resolve().parents[1]


def test_native_cfd_reproduces_archived_metrics_and_wall_values():
    cases = cv.load_evidence(ROOT)
    assert len(cases) == 8
    best = {c['Re']: c for c in cv.finest(cases)}
    assert set(best) == {100, 400, 1000}
    expected = {100: (.003953841762187994, .0326071801663441),
                400: (.006019387627407963, .008972331836921303)}
    # The loader independently reproduces every raw archive metric. Check fixed
    # published values for the two distinct native-grid extraction paths here.
    for re in (100, 400):
        got = tuple(best[re]['scores'][c]['screened_relative_l2'] for c in ('u', 'v'))
        np.testing.assert_allclose(got, expected[re], rtol=1e-11)
    for c in cases:
        for component, endpoints in [('u', [0, 1]), ('v', [0, 0])]:
            coord, values = c['profiles'][component]
            np.testing.assert_array_equal(coord[[0, -1]], [0, 1])
            np.testing.assert_array_equal(values[[0, -1]], endpoints)
    assert best[400]['scores']['v']['published_relative_l2'] > .14
    assert best[400]['scores']['v']['screened_relative_l2'] < .01


def test_modified_native_archive_is_rejected_before_plotting(tmp_path):
    relative = Path('results/week01_2_pressure_velocity/simple-re100-n32.npz')
    dest = tmp_path / relative
    dest.parent.mkdir(parents=True)
    shutil.copy2(ROOT / relative.with_suffix('.json'), dest.with_suffix('.json'))
    dest.write_bytes((ROOT / relative).read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='Retained CFD archive changed'):
        cv.load_evidence(tmp_path)
