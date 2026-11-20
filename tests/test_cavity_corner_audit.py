import numpy as np
from flowmllab.cavity_corner_audit import vortex_peak, diagnostic_rows


def field(peak=True):
    x = np.linspace(0, 1, 101)
    X, Y = np.meshgrid(x, x)
    psi = np.maximum(0., .01 - (X-.08)**2 - (Y-.90)**2) if peak else -X*(1-X)*Y*(1-Y)
    return dict(x=x, y=x, psi=psi, u=np.ones_like(X), v=np.zeros_like(X))


def test_no_wall_point_misreported_as_vortex():
    value = vortex_peak(field(False))
    assert not value['resolved_positive_island']
    assert value['x_peak_L'] is None


def test_native_positive_island_and_exact_comparison():
    f = field(True)
    value = vortex_peak(f)
    assert value['resolved_positive_island']
    assert np.isclose(value['x_peak_L'], .08)
    assert np.isclose(value['y_peak_L'], .90)
    cases = {3200: {name: f for name in ('CFD', 'POD-DeepONet', 'Linear in log Re')}}
    for row in diagnostic_rows(cases):
        assert row['field_relative_l2'] == 0
        assert row['corner_relative_l2'] == 0
        assert row['peak_shift_L'] == 0
