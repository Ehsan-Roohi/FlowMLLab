import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cfd", Path(__file__).parents[1] / "qa/prepare_week13_cfd.py")
cfd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfd)


def test_matrix():
    assert len(cfd.CASES) == len(set(cfd.CASES)) == 16
    for re, depth in cfd.CASES:
        for nx in cfd.GRIDS:
            assert abs(nx * depth - round(nx * depth)) < 1e-10
            data = cfd.files(re, depth, nx, 20000, 8)
            assert len(data) == 9
            assert 'simulationType laminar' in data['constant/turbulenceProperties']
            assert f'{1/re:.17g}' in data['constant/transportProperties']
            assert f'({nx} {round(nx*depth)} 1)' in data['system/blockMeshDict']
            assert 'startFrom latestTime' in data['system/controlDict']
            assert 'value uniform (1 0 0)' in data['0/U']
            assert 'div(phi,U) Gauss linear' in data['system/fvSchemes']
