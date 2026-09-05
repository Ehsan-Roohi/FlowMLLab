"""Lightweight gates for the Week 10.1 plot reader and retained figures."""
from pathlib import Path
import json
import unittest
import numpy as np
from qa.plot_deeponet_cylinder_contours import cell_zone, exterior_triangulation, CENTER

ROOT = Path(__file__).resolve().parents[1]

class ContourTests(unittest.TestCase):
    def test_first_zone_excludes_boundary_zone(self):
        columns, values = cell_zone(b'VARIABLES="X","Y"\nZONE I=2 F=POINT\n0 1\n1 2\nZONE I=1\n9 9\n')
        self.assertEqual(columns, ['X','Y'])
        np.testing.assert_array_equal(values, [[0,1],[1,2]])

    def test_nonfinite_and_truncated_rejected(self):
        for data in [b'VARIABLES="X","Y"\nZONE I=2\n0 nan\n1 2\n',
                     b'VARIABLES="X","Y"\nZONE I=3\n0 1\n1 2\n']:
            with self.subTest(data=data), self.assertRaises(ValueError):
                cell_zone(data)

    def test_solid_intersections_masked(self):
        theta=np.linspace(0,2*np.pi,25)[:-1]
        xy=np.column_stack([np.cos(theta),np.sin(theta)])*.2+CENTER
        tri=exterior_triangulation(xy)
        self.assertTrue(tri.mask.any())

    def test_retained_provenance_and_color_bounds(self):
        folder=ROOT/'results/abinitio_deeponet_cylinder'
        data=json.loads((folder/'contour_manifest.json').read_text())
        self.assertTrue(data['manifest_hashes_verified'])
        self.assertTrue(data['asynchronous_snapshots'])
        self.assertEqual(data['runs']['DeepONet']['metadata']['NOUT'],'95')
        self.assertEqual(data['runs']['Exact']['metadata']['NOUT'],'98')
        for field, values in data['fields'].items():
            self.assertTrue((folder/f'{field}_exact_deeponet.png').is_file())
            for label in ['Exact','DeepONet']:
                self.assertEqual(values[label]['out_of_color_range_native'],0)

if __name__=='__main__': unittest.main()
