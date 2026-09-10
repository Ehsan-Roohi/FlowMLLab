"""Offline design, numerics and retention tests; actual solver gates run on Unity."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np

from week13_independence import CASES, session, x_vertices, PolynomialField, temporal_gate
from run_week13_independence import prune


class TestIndependence(unittest.TestCase):
    def test_eight_one_factor_cases(self):
        self.assertEqual(len(CASES), 8)
        base = CASES[0]
        for c in CASES:
            self.assertAlmostEqual(round(.25/c['dt'])*c['dt'], .25)
            self.assertEqual(len(x_vertices(c)), c['nx']+1)
            self.assertTrue(np.all(np.diff(x_vertices(c)) > 0))
            changes = sum([c['nx'] != base['nx'] or c['ny'] != base['ny'],
                           c['order'] != base['order'], c['dt'] != base['dt']])
            self.assertLessEqual(changes, 1)

    def test_nested_mesh_and_identical_lid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for c in CASES:
                dest = root/c['label']
                session(dest, c)
                tree = ET.parse(dest/'cavity.xml')
                self.assertEqual(len(tree.findall('.//ELEMENT/Q')), c['nx']*c['ny'])
                self.assertEqual(tree.find('.//EXPANSIONS/E').get('NUMMODES'), str(c['order']+1))
                lid = tree.find(".//BOUNDARYCONDITIONS/REGION[@REF='1']/D[@VAR='u']")
                self.assertEqual(lid.get('VALUE'), '(x>0)*(x<1)')
                self.assertIn('Kinvis = 0.002', [p.text for p in tree.findall('.//PARAMETERS/P')])
                self.assertTrue(all(np.min(abs(x_vertices(c)-x)) < 1e-13
                                    for x in x_vertices(CASES[0])))

    def test_polynomial_paths_and_vorticity_at_tiny_scale(self):
        # psi = amplitude*x^2*(1-x)^2*y^2*(5-y)^2, exact closed-wall field.
        ex, ey = np.array([0., .3, 1.]), np.array([0., 2., 5.])
        x = np.unique(np.r_[np.linspace(0,.3,7), np.linspace(.3,1,7)])
        y = np.unique(np.r_[np.linspace(0,2,7), np.linspace(2,5,7)])
        xx, yy = np.meshgrid(x, y)
        for amplitude in [1., 1e-8, 1e-14]:
            f, g = xx**2*(1-xx)**2, yy**2*(5-yy)**2
            df, dg = 2*xx-6*xx**2+4*xx**3, 50*yy-30*yy**2+4*yy**3
            fields = dict(u=amplitude*f*dg, v=-amplitude*df*g, p=np.zeros_like(xx))
            poly = PolynomialField(x, y, fields, ex, ey, 6)
            for cx, cy in [(.2,1.1), (.55,3.7), (.3,2.)]:
                fx, gy = cx**2*(1-cx)**2, cy**2*(5-cy)**2
                exact = amplitude*fx*gy
                self.assertLess(abs(poly.psi(cx,cy)/exact-1), 1e-11)
                self.assertLess(abs(poly.psi(cx,cy,True)/exact-1), 1e-11)
                omega = amplitude*(fx*(50-60*cy+12*cy**2)+(2-12*cx+12*cx**2)*gy)
                observed = poly.value('u',cx,cy,dy=1)-poly.value('v',cx,cy,dx=1)
                self.assertLess(abs(observed-omega), amplitude*1e-9)

    def test_export_undersampling_rejected(self):
        x, y = np.linspace(0,1,3), np.linspace(0,5,3)
        p = PolynomialField(x,y,dict(u=np.zeros((3,3))), [0,1],[0,5],6)
        with self.assertRaisesRegex(ValueError,'undersamples'):
            p.value('u',.5,2.5)

    def snapshots(self):
        return [dict(time=t, vortices=[dict(x=.5,y=4-i,psi=(-1)**(i+1)*10**(-2*i),
                    omega_clockwise=(-1)**i*10**(-2*i)) for i in range(4)]) for t in (220,230,240)]

    def test_gate_checks_weak_vortex_and_vorticity(self):
        rows = self.snapshots()
        self.assertTrue(temporal_gate(rows)['passed'])
        rows[-1]['vortices'][-1]['psi'] *= 1.002
        self.assertFalse(temporal_gate(rows)['passed'])
        rows = self.snapshots()
        rows[-1]['vortices'][-1]['omega_clockwise'] *= 1.002
        self.assertFalse(temporal_gate(rows)['passed'])
        self.assertFalse(temporal_gate(rows[:2])['passed'])

    def test_gate_rejects_mismatched_times(self):
        rows = self.snapshots()
        rows[-1]['time'] = 241
        with self.assertRaisesRegex(ValueError,'ten'):
            temporal_gate(rows)

    def test_only_own_intermediates_pruned(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'new'
            root.mkdir()
            source = Path(temp)/'source.fld'
            source.write_text('immutable')
            records=[]
            for i, t in enumerate([220.,220.25,220.5,220.75]):
                chunk=root/f'step-{i:04d}'
                attempt=chunk/'attempt-00'
                attempt.mkdir(parents=True)
                (attempt/'cavity.fld').write_text('new fixture')
                (attempt/'solver.log').write_text('retain log')
                marker=chunk/'accepted.json'
                marker.write_text(json.dumps(dict(time=t,attempt='attempt-00')))
                records.append(marker)
            prune(root,records)
            self.assertEqual(source.read_text(),'immutable')
            self.assertTrue((records[0].parent/'attempt-00/cavity.fld').exists())
            self.assertFalse((records[1].parent/'attempt-00/cavity.fld').exists())
            self.assertTrue((records[1].parent/'attempt-00/solver.log').exists())
            self.assertTrue((records[-1].parent/'attempt-00/cavity.fld').exists())
            self.assertTrue((records[-2].parent/'attempt-00/cavity.fld').exists())
            prune(root,records)  # Idempotent.

    def test_pruning_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'new'
            root.mkdir()
            chunk=root/'step-0001'
            chunk.mkdir()
            marker=chunk/'accepted.json'
            marker.write_text(json.dumps(dict(time=220.25,attempt='../../outside')))
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                prune(root,[marker,marker,marker])


if __name__ == '__main__':
    unittest.main()
