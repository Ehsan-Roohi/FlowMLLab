"""Offline integrity tests, not a CFD validation or substitute for compute gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from prepare_week13_nektar import make_case
from run_week13_nektar_refinement import accepted_source, preserve_time
from run_week13_nektar_reynolds import field_time, run, sha


class TestMappedRestart(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_clock_preserves_coefficients(self):
        fld = self.root / 'mapped.fld'
        fld.write_text('<NEKTAR><Metadata><Time>0</Time></Metadata><ELEMENTS>abc123</ELEMENTS></NEKTAR>')
        preserve_time(fld, 260)
        self.assertEqual(field_time(fld), 260)
        self.assertEqual(ET.parse(fld).find('ELEMENTS').text, 'abc123')

    def test_missing_clock_and_invalid_time(self):
        fld = self.root / 'mapped.fld'
        fld.write_text('<NEKTAR><ELEMENTS>xyz</ELEMENTS></NEKTAR>')
        preserve_time(fld, 260)
        self.assertEqual(field_time(fld), 260)
        before = fld.read_bytes()
        for value in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                preserve_time(fld, value)
        self.assertEqual(fld.read_bytes(), before)

    def source(self):
        chunk = self.root / 'chunk-0259'
        attempt = chunk / 'attempt-00'
        attempt.mkdir(parents=True)
        spec = dict(re=1000, order=6, depth_over_width=5,
                    corner_convention='stationary_endpoints')
        (self.root / 'campaign.json').write_text(json.dumps(spec))
        (attempt / 'cavity.fld').write_text('<NEKTAR><Metadata><Time>260</Time></Metadata></NEKTAR>')
        marker = dict(attempt='attempt-00', time=260, clean_exit=True)
        for key, name in [('field_sha256', 'cavity.fld'),
                          ('primitive_vtu_sha256', 'cavity.vtu'),
                          ('vorticity_vtu_sha256', 'cavity-vorticity.vtu')]:
            if not (attempt / name).exists():
                (attempt / name).write_text('source fixture')
            marker[key] = sha(attempt / name)
        (chunk / 'accepted.json').write_text(json.dumps(marker))
        return chunk, attempt

    def test_source_verified_without_changes(self):
        chunk, attempt = self.source()
        before = {p.name: p.read_bytes() for p in attempt.iterdir()}
        self.assertEqual(accepted_source(chunk, 1000)[0], attempt.resolve())
        self.assertEqual(before, {p.name: p.read_bytes() for p in attempt.iterdir()})
        with self.assertRaisesRegex(ValueError, 'physics'):
            accepted_source(chunk, 500)

    def test_corrupted_source_rejected(self):
        chunk, attempt = self.source()
        (attempt / 'cavity.vtu').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            accepted_source(chunk, 1000)

    def test_attempt_escape_rejected(self):
        chunk, _ = self.source()
        marker = json.loads((chunk / 'accepted.json').read_text())
        marker['attempt'] = '../outside'
        (chunk / 'accepted.json').write_text(json.dumps(marker))
        with self.assertRaisesRegex(ValueError, 'outside'):
            accepted_source(chunk, 1000)

    def test_nested_mesh_and_physics(self):
        vertices = []
        for nx, ny in ((16, 80), (32, 160)):
            dest = self.root / str(nx)
            make_case(dest, re=1000, order=6, nx=nx, ny=ny,
                      dt=.00025, corner_convention='stationary_endpoints')
            tree = ET.parse(dest / 'cavity.xml')
            vertices.append({v.text for v in tree.findall('.//VERTEX/V')})
            self.assertEqual(tree.find('.//EXPANSIONS/E').get('NUMMODES'), '7')
            self.assertIn('Kinvis = 0.001', [v.text for v in tree.findall('.//PARAMETERS/P')])
        self.assertTrue(vertices[0].issubset(vertices[1]))

    def test_runner_starts_at_260_not_zero(self):
        fld = self.root / 'initial.fld'
        fld.write_text('<NEKTAR><Metadata><Time>260</Time></Metadata></NEKTAR>')
        gate = self.root / 'gate.json'
        gate.write_text(json.dumps(dict(gate_passed=True, time_match_verified=True,
            boundary_operability_passed=True, corner_convention='stationary_endpoints',
            Re=100, orders=[6], dt=[.00025],
            scope='short_time_restart_operability_and_agreement_not_steady_accuracy')))
        image = self.root / 'image.sif'
        image.write_text('fixture')
        initial = self.root / 'initial.json'
        initial.write_text(json.dumps(dict(re=1000, nx=32, ny=160, order=6, dt=.00025,
            corner_convention='stationary_endpoints', time=260, field=str(fld),
            field_sha256=sha(fld), operability_passed=True)))
        args = SimpleNamespace(re=1000, nx=32, ny=160, order=6, dt=.00025,
            corner_convention='stationary_endpoints', seconds=500, gate=str(gate),
            image=str(image), commit='test', output=str(self.root / 'run'),
            initial_state=str(initial), end_time=261)
        import subprocess
        with patch('run_week13_nektar_reynolds.run_process', side_effect=subprocess.TimeoutExpired('test', 1)):
            self.assertEqual(run(args), 75)
        self.assertFalse((self.root / 'run/chunk-0000').exists())
        session = ET.parse(self.root / 'run/chunk-0260/attempt-00/cavity.xml')
        self.assertEqual(session.find(".//FUNCTION[@NAME='InitialConditions']/F").get('FILE'), str(fld.resolve()))
        self.assertFalse((self.root / 'run/chunk-0260/accepted.json').exists())
        fld.write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            run(args)


if __name__ == '__main__':
    unittest.main()
