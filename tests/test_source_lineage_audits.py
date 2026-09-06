import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from qa.audit_cylinder_source import compare_import, read_ordered_point
from qa.audit_nozzle_archives import audit


class SourceLineageTests(unittest.TestCase):
    def test_week71_source_and_html_keep_scaling_note_and_figures(self):
        root = Path(__file__).resolve().parents[1]
        name = 'W7_1_Hypersonic_Rarefied_Cylinder_DeepONet'
        notebook = json.loads((root / 'notebooks/week07_1' / (name + '.ipynb')).read_text(encoding='utf-8'))
        prose = next(c['source'] for c in notebook['cells'] if c['id'] == 'f1171a5c9061')
        self.assertEqual(prose.count('$$'), 2)
        self.assertIn('**no freestream division**', prose)
        html = (root / 'docs/notebooks' / (name + '.html')).read_text(encoding='utf-8')
        self.assertIn('<strong>no freestream division</strong>', html)
        self.assertIn('not newly sealed tests', html)
        self.assertGreaterEqual(html.count('src="data:image/png;base64,'), 2)
        self.assertNotIn('FigureCanvasAgg is non-interactive', html)

    def test_retained_source_audit_matches_manifest(self):
        folder = Path(__file__).resolve().parents[1] / 'data/hypersonic_cylinder'
        report = json.loads((folder / 'source_audit.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual(report['source_sha256'], manifest['source_archive_sha256'])
        self.assertEqual(report['derivative_sha256'], manifest['artifact_sha256'])
        self.assertEqual(report['source_columns_in_target_order'], ['MA', 'TOV', 'P'])
        self.assertFalse(report['freestream_division_in_course_import'])
        self.assertEqual(report['matched_retained_rows'], 44500)
        self.assertEqual(len(report['cases']), 20)
        self.assertEqual(sum(c['matched_retained_rows'] for c in report['cases']), 44500)

    def test_source_names_control_target_order(self):
        values = np.array([[200., 5., 1.2], [300., 4., 2.]])
        self.assertEqual(compare_import(['TOV', 'MA', 'P'], values, np.array([1]),
                                        np.array([[4., 300., 2.]], dtype=np.float32)), 1)

    def test_freestream_rescaling_does_not_pass_source_identity(self):
        with self.assertRaises(ValueError):
            compare_import(['MA', 'TOV', 'P'], np.array([[5., 200., 1.2]]),
                           np.array([0]), np.array([[5., 1., 1.]], dtype=np.float32))

    def read_fixture(self, zone, data):
        memory = io.BytesIO()
        with zipfile.ZipFile(memory, 'w') as archive:
            archive.writestr('case.dat', 'TITLE="fixture"\nVARIABLES="MA" "TOV" "P"\n' + zone + '\n' + data)
        memory.seek(0)
        with zipfile.ZipFile(memory) as archive:
            return read_ordered_point(archive, 'case.dat')

    def test_ordered_point_parser(self):
        names, values, grid = self.read_fixture('ZONE I=2, J=1, DATAPACKING=POINT', '5 200 1\n6 210 2\n')
        self.assertEqual(names, ['MA', 'TOV', 'P'])
        self.assertEqual(grid, [2, 1])
        self.assertEqual(values.shape, (2, 3))

    def test_short_zone_is_rejected(self):
        with self.assertRaises(ValueError):
            self.read_fixture('ZONE I=2, J=2, DATAPACKING=POINT', '5 200 1\n6 210 2\n')

    def test_block_zone_is_not_silently_read_as_point(self):
        with self.assertRaises(ValueError):
            self.read_fixture('ZONE I=2, J=1, DATAPACKING=BLOCK', '5 6 200\n210 1 2\n')

    def test_nozzle_byte_identity_is_separate_from_line_endings(self):
        expected = b'header\n1 2 3\n'
        digest = hashlib.sha256(expected).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('exact/P=16.dat', expected)
                archive.writestr('windows/P=16.dat', expected.replace(b'\n', b'\r\n'))
                archive.writestr('changed/P=16.dat', b'header\n1 2 4\n')
                archive.writestr('solver.f90', '! candidate, not proof of producing source')
                archive.writestr('untrusted.py', "raise RuntimeError('do not execute')")
            result = audit(path, {'P=16.dat': digest})
        self.assertEqual(len(result['exact_published_snapshot_copies']), 1)
        self.assertEqual(len(result['published_snapshot_copies_after_CRLF_to_LF_only']), 1)
        self.assertEqual(len(result['same_basename_different_bytes']), 1)
        self.assertEqual(result['candidate_solver_or_input_entries'], ['solver.f90'])


if __name__ == '__main__':
    unittest.main()
