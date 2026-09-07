"""Guard the curated course entry points without running scientific models."""
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class CourseNavigationTests(unittest.TestCase):
    def test_curated_relative_links_resolve(self):
        # These entry points use simple inline links, not nested link targets.
        for name in ('README.md', 'COURSE_MAP.md', 'THEORY_GAP_MATRIX.md'):
            source = ROOT / name
            for target in re.findall(r'\]\(([^\s)]+)\)', source.read_text(encoding='utf-8')):
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                with self.subTest(document=name, target=target):
                    self.assertTrue((source.parent / unquote(parsed.path)).exists())

    def test_scattering_lab_is_discoverable_from_both_entry_points(self):
        target = 'notebooks/week10_1/W10_1_Collision_Map_Surrogate_Audit.ipynb'
        for name in ('README.md', 'COURSE_MAP.md'):
            with self.subTest(document=name):
                self.assertIn(target, (ROOT / name).read_text(encoding='utf-8'))

    def test_week72_is_discoverable_from_course_entry_points(self):
        target = 'notebooks/week07_2/README.md'
        for name in ('README.md', 'COURSE_MAP.md', 'THEORY_GAP_MATRIX.md'):
            with self.subTest(document=name):
                self.assertIn(target, (ROOT / name).read_text(encoding='utf-8'))

    def test_proposals_are_distinct_from_implemented_uq(self):
        matrix = (ROOT / 'THEORY_GAP_MATRIX.md').read_text(encoding='utf-8')
        self.assertIn('notebooks/week02_1/Probabilistic_UQ_CFD.ipynb', matrix)
        self.assertIn('## Proposed increments, not released lessons', matrix)


if __name__ == '__main__':
    unittest.main()
