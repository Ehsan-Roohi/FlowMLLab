"""Small reproducibility gates for the reading companion, not PINN training."""
import unittest
from qa.build_week04_2_lecture import EQS, ROOT, TABLES, analytic_preflight


class PINNCavityLectureTests(unittest.TestCase):
    def test_analytic_preflight(self):
        result = analytic_preflight()
        self.assertLess(result['kovasznay_max_residual'], 1e-12)
        self.assertLess(result['lifting_max_wall_error'], 1e-12)
        self.assertEqual(result['cfd_archive_sha256'],
                         '09b96b744ee4d18126d8dcc92feb60e128774a1b4d41bb3d8c90a63ccfbabc36')

    def test_source_directives_and_evidence_boundaries(self):
        source = (ROOT/'lectures/source/week04_2_pinn_cavity.md').read_text(encoding='utf-8')
        for line in source.splitlines():
            if line.startswith('@equation '): self.assertIn(line.split()[1], EQS)
            if line.startswith('@table '): self.assertIn(line.split()[1], TABLES)
            if line.startswith('@figure '): self.assertIn(line.split()[1], ['cavity','graph','lid'])
        self.assertEqual(source.count('@equation '), 10)
        self.assertEqual(source.count('@figure '), 3)
        self.assertIn('SOAP_STEPS=0', source)
        self.assertIn('not a newly trained PINN', source)
        self.assertIn('fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b', source)


if __name__ == '__main__':
    unittest.main()
