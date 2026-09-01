import csv
from pathlib import Path
import unittest

import numpy as np

from flowmllab.demo import available_blind_reynolds, load_blind_demo_case


ROOT = Path(__file__).resolve().parents[1]


class BlindDemoTests(unittest.TestCase):
    def test_demo_exposes_only_frozen_blind_cases(self):
        self.assertEqual(available_blind_reynolds(ROOT), (175.0, 275.0, 375.0))

    def test_demo_metrics_match_machine_readable_evidence(self):
        evidence_path = ROOT / "results" / "pod_deeponet" / "deeponet_metrics.csv"
        with evidence_path.open(newline="", encoding="utf-8") as stream:
            ensemble = {
                float(row["Re"]): row
                for row in csv.DictReader(stream)
                if row["method"] == "three-seed POD-DeepONet ensemble"
            }

        for reynolds in available_blind_reynolds(ROOT):
            case = load_blind_demo_case(reynolds, ROOT)
            row = ensemble[reynolds]
            self.assertAlmostEqual(case.relative_l2_uv, float(row["relative_L2_uv"]), places=14)
            self.assertAlmostEqual(
                case.maximum_vector_error,
                float(row["max_vector_error"]),
                places=14,
            )
            self.assertAlmostEqual(case.wall_rms_error, float(row["wall_rms_error"]), places=14)
            self.assertAlmostEqual(case.relative_l2_p, float(row["relative_L2_p"]), places=14)
            self.assertAlmostEqual(case.mae_p, float(row["MAE_p"]), places=14)
            self.assertAlmostEqual(
                case.maximum_pressure_error,
                float(row["max_p_error"]),
                places=14,
            )
            self.assertTrue(np.isfinite(case.reference_speed).all())
            self.assertTrue(np.isfinite(case.prediction_speed).all())
            self.assertTrue(np.isfinite(case.prediction_p).all())
            self.assertAlmostEqual(float(np.mean(case.prediction_p)), 0.0, places=14)


if __name__ == "__main__":
    unittest.main()
