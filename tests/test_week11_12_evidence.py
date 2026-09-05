"""Checks for retained, real-research classroom evidence (no training needed)."""
from pathlib import Path
import csv
import hashlib
import json
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ResearchEvidenceTests(unittest.TestCase):
    def test_six_fixed_checkpoint_runs(self):
        folder=ROOT/'results/week11_research'
        report=json.loads((folder/'research_manifest.json').read_text())
        self.assertEqual(len(report['runs']),6)
        self.assertEqual(len(set(report['ids'])),6)
        self.assertFalse(report['new_training'])
        self.assertFalse(report['new_solver_runs'])
        self.assertFalse(report['human_accuracy'])
        self.assertEqual(report['thresholds'],{'shock':.97,'vortex_core':.85})
        for name,sha in report['figure_sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha)

    def test_dsmc_provenance_and_seed_coverage(self):
        folder=ROOT/'results/week12_research'
        report=json.loads((folder/'research_manifest.json').read_text())
        self.assertEqual(report['recomputed_scores'],80)
        self.assertEqual(report['archived_scores_checked'],64)
        self.assertLess(report['max_archived_score_difference'],2e-6)
        for name,sha in report['files'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha,name)
        with (folder/'metrics.csv').open() as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),80)
        self.assertEqual({int(r['seed']) for r in rows},set(report['seeds']))
        self.assertEqual(len({(r['seed'],r['field'],r['method']) for r in rows}),80)

    def test_displayed_errors_recompute(self):
        folder=ROOT/'results/week12_research'
        report=json.loads((folder/'research_manifest.json').read_text())
        with (folder/'metrics.csv').open() as f:
            rows=list(csv.DictReader(f))
        with np.load(folder/'first_seed_fields.npz',allow_pickle=False) as arrays:
            for r in rows:
                if int(r['seed'])!=report['illustrated_seed']:
                    continue
                target=arrays[r['field']+'_reference'].astype(float)
                estimate=arrays[r['field']+'_'+r['method']].astype(float)
                self.assertEqual(estimate.shape,(100,100))
                self.assertTrue(np.isfinite(estimate).all())
                error=np.linalg.norm(estimate-target)/np.linalg.norm(target)
                self.assertAlmostEqual(error,float(r['reference_nrmse']),places=8)


if __name__=='__main__':
    unittest.main()
