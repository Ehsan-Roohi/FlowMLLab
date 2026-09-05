"""Audit the compact import; this does not re-run GPU checkpoint inference."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class V5ImportedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.read_csv(ROOT / 'results/step_architecture_v5/seed_metrics.csv')

    def test_complete_matched_budget_matrix(self):
        self.assertEqual(len(self.rows), 18)
        self.assertFalse(self.rows.duplicated(['model', 'sampler', 'seed']).any())
        for _, group in self.rows.groupby(['sampler', 'seed']):
            self.assertEqual(set(group.model), {'mlp', 'deeponet', 'geom'})
            self.assertEqual(group.updates.nunique(), 1)
            self.assertTrue((group.target_exposures == 5_400_000).all())
            self.assertEqual(group.ceiling_percent.nunique(), 1)

    def test_rejected_checkpoints_remain_missing(self):
        selected = self.rows.status == 'selected'
        self.assertEqual(int(selected.sum()), 6)
        cols = ['selected_epoch', 'selected_global_percent', 'selected_vortex_percent']
        self.assertTrue(self.rows.loc[~selected, cols].isna().all().all())
        self.assertTrue(np.isfinite(self.rows.loc[selected, cols]).all().all())
        self.assertTrue((self.rows.loc[selected, 'selected_global_percent'] <=
                         self.rows.loc[selected, 'ceiling_percent']).all())
        self.assertTrue(np.isfinite(self.rows[['terminal_global_percent', 'terminal_vortex_percent']]).all().all())

    def test_reported_selected_means_and_failure_are_retained(self):
        group = self.rows[self.rows.sampler == 'uniform'].groupby('model')
        self.assertAlmostEqual(group.selected_global_percent.mean()['geom'], 6.1862410791923)
        self.assertAlmostEqual(group.selected_vortex_percent.mean()['geom'], 60.850251811892406)
        self.assertAlmostEqual(group.selected_vortex_percent.mean()['mlp'], 103.73765189792756)
        failed = self.rows[(self.rows.model == 'geom') & (self.rows.seed == 691) & (self.rows.sampler == 'uniform')]
        self.assertGreater(float(failed.terminal_vortex_percent.iloc[0]), 169)
