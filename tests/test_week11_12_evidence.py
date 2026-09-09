"""Checks for retained, real-research classroom evidence (no training needed)."""
from pathlib import Path
import csv
import hashlib
import json
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ResearchEvidenceTests(unittest.TestCase):
    def test_reconstruction_protocol_and_retained_hashes(self):
        folder=ROOT/'results/week11_reconstruction'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,digest in manifest.items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),digest,name)
        protocol=json.loads((folder/'protocol.json').read_text())
        self.assertEqual(protocol['train'],[90,110])
        self.assertEqual(protocol['validation'],100)
        self.assertEqual(protocol['retained_test'],105)
        self.assertEqual(protocol['seeds'],[17,29,43])
        self.assertEqual(protocol['epochs'],60)
        self.assertEqual(protocol['source_sha256'],hashlib.sha256((ROOT/'qa/run_week11_reconstruction.py').read_bytes()).hexdigest())
        rows=json.loads((folder/'metrics.json').read_text())
        self.assertEqual(len(rows),7)
        for method in ('reconstruction','segmentation'):
            self.assertEqual({r['seed'] for r in rows if r['method']==method},{17,29,43})
        for row in rows:
            self.assertTrue(0 <= row['dice'] <= 1)
            self.assertTrue(0 <= row['iou'] <= 1)
            if row['method']=='segmentation':
                self.assertNotIn('velocity_relative_l2',row)

    def test_six_fixed_checkpoint_runs(self):
        folder=ROOT/'results/week11_research'
        report=json.loads((folder/'research_manifest.json').read_text())
        self.assertEqual(len(report['runs']),6)
        self.assertEqual(len(set(report['ids'])),6)
        self.assertFalse(report['new_training'])
        self.assertFalse(report['new_solver_runs'])
        self.assertFalse(report['human_accuracy'])
        self.assertEqual(report['thresholds'],{'shock':.97,'vortex_core':.85})
        identity=report['model_identity']
        self.assertEqual(identity['name'],'Harmonized Joint (HJ)')
        self.assertEqual(identity['variant'],'task-preserving shock repair')
        self.assertEqual(identity['kit'],'shock_repair_v2_native86_v1')
        self.assertEqual(identity['displayed_outputs'],['shock','vortex_core'])
        self.assertEqual(identity['unet_role'],
                         'capacity-matched baseline / separate reconstruction experiment')
        for name,sha in report['figure_sha256'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),sha)
        replot=json.loads((folder/'presentation_replot_manifest.json').read_text())
        self.assertIn('no inference',replot['operation'])
        for stem,item in replot['items'].items():
            self.assertEqual(hashlib.sha256((folder/f'{stem}.png').read_bytes()).hexdigest(),
                             item['png_sha256'])
            self.assertEqual(hashlib.sha256((folder/f'{stem}.pdf').read_bytes()).hexdigest(),
                             item['pdf_sha256'])

    def test_week11_model_name_and_verified_movies_are_visible(self):
        documents=[ROOT/'README.md',ROOT/'notebooks/week11/README.md',
                   ROOT/'results/week11_research/README.md']
        for document in documents:
            text=document.read_text(encoding='utf-8')
            self.assertIn('Harmonized Joint',text,document)
            self.assertIn('j9rO5j3sudA',text,document)
            self.assertIn('hh3K40KRBUQ',text,document)

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
