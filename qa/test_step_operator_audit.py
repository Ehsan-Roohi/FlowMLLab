"""Offline numerical and artifact tests; never launches training or edits evidence."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest

import nbformat
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/step_operator_audit'
NB = ROOT / 'notebooks/week15/W15_Geometry_Operators_Step_Audit.ipynb'
EXPECTED = '190bb252c2739fc2acfa0841233652144b82eb9ca3a01ad6ddb2ae0f0429ade1'
EXPECTED_DATASET = '28d4d4c440cdc4c1ac1d13749ce00b0690d99f29cf20fd56c65fc00b6a8058fd'
notebook = nbformat.read(NB, as_version=4)
# Test exactly the implementations students execute, not a separately retyped formula.
namespace = {'np': np, 'binary_erosion': binary_erosion}
for cell in notebook.cells:
    if cell.cell_type == 'code' and 'def diagnostics(' in cell.source:
        tree = ast.parse(cell.source)
        defs = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
        exec(compile(defs, str(NB), 'exec'), namespace)
diagnostics, curl, relative = (namespace[k] for k in ('diagnostics', 'curl', 'relative'))
fields, reports = {}, {}
with tarfile.open(DATA/'wake_predictions.tgz') as tar:
    members = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
for model in ('geom', 'fno', 'ufno'):
    prefix = f'wake_focused/{model}/seed_17/'
    reports[model] = json.loads(members[prefix+'metrics.json'])
    for re in (25, 50, 100):
        with np.load(io.BytesIO(members[prefix+f'g011_Re{re}_medium_prediction.npz']), allow_pickle=False) as z:
            fields[model, re] = {k:z[k].copy() for k in z.files}


class AuditTests(unittest.TestCase):
    def test_project_vanilla_deeponet_code_and_v5_evidence(self):
        source = (ROOT/'qa/step_architecture_v5.py').read_text(encoding='utf-8')
        self.assertIn("name='vanilla_deeponet'", source)
        self.assertIn('for width in (128, 128)', source)
        self.assertIn('contraction(rank=48, output_dim=2)', source)
        v5 = pd.read_csv(ROOT/'results/step_architecture_v5/seed_metrics.csv')
        self.assertEqual(len(v5), 18)
        self.assertEqual(set(v5['model']), {'mlp','deeponet','geom'})
        deep = v5[v5['model']=='deeponet'].groupby('sampler')[
            ['terminal_global_percent','terminal_vortex_percent']].mean()
        self.assertAlmostEqual(deep.loc['uniform','terminal_global_percent'], 14.0063, places=3)
        self.assertAlmostEqual(deep.loc['zonal','terminal_global_percent'], 23.3082, places=3)
        self.assertTrue((v5[v5['model']=='deeponet']['status']=='no_eligible_checkpoint').all())

    def test_openfoam_ordinary_deeponet_three_seed_evidence(self):
        root = ROOT/'results/week15_ordinary_deeponet'
        metric_files = sorted(root.glob('ordinary-deeponet-seed*/case_metrics.csv'))
        manifest_files = sorted(root.glob('ordinary-deeponet-seed*/manifest.json'))
        self.assertEqual(len(metric_files), 3)
        self.assertEqual(len(manifest_files), 3)
        table = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
        self.assertEqual(len(table), 36)
        self.assertEqual(set(table.seed), {17, 29, 43})
        self.assertEqual(set(table.geometry), {'g009', 'g023', 'g036', 'g048'})
        self.assertAlmostEqual(table.velocity_percent.mean(), 28.9601068, places=5)
        self.assertAlmostEqual(table.pressure_percent.mean(), 299.5745888, places=4)
        self.assertAlmostEqual(table.reverse_iou.mean(), 0.4056609, places=6)
        for path in manifest_files:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(manifest['dataset_sha256'], EXPECTED_DATASET)
            self.assertEqual(manifest['epochs'], 400)
            self.assertEqual(manifest['split']['test_geometry_ids'], [9, 23, 36, 48])
            self.assertEqual(manifest['excluded_geometry_inputs'], ['mask', 'SDF', 'geometry ID'])

    def test_archive_and_all_member_hashes(self):
        self.assertEqual(hashlib.sha256((DATA/'wake_predictions.tgz').read_bytes()).hexdigest(), EXPECTED)
        manifest = json.loads((DATA/'generated/source_manifest.json').read_text())
        self.assertEqual(set(manifest['members']), set(members))
        for name, raw in members.items():
            self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['members'][name])

    def test_notebook_executed_without_errors(self):
        nbformat.validate(notebook)
        code = [c for c in notebook.cells if c.cell_type == 'code']
        self.assertEqual(len(code), 13)
        self.assertEqual([c.execution_count for c in code], list(range(1, 14)))
        self.assertFalse(any(o.output_type == 'error' for c in code for o in c.outputs))
        self.assertEqual(sum('image/png' in o.get('data', {}) for c in code for o in c.outputs), 11)

    def test_full_dataset_hash_shapes_and_case_splits(self):
        path = DATA/'source/dataset.npz'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), EXPECTED_DATASET)
        with np.load(path, allow_pickle=False) as z:
            self.assertEqual(set(z.files), {'raw','queries','masks','Re','shape'})
            self.assertEqual(z['raw'].shape, (130,18000,3))
            self.assertEqual(z['queries'].shape, (130,18000,3))
            self.assertEqual(z['masks'].shape, (130,18000))
            self.assertEqual(tuple(z['shape']), (60,300))
            self.assertTrue(np.isfinite(z['raw']).all())
            self.assertTrue(np.isfinite(z['queries']).all())
            hashes = [hashlib.sha256(np.packbits(m).tobytes()).hexdigest() for m in z['masks']]
            order = list(dict.fromkeys(hashes))
            gids = np.array([order.index(h)+1 for h in hashes])
            self.assertEqual(len(order), 51)
            source = reports['geom']['geometry_ids']
            counts = [int(np.isin(gids, source[k]).sum()) for k in ('train','validation','test')]
            self.assertEqual(counts, [106,21,3])

    def test_dataset_g011_matches_retained_reference_bitwise(self):
        with np.load(DATA/'source/dataset.npz', allow_pickle=False) as z:
            hashes = [hashlib.sha256(np.packbits(m).tobytes()).hexdigest() for m in z['masks']]
            order = list(dict.fromkeys(hashes))
            gids = np.array([order.index(h)+1 for h in hashes])
            for i in np.where(gids == 11)[0]:
                re = int(z['Re'][i]); f = fields['geom',re]
                np.testing.assert_array_equal(z['raw'][i], f['truth'])
                np.testing.assert_array_equal(z['masks'][i], f['mask'])
                np.testing.assert_allclose(z['queries'][i,:,:2], f['coordinates']/[5,1], rtol=0, atol=0)

    def test_nine_finite_fields_and_shared_references(self):
        self.assertEqual(len(fields), 9)
        for (m, re), f in fields.items():
            self.assertEqual(f['prediction'].shape, (18000, 3))
            self.assertTrue(np.isfinite(f['prediction']).all())
            self.assertTrue(np.isfinite(f['truth']).all())
            for key in ('truth', 'coordinates', 'mask', 'shape'):
                np.testing.assert_array_equal(f[key], fields['geom', re][key])

    def test_recorded_splits_and_dataset_consistent(self):
        baseline = reports['geom']
        for r in reports.values():
            self.assertEqual(r['dataset_sha256'], baseline['dataset_sha256'])
            self.assertEqual(r['geometry_ids'], baseline['geometry_ids'])
            groups = [set(r['geometry_ids'][s]) for s in ('train', 'validation', 'test')]
            self.assertEqual([len(g) for g in groups], [41, 9, 1])
            self.assertEqual(groups[2], {11})
            for i in range(3):
                for j in range(i):
                    self.assertFalse(groups[i] & groups[j])

    def test_eighteen_source_field_metrics(self):
        for (m, re), f in fields.items():
            got = diagnostics(f)
            src = next(t for t in reports[m]['test'] if t['Re'] == re)
            self.assertAlmostEqual(got['velocity_L2_pct'], src['velocity_relative_l2_percent'], delta=.001)
            self.assertAlmostEqual(got['pressure_raw_L2_pct'], src['relative_l2_percent'][2], delta=.001)

    def test_all_csv_results_recomputed(self):
        table = pd.read_csv(DATA/'generated/recomputed_metrics.csv')
        self.assertEqual(len(table), 9)
        labels = {'geom':'Geo-DeepONet', 'fno':'FNO', 'ufno':'U-FNO'}
        for (m, re), f in fields.items():
            row = table[(table.model == labels[m]) & (table.Re == re)].iloc[0]
            for name, value in diagnostics(f).items():
                self.assertAlmostEqual(value, row[name], delta=1e-10)

    def test_perfect_prediction_all_nine_fields(self):
        for f in fields.values():
            copy = dict(f, prediction=f['truth'].copy())
            got = diagnostics(copy)
            for key in ('velocity_L2_pct', 'pressure_raw_L2_pct', 'pressure_centered_L2_pct', 'vorticity_interior_L2_pct'):
                self.assertEqual(got[key], 0)
            self.assertEqual(got['reverse_IoU'], 1)

    def test_curl_axes_sign_and_nonuniform_grid(self):
        x = np.linspace(.1, 2, 19)**1.4
        y = np.linspace(.1, 1, 15)**1.3
        X, Y = np.meshgrid(x, y)
        a = np.stack([-3*Y, 2*X, 0*X], axis=-1)
        np.testing.assert_allclose(curl(a, x, y), 5, atol=1e-11)
        b = np.stack([X**2, Y**2, 0*X], axis=-1)
        np.testing.assert_allclose(curl(b, x, y), 0, atol=1e-11)

    def test_pressure_gauge_and_scale(self):
        f = fields['geom', 50]
        base = diagnostics(f)
        for scale in (1/50, 50):
            t, p = f['truth'].astype(float), f['prediction'].astype(float)
            t[:, 2] *= scale
            p[:, 2] *= scale
            got = diagnostics(dict(f, truth=t, prediction=p))
            self.assertAlmostEqual(got['pressure_raw_L2_pct'], base['pressure_raw_L2_pct'], places=10)
            self.assertAlmostEqual(got['pressure_centered_L2_pct'], base['pressure_centered_L2_pct'], places=10)
        p = f['truth'].astype(float)
        p[:, 2] += 7
        got = diagnostics(dict(f, prediction=p))
        self.assertLess(got['pressure_centered_L2_pct'], 1e-10)
        self.assertGreater(got['pressure_raw_L2_pct'], 0)

    def test_solid_values_do_not_contaminate_diagnostics(self):
        f = fields['geom', 50]
        p = f['prediction'].astype(float)
        p[~f['mask']] = 1e9
        self.assertEqual(diagnostics(dict(f, prediction=p)), diagnostics(f))

    def test_zero_reference_denominator_not_fake_accuracy(self):
        self.assertTrue(np.isnan(relative(np.ones(4), np.zeros(4))))

    def test_source_iou_disagreement_preserved(self):
        table = pd.read_csv(DATA/'generated/recomputed_metrics.csv')
        np.testing.assert_allclose(table.reverse_IoU - table.source_reverse_IoU,
                                   table.reverse_IoU_delta_from_source, atol=1e-14)
        self.assertGreater(abs(table.reverse_IoU_delta_from_source).max(), .001)


if __name__ == '__main__':
    unittest.main(verbosity=2)
