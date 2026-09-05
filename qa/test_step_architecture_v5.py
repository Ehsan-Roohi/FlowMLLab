"""Protocol tests; run with python -I qa/test_step_architecture_v5.py.

TensorFlow training and checkpoint restore are covered separately by the runner's
six-arm, two-epoch `smoke` command. These tests check scientific bookkeeping.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

RUNNER = Path(__file__).with_name('step_architecture_v5.py')
spec = importlib.util.spec_from_file_location('architecture_v5_tested', RUNNER)
v5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v5
spec.loader.exec_module(v5)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.case_ids = np.repeat(np.arange(5, dtype='int32'), 40)
        self.u = np.tile(np.r_[[-2., -1.], np.ones(38)], 5)

    def test_zonal_draws_use_raw_velocity_and_retain_repeats(self):
        ids = v5.sampled_rows(self.u, 100, .6, 690)
        self.assertEqual(len(ids), 100)
        self.assertEqual(int((self.u[ids] < 0).sum()), 60)
        self.assertLess(len(np.unique(ids[self.u[ids] < 0])), 60)
        np.testing.assert_array_equal(ids, v5.sampled_rows(self.u, 100, .6, 690))

    def test_uniform_draws_ignore_values_and_do_not_repeat(self):
        a = v5.sampled_rows(self.u, 100, None, 691)
        b = v5.sampled_rows(-100*self.u, 100, None, 691)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), len(a))

    def test_batches_keep_every_draw_and_only_one_geometry(self):
        ids = v5.sampled_rows(self.u, 100, .6, 690)
        for epoch in (1, 2, 90):
            batches = v5.epoch_batches(ids, self.case_ids, 8, 690, epoch)
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.sort(ids))
            for batch in batches:
                self.assertLessEqual(len(batch), 8)
                self.assertEqual(len(np.unique(self.case_ids[batch])), 1)
            again = v5.epoch_batches(ids, self.case_ids, 8, 690, epoch)
            for a, b in zip(batches, again, strict=True):
                np.testing.assert_array_equal(a, b)

    def test_distinct_epochs_shuffle_but_keep_budget(self):
        ids = v5.sampled_rows(self.u, 100, .6, 690)
        a = v5.epoch_batches(ids, self.case_ids, 8, 690, 1)
        b = v5.epoch_batches(ids, self.case_ids, 8, 690, 2)
        self.assertEqual([len(a), sum(map(len, a))], [len(b), sum(map(len, b))])
        self.assertFalse(np.array_equal(np.concatenate(a), np.concatenate(b)))

    def test_joint_vector_error_not_mean_of_component_errors(self):
        y = np.array([[-3., 4.], [3., 4.]])
        r = v5.metrics(y, y*1.1)
        self.assertAlmostEqual(r['global_percent'], 10.)
        self.assertAlmostEqual(r['vortex_percent'], 10.)
        self.assertEqual(r['negative_u_iou'], 1.)

    def test_nonfinite_or_undefined_metrics_fail(self):
        y = np.array([[-3., 4.], [3., 4.]])
        for value in (np.nan, np.inf):
            p = y.copy()
            p[0, 0] = value
            with self.assertRaises(ValueError):
                v5.metrics(y, p)
        with self.assertRaises(ValueError):
            v5.metrics(np.ones((2, 2)), np.ones((2, 2)))

    def test_checkpoint_guard_is_inclusive_and_not_relaxed(self):
        rows = [{'epoch': 5, 'global_percent': 5., 'vortex_percent': 30.},
                {'epoch': 10, 'global_percent': 6., 'vortex_percent': 20.},
                {'epoch': 15, 'global_percent': 6.001, 'vortex_percent': 1.}]
        self.assertEqual(v5.select_checkpoint(rows, 6.)['epoch'], 10)
        self.assertIsNone(v5.select_checkpoint(rows, 4.9))

    def test_equal_scores_choose_earlier_epoch(self):
        rows = [{'epoch': e, 'global_percent': 5., 'vortex_percent': 30.} for e in (10, 5)]
        self.assertEqual(v5.select_checkpoint(rows, 6.)['epoch'], 5)

    def test_test_file_open_is_blocked_even_when_file_absent(self):
        code = (
            'import importlib.util,sys; '
            's=importlib.util.spec_from_file_location("v5",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
            'sys.addaudithook(m.forbid_test); open(m.FORBIDDEN_TEST,"rb")'
        )
        result = subprocess.run([sys.executable, '-I', '-c', code, str(RUNNER)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('V5_TEST_ARCHIVE_ACCESS_FORBIDDEN', result.stderr)

    def test_config_freezes_three_models_and_keeps_test_out(self):
        c = v5.configuration(RUNNER)
        self.assertEqual(c['seeds'], [690, 691, 692])
        self.assertEqual(c['models'], ('mlp', 'deeponet', 'geom'))
        self.assertEqual(c['sample_size'], 60000)
        self.assertEqual(c['epochs'], 90)
        self.assertEqual(set(v5.DEV) & set(v5.VAL), set())
        self.assertFalse(set((44, 67)) & set(v5.DEV+v5.VAL))

    def test_atomic_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'record.json'
            v5.save_json(path, {'value': 1})
            with self.assertRaises(ValueError):
                v5.save_json(path, {'value': float('nan')})
            self.assertEqual(path.read_text().strip(), '{\n  "value": 1\n}')

    def test_submit_snapshots_and_records_exact_job_without_changing_old_run(self):
        with tempfile.TemporaryDirectory(prefix='v5 test space ') as temp:
            base = Path(temp)
            python = base/'conda-tf220-clean'/'bin'/'python'
            python.parent.mkdir(parents=True)
            python.touch()
            old = base/'runs'/'old-run'
            old.mkdir(parents=True)
            (old/'report.json').write_text('preserve')
            calls = []

            def snapshot(out, root):
                target = out/'step_architecture_v5.py'
                target.write_bytes(RUNNER.read_bytes())
                return target

            def command(args):
                calls.append(args)
                if args[0] == 'squeue':
                    return ''
                if args[0] == 'sbatch':
                    return '123456;unity'
                return str(base/'nvidia'/'cublas'/'lib')

            with mock.patch.object(v5, 'snapshot', side_effect=snapshot), mock.patch.object(v5, 'command', side_effect=command):
                v5.submit(SimpleNamespace(base=base))
            out = Path((base/v5.POINTER).read_text().strip())
            self.assertEqual((out/'JOB_ID').read_text().strip(), '123456')
            self.assertEqual((old/'report.json').read_text(), 'preserve')
            self.assertEqual(subprocess.run(['bash', '-n', str(out/'job.sbatch')]).returncode, 0)
            batch = next(c for c in calls if c[0] == 'sbatch')
            self.assertIn('--job-name=step-arch-v5', batch)
            self.assertIn('--chdir='+str(out), batch)

    def test_active_job_prevents_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            python = base/'conda-tf220-clean'/'bin'/'python'
            python.parent.mkdir(parents=True)
            python.touch()
            with mock.patch.object(v5, 'command', return_value='123 RUNNING'):
                with self.assertRaisesRegex(RuntimeError, 'V5_ALREADY_ACTIVE'):
                    v5.submit(SimpleNamespace(base=base))
            self.assertFalse((base/v5.POINTER).exists())


if __name__ == '__main__':
    unittest.main()
