"""Exercise staged scheduling and resume without spending GPU resources."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'continuation', Path(__file__).resolve().parents[1] / 'qa/run_chris_depth_continuation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ContinuationTests(unittest.TestCase):
    def exercise(self, loss=1e-7, status=0):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'runs'
            commands = []

            def launch(command):
                commands.append(command)
                stage = Path(command[command.index('--output') + 1])
                stage.mkdir(parents=True)
                checkpoint = stage / 'model' / 'restart-100'
                checkpoint.parent.mkdir()
                for suffix in ('.index', '.meta', '.data-00000-of-00001'):
                    Path(str(checkpoint) + suffix).touch()
                (stage / 'resume.json').write_text(json.dumps(
                    dict(checkpoint='model/restart-100', last_external_loss=loss)))
                return SimpleNamespace(wait=lambda: status)

            argv = ['runner', '--archive', str(Path(tmp) / 'private.zip'), '--output', str(output)]
            with patch.object(runner.sys, 'argv', argv), patch.object(runner.signal, 'signal'), \
                    patch.object(runner.signal, 'SIGUSR1', 10, create=True), \
                    patch.object(runner.subprocess, 'Popen', side_effect=launch):
                if loss > 1e-5 and status == 0:
                    with self.assertRaisesRegex(RuntimeError, 'failed progression gate'):
                        runner.main()
                    self.assertEqual(len(commands), 1)
                    return
                self.assertEqual(runner.main(), status)
                if status:
                    self.assertEqual(len(commands), 1)
                    return
                self.assertEqual(len(commands), 4)
                self.assertNotIn('--refine-from', commands[0])
                for i in range(1, 4):
                    checkpoint = commands[i][commands[i].index('--refine-from') + 1]
                    self.assertIn(f'stage-{i-1}-', checkpoint)
                self.assertEqual(runner.main(), 0)
                self.assertEqual(len(commands), 4)  # No retraining completed stages.

    def test_transfers_previous_checkpoint_and_resumes(self):
        self.exercise()

    def test_bad_stage_does_not_launch_deeper_case(self):
        self.exercise(loss=0.1)

    def test_preemption_does_not_launch_deeper_case(self):
        self.exercise(status=99)


if __name__ == '__main__':
    unittest.main()
