"""Restart export regression without TensorFlow or private author sources."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location(
    "chris_runner", Path(__file__).resolve().parents[1] / "qa/run_chris_deep_original.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RestartExportTests(unittest.TestCase):
    def test_skipped_phases_preserve_existing_exports(self):
        exporter = Mock(side_effect=ValueError("empty arrays"))
        guarded = runner.restart_safe_saveplot(exporter)
        for _ in range(3):
            guarded(SimpleNamespace(steps=[]), object(), issave=True)
        exporter.assert_not_called()

    def test_resumed_history_is_exported_unchanged(self):
        exporter = Mock(return_value="saved")
        history, state = SimpleNamespace(steps=[1, 1000]), object()
        result = runner.restart_safe_saveplot(exporter)(history, state, issave=True)
        self.assertEqual(result, "saved")
        exporter.assert_called_once_with(history, state, issave=True)

    def test_real_export_errors_are_not_hidden(self):
        exporter = Mock(side_effect=OSError("disk full"))
        with self.assertRaises(OSError):
            runner.restart_safe_saveplot(exporter)(SimpleNamespace(steps=[1]), object())


if __name__ == "__main__":
    unittest.main()
