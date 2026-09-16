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
    def test_capacity_adapter_changes_only_hidden_width(self):
        calls = []
        received = []

        def base_fnn(layers, activation, initializer):
            received.append((layers, activation, initializer))
            return object()

        adapter = runner.capacity_fnn_adapter(base_fnn, 64, calls)
        adapter([5] + [32] * 6 + [3], "tanh", "Glorot normal")
        self.assertEqual(received, [([5] + [64] * 6 + [3],
                                     "tanh", "Glorot normal")])
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adapter([5] + [32] * 6 + [3], "tanh", "Glorot normal")

    def test_capacity_adapter_rejects_changed_author_architecture(self):
        adapter = runner.capacity_fnn_adapter(lambda *a: None, 64, [])
        with self.assertRaisesRegex(ValueError, "Unexpected author FNN"):
            adapter([5, 32, 32, 3], "tanh", "Glorot normal")

    def test_repeated_precision_loss_is_an_external_plateau(self):
        self.assertTrue(runner.external_plateau(
            7.548158881160543e-08,
            7.548158881160543e-08,
            "Desired error not necessarily achieved due to precision loss.",
        ))

    def test_real_improvement_is_not_an_external_plateau(self):
        self.assertFalse(runner.external_plateau(
            8.0e-8, 7.5e-8,
            "Desired error not necessarily achieved due to precision loss.",
        ))

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

