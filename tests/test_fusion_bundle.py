import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from flowmllab.fusion_bundle import save_bundle, load_bundle


class BundleTests(unittest.TestCase):
    def test_frozen_transforms_and_model_identity(self):
        rng = np.random.default_rng(9)
        arrays = [rng.normal(size=(31, k)) for k in (1, 2, 3)]
        arrays[1][:, 1] = 5  # Constant features must retain sklearn semantics.
        scalers = [MinMaxScaler().fit(arrays[0]), MinMaxScaler().fit(arrays[1]),
                   StandardScaler().fit(arrays[2])]
        with tempfile.TemporaryDirectory() as root:
            model = Path(root) / 'model.keras'
            model.write_bytes(b'test model identity')
            save_bundle(root, [model.name], scalers, {'seed': 9})
            _, frozen = load_bundle(root)
            for a, fitted, replay in zip(arrays, scalers, frozen):
                unseen = a * 2 + 1
                np.testing.assert_allclose(replay.transform(unseen), fitted.transform(unseen))
                np.testing.assert_allclose(replay.inverse_transform(replay.transform(unseen)), unseen)
            with self.assertRaises(FileExistsError):
                save_bundle(root, [model.name], scalers, {})
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_bundle(root)

    def test_unbundled_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                load_bundle(root)
