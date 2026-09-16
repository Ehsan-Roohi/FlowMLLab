"""Independent contracts for the retained hydrofoil segmentation lab."""
from pathlib import Path
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Requires optional reconstruction/PyTorch dependency')
class CavitationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from flowmllab import cavitation_detection as cv
        cls.cv = cv
        cls.data = Path(__file__).resolve().parents[1]/'data/week11_cavitation'
        torch.set_num_threads(2)
        cls.manifest, cls.cases = cv.load_bundle(cls.data)
        cls.model = cv.load_model(cls.data/'detector_weights.npz')

    def test_raster_topology_inclusive_threshold_and_disconnection(self):
        wall = np.zeros((20, 30), bool)
        wall[10:13, 3:14] = True
        alpha = np.zeros(wall.shape, 'float32')
        alpha[9, 5:10] = .20  # exactly inclusive, face adjacent to wall
        alpha[4:6, 20:23] = .9  # detached component
        alpha[4, 2] = .19
        label = self.cv.raster_baseline(alpha, wall)
        self.assertTrue(np.all(label[9, 5:10] == 1))
        self.assertTrue(np.all(label[4:6, 20:23] == 2))
        self.assertEqual(label[4, 2], 0)
        self.assertFalse(label[wall].any())

    def test_ignore_and_empty_score_convention(self):
        ref = np.array([[1, 2, 255], [0, 0, 255]], dtype='uint8')
        pred = np.array([[1, 0, 2], [1, 0, 1]], dtype='uint8')
        measured = self.cv.scores(pred, ref)
        self.assertEqual(measured['attached_2d'], dict(dice=2/3, tp=1, fp=1, fn=0))
        self.assertEqual(measured['disconnected_2d'], dict(dice=0., tp=0, fp=0, fn=1))
        empty = np.zeros((2, 3), dtype='uint8')
        self.assertIsNone(self.cv.scores(empty, empty)['attached_2d']['dice'])

    def test_reference_cannot_change_inference(self):
        c = self.cases['Case1LES']
        result = self.cv.predict(self.model, c['alpha'][22], c['wall'])
        np.testing.assert_array_equal(result[0], c['archived_prediction'][22])
        other_reference = np.zeros_like(c['reference'][22])
        self.assertNotEqual(self.cv.scores(result[0], other_reference),
                            self.cv.scores(result[0], c['reference'][22]))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 126275)

    def test_zero_and_noisy_fields_use_the_same_frozen_model(self):
        c = self.cases['Case1LES']
        before = {k:v.clone() for k,v in self.model.state_dict().items()}
        zero = self.cv.predict(self.model, np.zeros_like(c['alpha'][0]), c['wall'])
        self.assertEqual(int((zero>0).sum()), 0)
        rng = np.random.default_rng(2)
        noisy = np.clip(c['alpha'][22]+rng.normal(0,.02,c['wall'].shape), 0, 1)
        p = self.cv.predict(self.model, noisy, c['wall'])
        self.assertEqual(p.shape, (1, *c['wall'].shape))
        self.assertTrue(np.isin(p, [0,1,2]).all())
        for k,v in self.model.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_optional_training_does_not_mutate_the_parent(self):
        parent = self.cv.load_model(self.data/'parent_weights.npz')
        before = {k:v.clone() for k,v in parent.state_dict().items()}
        # Supply only TRAIN cases: an accidental dependency on validation/test fails.
        train = {k:self.cases[k] for k in self.manifest['protocol']['training_cases']}
        model, history = self.cv.adapt_from_parent(parent, train, self.manifest['protocol'], steps=2)
        self.assertTrue(np.isfinite(history[-1]['loss']))
        self.assertTrue(any(not torch.equal(v, before[k]) for k,v in model.state_dict().items()))
        for k,v in parent.state_dict().items():
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
