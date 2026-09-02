from __future__ import annotations

import unittest

import numpy as np

from flowmllab import cylinder_ml


def synthetic_wake(reynolds: np.ndarray, phase: np.ndarray) -> dict[str, np.ndarray]:
    """Low-rank smooth wake used only for deterministic unit tests."""
    y, x = np.mgrid[-1.0:1.0:7j, 0.0:4.0:11j]
    re = np.asarray(reynolds)[:, None, None]
    phi = np.asarray(phase)[:, None, None]
    z = (re - 70.0) / 30.0
    envelope = np.exp(-0.35 * x) * (1.0 - y**2)
    u = 1.0 - (0.18 + 0.02 * z) * envelope + 0.03 * (1.0 + 0.2 * z) * np.cos(phi) * y * envelope
    v = 0.05 * (1.0 + 0.1 * z) * np.sin(phi) * envelope
    p = 0.12 * z * np.exp(-x) + 0.02 * np.cos(phi) * envelope
    return {"u": u, "v": v, "p": p}


class CylinderMLTests(unittest.TestCase):
    def test_pack_unpack_round_trip(self) -> None:
        fields = synthetic_wake(np.array([40.0, 60.0]), np.array([0.0, 1.0]))
        vectors, layout = cylinder_ml.pack_fields(fields, ("u", "v", "p"))
        restored = cylinder_ml.unpack_fields(vectors, layout)
        for name in fields:
            self.assertTrue(np.array_equal(fields[name], restored[name]))

    def test_centered_pod_reconstructs_low_rank_snapshots(self) -> None:
        phase = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
        fields = synthetic_wake(np.full(phase.size, 70.0), phase)
        snapshots, _ = cylinder_ml.pack_fields(fields)
        pod = cylinder_ml.fit_pod(snapshots, rank=3)
        reconstructed = cylinder_ml.reconstruct_pod(
            pod, cylinder_ml.project_pod(pod, snapshots)
        )
        self.assertTrue(np.allclose(pod.modes.T @ pod.modes, np.eye(3), atol=1e-12))
        self.assertLess(np.linalg.norm(reconstructed - snapshots), 1.0e-12)

    def test_reynolds_split_never_leaks_snapshots(self) -> None:
        labels = np.repeat([40.0, 47.0, 80.0, 120.0], 6)
        split = cylinder_ml.casewise_reynolds_split(labels, test_reynolds=[47.0, 120.0])
        self.assertTrue(np.array_equal(split.test_reynolds, [47.0, 120.0]))
        self.assertTrue(np.array_equal(split.train_reynolds, [40.0, 80.0]))
        self.assertTrue(np.all(np.isin(labels[split.test_indices], [47.0, 120.0])))
        self.assertFalse(
            np.intersect1d(labels[split.train_indices], labels[split.test_indices]).size
        )

    def test_pod_regressor_predicts_unseen_intermediate_reynolds_case(self) -> None:
        train_cases = np.array([40.0, 70.0, 100.0])
        phase_grid = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        train_re = np.repeat(train_cases, phase_grid.size)
        train_phase = np.tile(phase_grid, train_cases.size)
        model = cylinder_ml.fit_pod_regressor(
            synthetic_wake(train_re, train_phase),
            train_re,
            train_phase,
            field_names=("u", "v", "p"),
            rank=6,
            reynolds_degree=2,
            phase_harmonics=1,
            ridge=0.0,
        )
        test_re = np.full(8, 85.0)
        test_phase = np.linspace(0.13, 2.0 * np.pi + 0.13, 8, endpoint=False)
        truth = synthetic_wake(test_re, test_phase)
        metrics = cylinder_ml.reconstruction_diagnostics(
            truth, model.predict_fields(test_re, test_phase)
        )
        self.assertLess(metrics["combined_relative_l2"], 1.0e-11)

    def test_flow_diagnostics_and_solid_speed(self) -> None:
        y, x = np.mgrid[-1.0:1.0:31j, -1.0:1.0:31j]
        # Rigid rotation is exactly divergence-free under centered differences.
        fields = {"u": -y, "v": x}
        solid = x**2 + y**2 < 0.2**2
        stopped = {name: values.copy() for name, values in fields.items()}
        stopped["u"][solid] = 0.0
        stopped["v"][solid] = 0.0
        metrics = cylinder_ml.flow_diagnostics(
            {"u": np.zeros_like(x), "v": np.zeros_like(y)},
            dx=x[0, 1] - x[0, 0],
            dy=y[1, 0] - y[0, 0],
            solid_mask=solid,
        )
        self.assertEqual(metrics["divergence_rms"], 0.0)
        self.assertEqual(metrics["solid_speed_rms"], 0.0)
        self.assertEqual(np.asarray(metrics["vorticity"]).shape, x.shape)


if __name__ == "__main__":
    unittest.main()
