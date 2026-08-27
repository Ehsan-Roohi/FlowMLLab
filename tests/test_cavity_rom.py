from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from flowmllab import cavity_rom


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))
import w4utils  # noqa: E402


class CavityROMTests(unittest.TestCase):
    def test_fom_kernel_matches_week4_solver(self) -> None:
        extension = cavity_rom.simulate_fom(
            100, n=25, dt=1.0e-3, steps=20, snapshot_stride=20
        )
        course = w4utils.run_cavity(
            100,
            N=25,
            dt=1.0e-3,
            max_steps=20,
            min_steps=21,
            verbose=False,
        )
        self.assertLess(np.max(np.abs(extension["final_u"] - course["u"])), 1.0e-13)
        self.assertLess(np.max(np.abs(extension["final_v"] - course["v"])), 1.0e-13)
        self.assertLess(
            np.max(np.abs(extension["final_omega"] - course["omega"])), 1.0e-11
        )

    def test_centered_pod_is_orthonormal(self) -> None:
        trajectory = cavity_rom.simulate_fom(
            100, n=25, dt=2.0e-3, steps=120, snapshot_stride=20
        )
        pod = cavity_rom.fit_pod([trajectory["states"]], rank=3)
        identity = pod.modes.T @ pod.modes
        self.assertTrue(np.allclose(identity, np.eye(3), atol=1.0e-12))
        self.assertGreater(pod.cumulative_energy[2], 0.99)

    def test_deim_sampled_convection_is_exact_at_selected_points(self) -> None:
        trajectory = cavity_rom.simulate_fom(
            100, n=25, dt=2.0e-3, steps=160, snapshot_stride=20
        )
        pod = cavity_rom.fit_pod([trajectory["states"]], rank=4)
        nonlinear = cavity_rom.convection_snapshots(
            [trajectory["states"]], 100, pod.n
        )
        nonlinear_basis = cavity_rom.fit_nonlinear_basis(nonlinear, max_dimension=5)
        deim = cavity_rom.fit_deim(pod, nonlinear_basis, 5)
        coefficients = cavity_rom.project_state(pod, trajectory["states"][3])
        reconstructed = cavity_rom.reconstruct_state(pod, coefficients)
        full, _ = cavity_rom.rhs_terms(reconstructed, 100, pod.n)
        sampled = cavity_rom.sampled_convection(deim, coefficients)
        self.assertTrue(np.allclose(sampled, full[deim.indices], atol=1.0e-11))

    def test_micro_rom_trajectories_remain_physical(self) -> None:
        trajectory = cavity_rom.simulate_fom(
            100, n=25, dt=2.0e-3, steps=200, snapshot_stride=20
        )
        pod = cavity_rom.fit_pod([trajectory["states"]], rank=5)
        nonlinear = cavity_rom.convection_snapshots(
            [trajectory["states"]], 100, pod.n
        )
        basis = cavity_rom.fit_nonlinear_basis(nonlinear, max_dimension=6)
        deim = cavity_rom.fit_deim(pod, basis, 6)
        galerkin = cavity_rom.simulate_pod_galerkin(pod, 100, 2.0e-3, 200, 20)
        hyper = cavity_rom.simulate_pod_deim(deim, 100, 2.0e-3, 200, 20)
        self.assertTrue(np.isfinite(galerkin["states"]).all())
        self.assertTrue(np.isfinite(hyper["states"]).all())
        diagnostics = cavity_rom.physical_diagnostics(hyper["states"][-1], pod.n)
        self.assertEqual(diagnostics["wall_rms_error"], 0.0)
        self.assertLess(diagnostics["divergence_l2"], 1.0e-12)

    def test_model_archive_round_trip(self) -> None:
        trajectory = cavity_rom.simulate_fom(
            100, n=25, dt=2.0e-3, steps=100, snapshot_stride=20
        )
        pod = cavity_rom.fit_pod([trajectory["states"]], rank=3)
        nonlinear = cavity_rom.convection_snapshots(
            [trajectory["states"]], 100, pod.n
        )
        basis = cavity_rom.fit_nonlinear_basis(nonlinear, max_dimension=4)
        model = cavity_rom.fit_deim(pod, basis, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = cavity_rom.save_deim_model(Path(directory) / "model.npz", model)
            restored = cavity_rom.load_deim_model(path)
        coefficients = cavity_rom.project_state(pod, trajectory["states"][2])
        self.assertTrue(
            np.allclose(
                cavity_rom.deim_modal_rhs(model, coefficients, 175),
                cavity_rom.deim_modal_rhs(restored, coefficients, 175),
            )
        )


if __name__ == "__main__":
    unittest.main()
