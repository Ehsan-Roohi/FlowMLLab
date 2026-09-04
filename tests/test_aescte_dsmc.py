from pathlib import Path
import unittest

import numpy as np

from flowmllab.aescte_dsmc import (
    CAVITY_KNUDSEN,
    CAVITY_LID_SPEEDS_MS,
    cavity_case,
    fit_pod_polynomial_operator,
    load_cavity_archive,
    load_diatomic_shock_raw,
    logarithmic_kn_prediction,
    maxwell_speed_pdf,
    normalized_rmse,
    predict_pod_polynomial_operator,
)


ROOT = Path(__file__).resolve().parents[1]


class AESCTEDSMCWeek10Tests(unittest.TestCase):
    def test_complete_cavity_archive_and_case_identity(self) -> None:
        data = load_cavity_archive(
            ROOT / "results/aescte_dsmc/cavity_fields_14cases.npz"
        )
        self.assertEqual(data["temperature_k"].shape, (14, 50, 50))
        self.assertEqual(set(np.unique(data["lid_speed_ms"])), set(CAVITY_LID_SPEEDS_MS))
        self.assertEqual(set(np.unique(data["knudsen"])), set(CAVITY_KNUDSEN))
        self.assertEqual(cavity_case(data, 10, 0.05), 2)
        self.assertEqual(cavity_case(data, 30, 0.5), 11)

    def test_log_kn_cavity_reproduction_passes_primary_gate(self) -> None:
        data = load_cavity_archive(
            ROOT / "results/aescte_dsmc/cavity_fields_14cases.npz"
        )
        for lid in CAVITY_LID_SPEEDS_MS:
            for kn in (0.05, 0.5):
                prediction = logarithmic_kn_prediction(
                    data, lid_speed_ms=lid, test_knudsen=kn
                )
                index = cavity_case(data, lid, kn)
                self.assertLess(
                    normalized_rmse(data["u_ms"][index], prediction["u_ms"], lid),
                    2.0,
                )
                self.assertLess(
                    normalized_rmse(data["v_ms"][index], prediction["v_ms"], lid),
                    2.0,
                )
                self.assertLess(
                    normalized_rmse(
                        data["temperature_k"][index],
                        prediction["temperature_k"],
                        50.0,
                    ),
                    1.0,
                )

    def test_diatomic_heldout_mach_reproduction(self) -> None:
        data = load_diatomic_shock_raw(ROOT)
        train = np.flatnonzero(~np.isclose(data["mach"], 1.7))
        target = int(np.flatnonzero(np.isclose(data["mach"], 1.7))[0])
        for field in (
            "rotational_temperature",
            "translational_temperature",
            "normalized_velocity",
        ):
            model = fit_pod_polynomial_operator(
                data["mach"], data[field], train, rank=4, degree=3
            )
            prediction = predict_pod_polynomial_operator(model, [1.7])[0]
            error = (
                100
                * np.linalg.norm(prediction - data[field][target])
                / np.linalg.norm(data[field][target])
            )
            self.assertLess(error, 0.35)

    def test_maxwell_pdf_is_normalized(self) -> None:
        speed = np.linspace(0, 3000, 10000)
        argon_mass = 39.948e-3 / 6.02214076e23
        pdf = maxwell_speed_pdf(speed, 325.0, argon_mass)
        self.assertLess(abs(np.trapezoid(pdf, speed) - 1.0), 1.0e-10)


if __name__ == "__main__":
    unittest.main()
