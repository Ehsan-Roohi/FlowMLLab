from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from copy import deepcopy

import numpy as np

from flowmllab.scientific_software import (
    audit_cavity_case,
    differential_diagnostics,
    evaluate_acceptance,
    manufactured_incompressible_field,
    verification_sweep,
    validate_week01_1_evidence,
    write_acceptance_record,
)


ROOT = Path(__file__).resolve().parents[1]


class ScientificSoftwareTests(unittest.TestCase):
    def test_shape_and_axis_contract(self) -> None:
        x, y, u, v, _ = manufactured_incompressible_field(33)
        result = differential_diagnostics(x, y, u, v)
        self.assertEqual(result.divergence.shape, (33, 33))
        self.assertEqual(result.vorticity.shape, (33, 33))
        with self.assertRaisesRegex(ValueError, "shape"):
            differential_diagnostics(x, y, u.T[:-1], v.T[:-1])

    def test_manufactured_solution_is_second_order(self) -> None:
        sweep = verification_sweep()
        self.assertGreater(float(sweep["observed_order"]), 1.95)
        rows = list(sweep["rows"])
        self.assertLess(float(rows[-1]["vorticity_relative_l2"]), 5.0e-4)
        self.assertLess(max(float(row["divergence_rms"]) for row in rows), 1.0e-12)

    def test_real_cavity_contract_and_acceptance(self) -> None:
        verification = verification_sweep()
        cavity = audit_cavity_case(ROOT, 100.0)
        self.assertEqual(cavity["grid"], [65, 65])
        self.assertEqual(len(str(cavity["dataset_sha256"])), 64)
        self.assertLess(float(cavity["interior_divergence_rms"]), 1.0e-12)
        self.assertLess(float(cavity["archive_vorticity_relative_l2"]), 4.0e-2)
        self.assertEqual(evaluate_acceptance(verification, cavity)["decision"], "accept")

    def test_failed_gate_is_retained(self) -> None:
        verification = verification_sweep()
        cavity = audit_cavity_case(ROOT, 100.0)
        strict = {
            "observed_order_min": 3.0,
            "fine_vorticity_relative_l2_max": 5.0e-4,
            "manufactured_divergence_rms_max": 1.0e-12,
            "cavity_divergence_rms_max": 1.0e-12,
            "cavity_vorticity_relative_l2_max": 4.0e-2,
            "wall_velocity_max_abs_error_max": 1.0e-12,
        }
        record = evaluate_acceptance(verification, cavity, strict)
        self.assertEqual(record["decision"], "reject")
        self.assertFalse(record["gates"]["second_order_verification"])

    def test_changed_dataset_identity_fails_closed(self) -> None:
        verification = verification_sweep()
        cavity = audit_cavity_case(ROOT, 100.0)
        cavity["dataset_sha256"] = "0" * 64
        record = evaluate_acceptance(verification, cavity)
        self.assertEqual(record["decision"], "reject")
        self.assertFalse(record["gates"]["dataset_identity"])

    def test_record_is_json_and_has_no_mutable_timestamp(self) -> None:
        record = evaluate_acceptance(verification_sweep(), audit_cavity_case(ROOT))
        with tempfile.TemporaryDirectory() as folder:
            path = write_acceptance_record(record, Path(folder) / "record.json")
            recovered = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(recovered["decision"], "accept")
        self.assertNotIn("timestamp", recovered)

    def test_retained_evidence_recomputes_with_roundoff_tolerance(self) -> None:
        record = validate_week01_1_evidence(ROOT)
        self.assertEqual(record["decision"], "accept")

    def test_evidence_allows_only_diagnostic_roundoff(self) -> None:
        record = evaluate_acceptance(verification_sweep(), audit_cavity_case(ROOT))
        record["verification"]["observed_order"] += 1e-12
        record["verification"]["rows"][-1]["divergence_rms"] += 2e-14
        record["cavity"]["archive_vorticity_relative_l2"] += 1e-14
        with mock.patch("flowmllab.scientific_software.evaluate_acceptance", return_value=record):
            self.assertEqual(validate_week01_1_evidence(ROOT)["decision"], "accept")

    def test_evidence_rejects_drift_nonfinite_and_contract_changes(self) -> None:
        original = evaluate_acceptance(verification_sweep(), audit_cavity_case(ROOT))
        changes = [
            ("cavity", "archive_vorticity_relative_l2", .0291),
            ("cavity", "archive_vorticity_relative_l2", float("nan")),
            ("cavity", "reynolds", 100.000000000001),
            ("cavity", "dataset_sha256", "0" * 64),
            ("thresholds", "cavity_divergence_rms_max", 1.001e-12),
            ("gates", "cavity_incompressibility", False),
            ("gates", "cavity_incompressibility", 1),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key, value=value):
                record = deepcopy(original)
                record[section][key] = value
                with mock.patch("flowmllab.scientific_software.evaluate_acceptance", return_value=record):
                    with self.assertRaises(ValueError):
                        validate_week01_1_evidence(ROOT)


if __name__ == "__main__":
    unittest.main()
