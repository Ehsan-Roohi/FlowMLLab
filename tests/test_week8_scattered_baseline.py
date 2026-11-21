"""Strict frozen-identity checks for the isolated Week 8 numerical replay."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qa.run_week8_scattered_baseline import (
    BASELINE_COLUMNS,
    METRIC_COLUMNS,
    REPRODUCTION_ATOL,
    REPRODUCTION_RTOL,
    compare_baseline,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results/gas_dynamics_week8/scattered_baseline.csv"


@pytest.fixture
def csv_pair(tmp_path):
    reference = tmp_path / "reference.csv"
    generated = tmp_path / "generated.csv"
    reference.write_bytes(FROZEN.read_bytes())
    generated.write_bytes(FROZEN.read_bytes())
    return generated, reference


def edit_cell(path, column, value, row=0):
    with path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.reader(stream))
    records[row + 1][records[0].index(column)] = value
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(records)


def test_identical_csv_and_observed_ci_roundoff_pass_without_writes(csv_pair):
    generated, reference = csv_pair
    original_reference = reference.read_bytes()
    assert compare_baseline(generated, reference)["status"] == "pass"
    # Actual final-digit differences from the failed Linux CI reproduction.
    edit_cell(generated, "relative_l2", "0.00012958960317506222")
    edit_cell(generated, "relative_linf", "0.0013593794717499566")
    candidate_before = generated.read_bytes()
    report = compare_baseline(generated, reference)
    assert report["status"] == "pass"
    assert report["rows"] == 4
    assert report["numeric_columns"] == list(METRIC_COLUMNS)
    assert reference.read_bytes() == original_reference
    assert generated.read_bytes() == candidate_before


@pytest.mark.parametrize("column,value", [
    ("physical_inputs", "log_p4_p1;gamma"),
    ("training_seed", "62"),
    ("blind_seed", "144"),
    ("training_states", "4095"),
    ("blind_test_states", "2047"),
    ("method", "different_method"),
    ("neighbors", "129"),
    ("smoothing", "1.00000000001e-08"),
    ("same_random_training_budget_as_mlp", "False"),
    ("training_states", "4096.0"),
])
def test_any_metadata_change_fails_exactly(csv_pair, column, value):
    generated, reference = csv_pair
    edit_cell(generated, column, value)
    with pytest.raises(ValueError, match="metadata mismatch"):
        compare_baseline(generated, reference)


@pytest.mark.parametrize("column", METRIC_COLUMNS)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-0.001", "not-a-number"])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_metrics_fail_in_both_reference_and_candidate(csv_pair, column, value, side):
    edit_cell(csv_pair[side], column, value)
    with pytest.raises(ValueError, match="baseline metric"):
        compare_baseline(*csv_pair)


@pytest.mark.parametrize("column", METRIC_COLUMNS)
def test_just_outside_roundoff_bound_fails(csv_pair, column):
    generated, reference = csv_pair
    with reference.open(newline="", encoding="utf-8") as stream:
        old = float(next(csv.DictReader(stream))[column])
    permitted = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(old)
    edit_cell(generated, column, repr(old + 1.1 * permitted))
    with pytest.raises(ValueError, match="metric drift"):
        compare_baseline(generated, reference)


@pytest.mark.parametrize("column", METRIC_COLUMNS)
def test_substantive_metric_drift_fails(csv_pair, column):
    generated, reference = csv_pair
    edit_cell(generated, column, "0.1")
    with pytest.raises(ValueError, match="metric drift"):
        compare_baseline(generated, reference)


@pytest.mark.parametrize("change", ["header_order", "extra_column", "missing_row", "extra_row", "row_order", "row_width"])
def test_schema_and_case_set_fail_closed(csv_pair, change):
    generated, reference = csv_pair
    with generated.open(newline="", encoding="utf-8") as stream:
        records = list(csv.reader(stream))
    if change == "header_order":
        records[0][0], records[0][1] = records[0][1], records[0][0]
    elif change == "extra_column":
        records[0].append("unexpected")
    elif change == "missing_row":
        records.pop()
    elif change == "extra_row":
        records.append(records[-1])
    elif change == "row_order":
        records[1], records[2] = records[2], records[1]
    else:
        records[1].append("unexpected")
    with generated.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream, lineterminator="\n").writerows(records)
    with pytest.raises(ValueError, match="Baseline (schema|row count|case order|row width) mismatch"):
        compare_baseline(generated, reference)


def test_cli_refuses_to_overwrite_its_reference(csv_pair):
    _, reference = csv_pair
    before = reference.read_bytes()
    with pytest.raises(SystemExit) as raised:
        main(["--output", str(reference), "--compare-to", str(reference)])
    assert raised.value.code == 2
    assert reference.read_bytes() == before
    assert len(BASELINE_COLUMNS) == 12
