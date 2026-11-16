"""Release gating for absent or incomplete PINN experiment evidence."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pinn_release_validation', ROOT/'qa/validate_cavity_pinn_studies.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)


def test_pending_variant_is_explicit_and_never_a_pass(tmp_path):
    result = v.validate_stratified(tmp_path/'missing.json', tmp_path/'baseline.json')
    assert result['integrity'] == 'pending'
    with pytest.raises(ValueError, match='Required stratified experiment'):
        v.validate_stratified(tmp_path/'missing.json', tmp_path/'baseline.json', required=True)


def test_duplicate_case_cannot_complete_a_twelve_run_matrix(tmp_path):
    path = tmp_path/'experiment.json'
    path.write_text(json.dumps(dict(schema_version=1, campaign_kind='ghia_stratified_matched_work',
        records=[dict(Re=100, seed=7, collar=.01)]*12)))
    with pytest.raises(ValueError, match='12 unique'):
        v.validate_stratified(path, tmp_path/'baseline.json', required=True)


def test_public_evidence_cannot_escape_declared_repository(tmp_path):
    outside = tmp_path/'outside.json'; outside.write_text('{}')
    with pytest.raises(ValueError, match='leaves its declared base'):
        v.checked(dict(path=str(outside), sha256=v.sha(outside)))
