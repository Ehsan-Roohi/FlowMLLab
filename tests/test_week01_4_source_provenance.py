"""Retained Week 1.4 provenance accepts newline conversion, never data changes."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('week01_4_provenance_qa', ROOT/'qa/validate_course_release.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
SOURCE_LF = (ROOT/'references/ghia/published_samples.csv').read_bytes().replace(b'\r\n', b'\n')
NOTEBOOK = json.loads((ROOT/'notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb').read_text(encoding='utf-8'))
TREE = ast.parse('\n'.join(''.join(cell['source']) for cell in NOTEBOOK['cells'] if cell['cell_type']=='code'))
ASSIGNMENTS = {target.id: node.value for node in TREE.body if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Name)}
REFERENCE = ast.literal_eval(ASSIGNMENTS['GHIA_REFERENCE'])


def representation(ending):
    return SOURCE_LF if ending == 'LF' else SOURCE_LF.replace(b'\n', b'\r\n')


@pytest.mark.parametrize('checkout_ending', ['LF', 'CRLF'])
@pytest.mark.parametrize('execution_ending', ['LF', 'CRLF'])
def test_lf_and_crlf_source_representations_preserve_exact_samples(checkout_ending, execution_ending):
    recorded = hashlib.sha256(representation(execution_ending)).hexdigest()
    report = qa.validate_week01_4_ghia_source(representation(checkout_ending), recorded, REFERENCE)
    assert report['execution_byte_sha256'] == recorded
    assert report['execution_line_ending'] == execution_ending
    assert report['checkout_line_ending'] == checkout_ending
    assert report['source_content_lf_sha256'] == hashlib.sha256(SOURCE_LF).hexdigest()
    assert report['embedded_samples_verified'] is True


@pytest.mark.parametrize('ending', ['LF', 'CRLF'])
def test_a_changed_numeric_sample_fails_even_with_allowed_line_endings(ending):
    original = representation(ending)
    assert b'0.84123' in original
    altered = original.replace(b'0.84123', b'0.84124', 1)
    with pytest.raises(AssertionError, match='content differs'):
        qa.validate_week01_4_ghia_source(altered, hashlib.sha256(original).hexdigest(), REFERENCE)


@pytest.mark.parametrize('altered', [SOURCE_LF.replace(b'\n', b'\r'), SOURCE_LF.replace(b'\n', b'\r\n', 1)])
def test_only_complete_lf_or_crlf_forms_are_accepted(altered):
    with pytest.raises(AssertionError, match='line endings'):
        qa.validate_week01_4_ghia_source(altered, hashlib.sha256(SOURCE_LF).hexdigest())


def test_valid_source_hash_does_not_mask_a_changed_embedded_sample():
    changed = copy.deepcopy(REFERENCE)
    changed[100]['u'][1] += 0.00001
    with pytest.raises(AssertionError, match='Embedded Ghia samples differ'):
        qa.validate_week01_4_ghia_source(SOURCE_LF, hashlib.sha256(SOURCE_LF).hexdigest(), changed)


def test_actual_standalone_notebook_and_retained_evidence_share_source_content():
    report = qa.validate_week01_4_standalone(NOTEBOOK, 'notebooks/week01_4/W1_4_Spectral_DG_CPU.ipynb')
    evidence = json.loads((ROOT/'results/week01_4_spectral_dg/execution.json').read_text())
    evidence_source = qa.validate_week01_4_ghia_source(SOURCE_LF, evidence['ghia_source_sha256'])
    assert report['ghia_source']['execution_byte_sha256'] == ast.literal_eval(ASSIGNMENTS['GHIA_SOURCE_SHA256'])
    assert report['ghia_source']['source_content_lf_sha256'] == evidence_source['source_content_lf_sha256']
    assert report['ghia_source']['embedded_samples_verified'] is True