"""The release summary must handle the real supplied-source success schema."""
import copy
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_summary_ci',ROOT/'qa/validate_course_release.py')
qa=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def test_supplied_upstream_pass_record_does_not_require_a_reason():
    record=dict(status='pass',source_commit='c3f211376b42b8dc30daad380eaef5e0ab800b5c',
                height_percent=[20,30,40,44,50,60,67,70,80],row_counts={'44':18080},
                compact_archive_match={'status':'pass','maximum_float32_quantization':{'u':0.0,'v':0.0}})
    original=copy.deepcopy(record)
    rendered=json.loads(qa.source_crosscheck_detail(record))
    assert rendered['source_commit']==record['source_commit']
    assert rendered['height_percent']==record['height_percent']
    assert rendered['compact_archive_match']['status']=='pass'
    assert record==original and 'reason' not in record


def test_missing_source_keeps_its_explicit_not_checked_reason():
    record=dict(status='not_checked',reason='pinned upstream checkout/cache was not supplied')
    assert qa.source_crosscheck_detail(record)==record['reason']
    assert record['status']=='not_checked'


def test_failure_reason_is_not_relabelled_as_success():
    record=dict(status='fail',reason='Synthetic source mismatch')
    assert qa.source_crosscheck_detail(record)=='Synthetic source mismatch'
    assert record['status']=='fail'
