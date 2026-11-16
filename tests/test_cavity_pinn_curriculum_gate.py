"""Final curriculum gates reject incomplete results and stale lesson reviews."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pinn_curriculum_gate',ROOT/'qa/validate_cavity_pinn_curriculum.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def ghia():
    return [dict(Re=re,seed=seed,collar=eps,outcome='scored') for re in (100,400,1000,3200,5000,7500,10000)
            for seed in (7,23,61) for eps in (.01,.02)]


def stratified():
    return [dict(Re=re,seed=seed,collar=eps) for re in (100,7500) for seed in (7,23,61) for eps in (.01,.02)]


def test_complete_ghia_alone_does_not_certify_new_stratified_lesson():
    namespace=dict(CAMPAIGN=dict(records=ghia()),STRATIFIED=dict(records=stratified()[:-1]))
    with pytest.raises(AssertionError): exec(c.probe_source('ghia'),namespace)
    namespace['STRATIFIED']['records']=stratified()
    exec(c.probe_source('ghia'),namespace)


def test_unscored_cost_smoke_is_not_a_completed_d5_teaching_result():
    records=[dict(seed=seed,method=method,outcome='cost_smoke_unscored')
             for seed in (12345,23456,34567) for method in ('adam_lbfgs','soap_ssb')]
    namespace=dict(CAMPAIGN=dict(records=records))
    with pytest.raises(AssertionError): exec(c.probe_source('d5'),namespace)
    for row in records: row['outcome']='controlled_pilot_scored'
    exec(c.probe_source('d5'),namespace)


def test_stale_pending_lecture_review_blocks_release_before_execution(tmp_path,monkeypatch):
    qa=tmp_path/'qa';qa.mkdir()
    validator=qa/'validate_cavity_pinn_studies.py';validator.write_text('synthetic')
    report=dict(validator_sha256=c.sha(validator),all_studies_present=True,
        release_evidence_check='passed_for_available_evidence',
        **{key:dict(integrity='passed') for key in ('ghia','d5','stratified')})
    (qa/'cavity_pinn_studies_validation.json').write_text(json.dumps(report))
    (qa/'cavity_pinn_lecture_review.json').write_text(json.dumps(dict(status='D5 pending')))
    monkeypatch.setattr(c,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='lecture review is pending'):c.verify_reviews()



def test_standalone_figure_pdf_requires_current_hash_and_every_page_review(tmp_path,monkeypatch):
    from pypdf import PdfWriter
    path=tmp_path/'figure.pdf'
    writer=PdfWriter();writer.add_blank_page(width=100,height=100)
    with path.open('wb') as stream:writer.write(stream)
    item=dict(path='figure.pdf',sha256=c.sha(path),pages=1,individual_page_reviews=[])
    monkeypatch.setattr(c,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='Every standalone figure PDF page'):c.verify_figure_pdf(item)
    item['individual_page_reviews']=[dict(page=1,png_sha256='a'*64,status='Passed synthetic review')]
    assert c.verify_figure_pdf(item)['individually_reviewed_pages']==1
    path.write_bytes(path.read_bytes()+b'\n% Changed after review\n')
    with pytest.raises(ValueError,match='Reviewed teaching artifact changed'):c.verify_figure_pdf(item)
