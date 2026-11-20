"""A rebuilt Week4 guide needs its own complete review, not legacy preservation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "week04_replacement_gate", ROOT/"qa/validate_cavity_pinn_curriculum.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    """Synthetic review fixtures never alter the real teaching receipts."""
    from pypdf import PdfWriter
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path/"qa").mkdir()
    sources = tmp_path/"lectures/source"
    sources.mkdir(parents=True)
    source = sources/"week04_lab_guide.py"
    source.write_text("# synthetic replacement source\n", encoding="utf-8")
    evidence = sources/"cavity_expansion.json"
    evidence.write_text('{"synthetic": true}\n', encoding="utf-8")
    pdf = tmp_path/"lectures/week04_cavity_surrogates_deeponet.pdf"
    writer = PdfWriter()
    for _ in range(103):
        writer.add_blank_page(width=100, height=100)
    with pdf.open("wb") as stream:
        writer.write(stream)
    legacy = dict(total_pages=65, sha256="b"*64)
    evidence_item = dict(
        path="lectures/source/cavity_expansion.json", sha256=gate.sha(evidence)
    )
    review = dict(
        schema_version=1,
        review_type="week04_full_guide_replacement",
        status="complete",
        pdf="lectures/week04_cavity_surrogates_deeponet.pdf",
        pdf_sha256=gate.sha(pdf),
        source="lectures/source/week04_lab_guide.py",
        source_sha256=gate.sha(source),
        evidence_source=evidence_item["path"],
        evidence_source_sha256=evidence_item["sha256"],
        pre_revision=dict(pages=65, sha256=legacy["sha256"]),
        guide_pages=49,
        retained_evidence_pages=54,
        total_pages=103,
        page_map=[dict(page=p, title=f"Synthetic guide {p}") for p in range(1,50)],
        appendix_map=[
            dict(id=f"E{i:02d}", page=49+i, title=f"Synthetic evidence {i}")
            for i in range(1,55)
        ],
        visual_review=dict(
            pdf_sha256=gate.sha(pdf), dpi=125, out_of_bounds_text=[],
            pages=[
                dict(page=p, image_sha256=hashlib.sha256(str(p).encode()).hexdigest(),
                     reviewer="synthetic test fixture", status="passed",
                     review_basis="Synthetic fixture only; not an actual page inspection")
                for p in range(1,104)
            ],
        ),
    )

    def run(value=None):
        receipt = review if value is None else value
        (tmp_path/"qa/week04_lecture_review.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        return gate.verify_week04_replacement(legacy, evidence_item)

    return tmp_path, review, run


def test_complete_replacement_does_not_claim_old_pages_preserved(reviewed):
    _, _, run = reviewed
    result = run()
    assert result["pdf"]["pages"] == 103
    assert result["pdf"]["preserved_pages"] == 0
    assert result["pdf"]["individually_reviewed_pages"] == 103
    assert result["pdf"]["available_render_files_checked"] == 0
    assert result["source"]["path"] == "lectures/source/week04_lab_guide.py"


@pytest.mark.parametrize("status", ["built_pending_visual_review", "pending", "failed"])
def test_pending_or_failed_replacement_review_rejected(reviewed, status):
    _, review, run = reviewed
    review["status"] = status
    with pytest.raises(ValueError, match="pending or incomplete"):
        run()


@pytest.mark.parametrize("field", ["source", "pdf", "evidence_source"])
def test_changed_current_artifact_rejected(reviewed, field):
    root, review, run = reviewed
    path = root/review[field]
    path.write_bytes(path.read_bytes()+b"\nchanged after review\n")
    with pytest.raises(ValueError, match="Reviewed teaching artifact changed"):
        run()


def test_rebinding_changed_science_does_not_bypass_legacy_evidence_identity(reviewed):
    root, review, run = reviewed
    path = root/review["evidence_source"]
    path.write_text('{"changed": true}\n', encoding="utf-8")
    review["evidence_source_sha256"] = gate.sha(path)
    with pytest.raises(ValueError, match="scientific evidence source differs"):
        run()


@pytest.mark.parametrize("failure", [
    "missing_page", "duplicate_page", "pending_page", "no_reviewer",
    "no_basis", "bad_render_hash", "stale_visual_pdf", "low_dpi",
    "unchecked_layout", "missing_guide_map", "missing_evidence_map",
    "wrong_legacy_pdf", "wrong_evidence_count", "unknown_schema",
])
def test_incomplete_or_unbound_replacement_review_rejected(reviewed, failure):
    _, original, run = reviewed
    review = copy.deepcopy(original)
    visual = review["visual_review"]
    if failure == "missing_page":
        visual["pages"].pop()
    elif failure == "duplicate_page":
        visual["pages"][-1]["page"] = 1
    elif failure == "pending_page":
        visual["pages"][0]["status"] = "pending"
    elif failure == "no_reviewer":
        visual["pages"][0]["reviewer"] = ""
    elif failure == "no_basis":
        visual["pages"][0]["review_basis"] = ""
    elif failure == "bad_render_hash":
        visual["pages"][0]["image_sha256"] = "unverified"
    elif failure == "stale_visual_pdf":
        visual["pdf_sha256"] = "c"*64
    elif failure == "low_dpi":
        visual["dpi"] = 72
    elif failure == "unchecked_layout":
        del visual["out_of_bounds_text"]
    elif failure == "missing_guide_map":
        review["page_map"].pop()
    elif failure == "missing_evidence_map":
        review["appendix_map"].pop()
    elif failure == "wrong_legacy_pdf":
        review["pre_revision"]["sha256"] = "c"*64
    elif failure == "wrong_evidence_count":
        review["retained_evidence_pages"] = 53
    elif failure == "unknown_schema":
        review["schema_version"] = 2
    with pytest.raises(ValueError):
        run(review)


def test_present_page_render_must_match_recorded_identity(reviewed):
    root, review, run = reviewed
    render = root/"tmp/page-001.png"
    render.parent.mkdir()
    render.write_bytes(b"synthetic render bytes")
    row = review["visual_review"]["pages"][0]
    row["image_path"] = "tmp/page-001.png"
    row["image_sha256"] = gate.sha(render)
    assert run()["pdf"]["available_render_files_checked"] == 1
    render.write_bytes(b"stale render bytes")
    with pytest.raises(ValueError, match="Reviewed teaching artifact changed"):
        run()


def test_render_reference_cannot_leave_checkout(reviewed):
    _, review, run = reviewed
    review["visual_review"]["pages"][0]["image_path"] = "../other-project/page.png"
    with pytest.raises(ValueError, match="render leaves the checkout"):
        run()
