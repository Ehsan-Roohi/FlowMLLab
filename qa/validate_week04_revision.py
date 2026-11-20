"""Check the executed Week 4 notebooks, linked lecture and frozen study identity."""
from pathlib import Path
import ast
import base64
import hashlib
import json

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_lab2_presentation(execution):
    """Bind the saved layout revision to the actual, unchanged CPU experiment."""
    record = json.loads((ROOT / "qa/week04_lab2_presentation.json").read_text())
    assert record["training_rerun"] is False
    for key in ("model_arrays_weights_metrics_histories_unchanged",
                "scientific_source_ast_unchanged", "notebook_code_unchanged"):
        assert record[key] is True, key
    assert record["original_executed_notebook_sha256"] == execution["notebook_sha256"]
    notebook = ROOT / execution["notebook"]
    doc = nbformat.read(notebook, as_version=4)
    code = "\n".join(c.source for c in doc.cells if c.cell_type == "code")
    assert hashlib.sha256(code.encode()).hexdigest() == execution["executed_code_sha256"]
    assert record["executed_code_sha256"] == execution["executed_code_sha256"]
    assert digest(notebook) == record["presented_notebook_sha256"]
    helper = ROOT / "flowmllab/cavity_lab2.py"
    original = ROOT / record["retained_training_helper_source"]
    assert original.resolve().is_relative_to(ROOT.resolve())
    assert digest(original) == execution["helper_source_sha256"] == record["training_helper_sha256"]
    assert digest(helper) == record["presentation_helper_sha256"]

    def scientific_ast(path):
        tree = ast.parse(path.read_bytes())
        tree.body = [node for node in tree.body
                     if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     or node.name not in {"plot_profiles", "plot_wall_diagnostics"}]
        return ast.dump(tree, include_attributes=False)

    assert scientific_ast(original) == scientific_ast(helper), "Lab 2 science changed after execution"
    results = ROOT / "results/week04_lab2"
    frozen = json.loads((results / "frozen_models.json").read_text())
    assert frozen["helper_source_sha256"] == record["training_helper_sha256"]
    protected = {"native_test_predictions.npz", "scalar_predictions.npz", "frozen_weights.npz",
                 "frozen_models.json", "field_metrics.csv", "week4_scalar_results.csv",
                 "training_histories.json"}
    assert set(record["protected_scientific_artifacts"]) == protected
    revised = {f"{stem}_re{re}.{suffix}" for stem in ("profiles", "wall_diagnostics")
               for re in (175, 275, 375) for suffix in ("png", "pdf")}
    assert set(record["revised_artifacts"]) == revised
    inline = {hashlib.sha256(base64.b64decode(o["data"]["image/png"])).hexdigest()
              for cell in doc.cells if cell.cell_type == "code" for o in cell.outputs
              if "image/png" in o.get("data", {})}
    for group in ("protected_scientific_artifacts", "revised_artifacts"):
        for name, expected in record[group].items():
            assert digest(results / name) == expected, (name, "Lab 2 artifact changed")
            if name.endswith(".png"):
                assert expected in inline, (name, "saved figure differs from notebook output")
    assert digest(ROOT / "qa/week04_lab2_replot.py") == record["replot_script_sha256"]
    return record


def check():
    evidence = json.loads((ROOT / "results/week04_teaching_outputs/execution.json").read_text())
    lab2 = json.loads((ROOT / "qa/week04_lab2_execution.json").read_text())
    lab2_presentation = check_lab2_presentation(lab2)
    lab5 = json.loads((ROOT / "results/week04_corner_audit/verification.json").read_text())
    records = {r["notebook"].replace("\\", "/"): r for r in evidence.values()}
    records[lab2["notebook"].replace("\\", "/")] = {
        "sha256": lab2_presentation["presented_notebook_sha256"],
        "elapsed_seconds": lab2["elapsed_run_all_seconds"],
        "scope": lab2["claim"] + " " + lab2_presentation["claim"],
    }
    records["notebooks/week04/W4_Lab5_All_Re_Ghia_Validation.ipynb"] = {
        "sha256": lab5["notebook_sha256"], "elapsed_seconds": lab5["elapsed_seconds"],
        "scope": lab5["execution"]["method"],
    }
    rows = []
    for path in sorted((ROOT / "notebooks/week04").glob("*.ipynb")):
        relative = path.relative_to(ROOT).as_posix()
        doc = nbformat.read(path, as_version=4)
        nbformat.validate(doc)
        codes = [c for c in doc.cells if c.cell_type == "code"]
        errors = [o for c in codes for o in c.outputs if o.output_type == "error"]
        plots = sum("image/png" in o.get("data", {}) for c in codes for o in c.outputs)
        assert codes and all(c.execution_count is not None for c in codes), (relative, "unexecuted cell")
        assert not errors, (relative, "error output")
        assert plots >= 3, (relative, "missing physical output")
        assert relative in records, (relative, "missing fresh execution receipt")
        record = records[relative]
        assert digest(path) == record["sha256"], (relative, "execution receipt does not match saved notebook")
        text = "\n".join(c.source for c in doc.cells if c.cell_type == "markdown")
        assert "aim" in text.lower(), (relative, "missing aim")
        rows.append({"notebook": relative, "sha256": record["sha256"], "code_cells": len(codes),
                     "figures": plots, "elapsed_seconds": record.get("elapsed_seconds"),
                     "execution_scope": record.get("scope", "local CPU Run All")})
    assert len(rows) == 11, "Nine numbered labs and the two retained companions must remain"
    for relative, expected in lab5["checked_sha256"].items():
        assert digest(ROOT / relative) == expected, (relative, "frozen scientific evidence changed")
    lecture_record = json.loads((ROOT / "qa/week04_lecture_review.json").read_text())
    assert lecture_record["status"] == "complete", "Lecture visual review is incomplete"
    assert lecture_record["review_type"] == "week04_full_guide_replacement"
    assert digest(ROOT / lecture_record["source"]) == lecture_record["source_sha256"]
    assert digest(ROOT / lecture_record["evidence_presentation"]) == lecture_record["evidence_presentation_sha256"]
    pdf = ROOT / lecture_record["pdf"]
    reader = PdfReader(pdf)
    assert digest(pdf) == lecture_record["pdf_sha256"]
    assert len(reader.pages) == lecture_record["total_pages"]
    assert lecture_record["retained_evidence_pages"] == 54
    assert digest(ROOT / lecture_record["evidence_source"]) == lecture_record["evidence_source_sha256"]
    visual = lecture_record["visual_review"]
    assert visual["pdf_sha256"] == lecture_record["pdf_sha256"]
    assert visual["dpi"] >= 125 and visual["out_of_bounds_text"] == []
    assert len(visual["pages"]) == len(reader.pages)
    assert {p["page"] for p in visual["pages"]} == set(range(1, len(reader.pages)+1))
    assert all(p["status"] == "passed" and p["reviewer"] and p["review_basis"]
               for p in visual["pages"]), "Every lecture page needs an individual visual review"
    # Every numbered lab has a page-level notebook link in the rebuilt guide.
    for number in range(1, 10):
        assert any(f"W4_Lab{number}_" in (p.get("notebook") or "")
                   for p in lecture_record["page_map"]), (number, "missing lecture mapping")
    result = {"status": "pass", "notebooks": rows, "notebook_count": len(rows),
              "total_inline_figures": sum(r["figures"] for r in rows),
              "lecture_pages": len(reader.pages), "lecture_sha256": digest(pdf),
              "retained_evidence_pages": 54, "corner_study_source_hashes_verified": len(lab5["checked_sha256"]),
              "hosted_colab_execution_claimed": False,
              "scope": "Complete CPU notebook executions on the hosts recorded in their receipts; retained research evidence is distinct from new fitting/CFD. Visual review is recorded separately."}
    (ROOT / "qa/week04_revision_validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "notebook_count", "total_inline_figures", "lecture_pages")}, indent=2))
    return result


if __name__ == "__main__":
    check()
