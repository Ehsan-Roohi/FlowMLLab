"""Build the coherent Week 4 lab guide, followed by retained scientific evidence.

Run from the checkout with the lecture/PDF dependencies and licensed Times New
Roman installed. Source prose is lectures/source/week04_lab_guide.py. The 54
retained evidence pages are generated from the unchanged cavity_expansion.json.
No simulation or model fitting is performed by this document builder.
"""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import hashlib
import importlib.util
import json
import os
import re
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))
os.environ.setdefault("FLOWMLLAB_TNR_DIR", "C:/Windows/Fonts" if os.name == "nt" else "")
from lecture_typography import register_reportlab, configure_matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image, Table, TableStyle, Flowable, KeepTogether)
from pypdf import PdfReader, PdfWriter
import matplotlib
matplotlib.use("Agg")
from matplotlib import mathtext, font_manager

TARGET = ROOT / "lectures/week04_cavity_surrogates_deeponet.pdf"
SOURCE = ROOT / "lectures/source/week04_lab_guide.py"
WORK = ROOT / "tmp/week04_lecture"
WORK.mkdir(parents=True, exist_ok=True)
FONTS = register_reportlab("W4GuideTNR")
configure_matplotlib()
W, H = A4
BODY = W - 108
STYLES = {
    "title": ParagraphStyle("w4title", fontName=FONTS["bold"], fontSize=18.5, leading=21.8, spaceAfter=11, textColor=colors.black),
    "sub": ParagraphStyle("w4sub", fontName=FONTS["bold"], fontSize=12, leading=15, spaceBefore=6, spaceAfter=6, textColor=colors.black),
    "body": ParagraphStyle("w4body", fontName=FONTS["regular"], fontSize=10.7, leading=14.25, spaceAfter=8, textColor=colors.black),
    "small": ParagraphStyle("w4small", fontName=FONTS["regular"], fontSize=9.2, leading=11.8, spaceAfter=6, textColor=colors.black),
    "caption": ParagraphStyle("w4caption", fontName=FONTS["regular"], fontSize=9.5, leading=12.3, spaceAfter=9, textColor=colors.black),
}


class Schematic(Flowable):
    """Vector boxes and arrows; scientific values never come from this diagram."""
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.width = BODY
        self.height = len(rows) * 66 + 4
    def draw(self):
        c = self.canv
        for r, row in enumerate(self.rows):
            n = len(row)
            gap = 22
            bw = (BODY - gap * (n - 1)) / n
            y = self.height - (r + 1) * 66 + 9
            for i, label in enumerate(row):
                x = i * (bw + gap)
                c.setStrokeColor(colors.black)
                c.setLineWidth(.6)
                c.roundRect(x, y, bw, 49, 3, fill=0, stroke=1)
                p = Paragraph(label, STYLES["small"])
                _, ph = p.wrap(bw - 12, 45)
                p.drawOn(c, x + 6, y + (49 - ph) / 2 + 2)
                if i < n - 1:
                    yy = y + 24
                    c.line(x + bw + 3, yy, x + bw + gap - 3, yy)
                    c.line(x + bw + gap - 7, yy + 3, x + bw + gap - 3, yy)
                    c.line(x + bw + gap - 7, yy - 3, x + bw + gap - 3, yy)


def equation(expression):
    expression = re.sub(r'\\(mathcal|mathbf|boldsymbol|widehat|overline|widetilde)\s+([A-Za-z])', r'\\\1{\2}', expression)
    expression = re.sub(r'\\(boldsymbol|widehat|overline)(\\[A-Za-z]+)', r'\\\1{\2}', expression)
    stream = BytesIO()
    mathtext.math_to_image("$" + expression + "$", stream,
        prop=font_manager.FontProperties(size=12.5), dpi=300, format="png", color="black")
    stream.seek(0)
    w, h = ImageReader(stream).getSize()
    scale = min(72 / 300, BODY / w)
    return [Spacer(1, 3), Image(stream, width=w * scale, height=h * scale), Spacer(1, 8)]


def figure(block):
    path = ROOT / block["path"]
    if not path.is_file():
        raise FileNotFoundError("Lecture figure must exist: " + str(path))
    w, h = ImageReader(str(path)).getSize()
    width = min(BODY, block.get("width", BODY))
    height = width * h / w
    if height > block.get("max_height", 340):
        height = block.get("max_height", 340)
        width = height * w / h
    return [Image(str(path), width=width, height=height), Spacer(1, 5),
            Paragraph(block["caption"], STYLES["caption"])]


def make_table(block):
    rows = [[Paragraph(str(v), STYLES["small"]) for v in row] for row in block["rows"]]
    widths = block.get("widths")
    if widths:
        widths = [BODY * x / sum(widths) for x in widths]
    else:
        widths = [BODY / len(rows[0])] * len(rows[0])
    t = Table(rows, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LINEABOVE", (0,0), (-1,0), .7, colors.black),
        ("LINEBELOW", (0,0), (-1,0), .5, colors.black),
        ("LINEBELOW", (0,-1), (-1,-1), .7, colors.black),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    return [t, Spacer(1, 9)]


def build():
    before = {}
    if TARGET.exists():
        old = PdfReader(TARGET)
        before = {"pages": len(old.pages), "sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest()}
    receipt_path = ROOT / "qa/week04_lecture_review.json"
    prior = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    source_sha256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("w4content", SOURCE)
    content = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(content)
    pages = content.get_pages(ROOT)
    story = []
    for i, page in enumerate(pages):
        if story:
            story.append(PageBreak())
        story.append(Paragraph(page["title"], STYLES["title"]))
        if page.get("notebook"):
            url = "https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week04/" + page["notebook"]
            story.append(Paragraph('<link href="' + url + '" color="black"><u>Open this lab notebook</u></link> | ' + escape(page.get("section", "")), STYLES["small"]))
        for block in page["blocks"]:
            kind = block["type"]
            if kind in STYLES:
                story.append(Paragraph(block["text"], STYLES[kind]))
            elif kind == "math":
                story.extend(equation(block["latex"]))
            elif kind == "figure":
                story.extend(figure(block))
            elif kind == "table":
                story.extend(make_table(block))
            elif kind == "diagram":
                story.extend([Schematic(block["rows"]), Spacer(1, 7)])
            else:
                raise ValueError(kind)
    def footer(c, doc):
        c.setFillColor(colors.black)
        c.setStrokeColor(colors.black)
        c.setFont(FONTS["regular"], 9)
        c.drawString(54, H - 30, "FlowMLLab | Week 4 | Cavity learning laboratories")
        c.line(54, H - 39, W - 54, H - 39)
        c.drawCentredString(W / 2, 26, str(doc.page))
    core_path = WORK / "guide.pdf"
    doc = SimpleDocTemplate(str(core_path), pagesize=A4, leftMargin=54, rightMargin=54,
        topMargin=56, bottomMargin=48, title="FlowMLLab Week 4: a guide to all nine cavity laboratories",
        author="Ehsan Roohi")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(core_path)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_sha256:
        raise RuntimeError("Lecture source changed during the build; rerun after edits finish")
    if len(reader.pages) != len(pages):
        preview = []
        for i, p in enumerate(reader.pages):
            preview.append((i+1, (p.extract_text() or "").splitlines()[:4]))
        (WORK / "overflow.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
        raise RuntimeError(f"Page overflow: rendered {len(reader.pages)}, expected {len(pages)}; inspect tmp/week04_lecture/overflow.json")
    writer = PdfWriter()
    writer.append(reader)
    for i, p in enumerate(pages):
        writer.add_outline_item(p["title"], i)
    writer.add_metadata({"/Title": "Week 4: cavity surrogate laboratories and retained evidence", "/Author": "Ehsan Roohi",
        "/FlowMLLabWeek4GuidePages": str(len(pages)), "/FlowMLLabCavityBasePages": str(len(pages)),
        "/FlowMLLabWeek4GuideSHA256": source_sha256})
    with TARGET.open("wb") as f:
        writer.write(f)
    from build_cavity_expansion_lectures import build as build_appendix
    build_appendix([TARGET.stem])
    final = PdfReader(TARGET)
    append_source = json.loads((ROOT / "lectures/source/cavity_expansion.json").read_text(encoding="utf-8"))[TARGET.stem]
    writer = PdfWriter()
    writer.append(final, import_outline=True)
    parent = writer.add_outline_item("Retained scientific evidence", len(pages))
    for i, page in enumerate(append_source):
        writer.add_outline_item(f"E{i+1:02d}. {page['title']}", len(pages)+i, parent=parent)
    writer.add_metadata(dict(final.metadata))
    payload = BytesIO()
    writer.write(payload)
    TARGET.write_bytes(payload.getvalue())
    final = PdfReader(TARGET)
    # Bookmarks change the PDF bytes after the appendix builder records its hash.
    # Bind the shared lecture manifest to the actual final artifact.
    manifest_path = ROOT / "qa/cavity_lecture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[TARGET.stem]["sha256"] = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record = {
        "status": "built_pending_visual_review", "pdf": TARGET.relative_to(ROOT).as_posix(),
        "pre_revision": prior.get("pre_revision", before),
        "guide_pages": len(pages), "retained_evidence_pages": len(append_source), "total_pages": len(final.pages),
        "source": SOURCE.relative_to(ROOT).as_posix(), "source_sha256": source_sha256,
        "pdf_sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "evidence_source": "lectures/source/cavity_expansion.json",
        "evidence_source_sha256": hashlib.sha256((ROOT / "lectures/source/cavity_expansion.json").read_bytes()).hexdigest(),
        "preservation": "Original teaching core expanded and reorganized; all 54 scientific evidence pages regenerated from unchanged scientific JSON with documented typography-only substitutions. Classical ROM reading guide integrated into the new core.",
        "evidence_presentation": "qa/week04_evidence_presentation.py",
        "evidence_presentation_sha256": hashlib.sha256((ROOT / "qa/week04_evidence_presentation.py").read_bytes()).hexdigest(),
        "page_map": [{"page":i+1,"title":p["title"],"notebook":p.get("notebook")} for i,p in enumerate(pages)],
        "appendix_map": [{"id":f"E{i+1:02d}","page":len(pages)+i+1,"title":p["title"]} for i,p in enumerate(append_source)],
    }
    receipt_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:record[k] for k in ["guide_pages","retained_evidence_pages","total_pages","pdf_sha256"]}, indent=2))


if __name__ == "__main__":
    build()
