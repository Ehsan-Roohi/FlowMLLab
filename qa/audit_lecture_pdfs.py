"""Read-only PDF typography/searchability audit; JSON evidence, no auto repair."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import fitz


def audit(path: Path) -> dict:
    document = fitz.open(path)
    font_counts, colors = Counter(), Counter()
    pages, outside, missing, embedded = [], [], [], {}
    for number, page in enumerate(document, 1):
        characters = 0
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"]
                    if not text.strip():
                        continue
                    characters += len(text)
                    font_counts[span["font"]] += len(text)
                    colors[f"#{span['color']:06x}"] += len(text)
                    bbox = fitz.Rect(span["bbox"])
                    if not (page.rect + (-1, -1, 1, 1)).contains(bbox):
                        outside.append({"page": number, "text": text[:140], "bbox": list(bbox)})
                    if any(c in text for c in ("\ufffd", "\u25a0", "\x00")):
                        missing.append({"page": number, "text": text[:140]})
        pages.append({"page": number, "searchable_characters": characters,
                      "image_count": len(page.get_images()),
                      "width": page.rect.width, "height": page.rect.height})
        for info in page.get_fonts(full=True):
            xref, extension, kind, basefont, resource, encoding, *_ = info
            embedded[str(xref)] = {"font": basefont, "type": kind,
                                  "embedded": extension != "n/a", "encoding": encoding}
    total = sum(font_counts.values())
    tnr = sum(count for name, count in font_counts.items() if "TimesNewRoman" in name.replace(" ", ""))
    document.close()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "page_count": len(pages), "pages": pages, "text_characters": total,
            "tnr_text_fraction": round(tnr / total, 4) if total else 0,
            "text_fonts": dict(font_counts.most_common()), "text_colors": dict(colors.most_common()),
            "low_text_pages": [p["page"] for p in pages if p["searchable_characters"] < 80],
            "outside_page_text": outside, "missing_glyph_candidates": missing,
            "font_resources": list(embedded.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted({pdf for path in args.paths for pdf in
                    (path.glob("*.pdf") if path.is_dir() else [path])})
    reports = [audit(path) for path in paths]
    result = {"note": "Read-only evidence. Font ratios include text in embedded vector figures; visual review is still required.",
              "files": reports,
              "totals": {"files": len(reports), "pages": sum(r["page_count"] for r in reports),
                         "image_only_files": [r["path"] for r in reports if not r["text_characters"]],
                         "outside_page_text_count": sum(len(r["outside_page_text"]) for r in reports),
                         "missing_glyph_candidate_count": sum(len(r["missing_glyph_candidates"]) for r in reports)}}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(json.dumps(result["totals"], indent=2))
    else:
        print(encoded)


if __name__ == "__main__":
    main()
