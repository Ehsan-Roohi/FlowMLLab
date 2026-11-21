"""Render late-course PDFs and record identities for individual visual review.

Rendering and text-bound checks are evidence preparation, not visual approval.
Each page's review remains pending until a person/agent inspects that image.
"""
from pathlib import Path
import argparse
import hashlib
import json
import fitz

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdfs', nargs='+')
    parser.add_argument('--dpi', type=int, default=125)
    args = parser.parse_args()
    out = ROOT / 'tmp/course_refresh_late/pdf_review'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'render_manifest.json'
    records = json.loads(path.read_text()) if path.exists() else {}
    for name in args.pdfs:
        source = ROOT / name
        directory = out / source.stem
        directory.mkdir(exist_ok=True)
        pages = []
        with fitz.open(source) as doc:
            for index, page in enumerate(doc):
                pix = page.get_pixmap(dpi=args.dpi, alpha=False)
                image = directory / f'page-{index+1:03d}.png'
                pix.save(image)
                spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
                         for line in block['lines'] for span in line['spans']]
                bounds = [span['text'] for span in spans if span['text'].strip() and
                          not (fitz.Rect(span['bbox']) in (page.rect + (-1,-1,1,1)))]
                pages.append(dict(page=index+1, image_path=image.relative_to(ROOT).as_posix(),
                                  image_sha256=sha(image.read_bytes()), pixel_sha256=sha(pix.samples),
                                  text_sha256=sha(page.get_text().encode()),
                                  fonts=sorted(set(s['font'] for s in spans)),
                                  text_colors=sorted(set(s['color'] for s in spans)),
                                  out_of_bounds_text=bounds, status='pending'))
        # Other scoped render jobs may finish while this larger PDF is rendering.
        records = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        records[name] = dict(pdf_sha256=sha(source.read_bytes()), dpi=args.dpi,
                             page_count=len(pages), pages=pages)
        path.write_text(json.dumps(records, indent=2)+'\n', encoding='utf8', newline='\n')
        print(name, len(pages), 'pages rendered', flush=True)


if __name__ == '__main__':
    main()
