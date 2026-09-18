"""Check executed notebooks, local links and expanded Week 11/12 PDF notes.

Use --render to produce local PDF review sheets under ignored tmp/.
"""
from pathlib import Path
import argparse
import re
import subprocess
import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def verify(render=False):
    for week in (11,12):
        name = 'W11_Shock_Vortex_Identification.ipynb' if week == 11 else 'W12_DSMC_Moment_Reconstruction.ipynb'
        path = ROOT / 'notebooks' / f'week{week}' / name
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert len({c.id for c in nb.cells}) == len(nb.cells)
        cells = [c for c in nb.cells if c.cell_type == 'code']
        assert all(c.execution_count for c in cells)
        assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
        assert any('image/png' in o.get('data',{}) for c in cells for o in c.outputs)
        for c in cells:
            for o in c.outputs:
                if o.output_type == 'stream' and o.get('name') == 'stderr':
                    print('NOTEBOOK WARNING:', o.text)
        print(f'Week {week}: {len(cells)} executed code cells, no errors')
        pdf = next((ROOT / 'lectures').glob(f'week{week}_*.pdf'))
        reader = PdfReader(pdf)
        expected_pages = len(reader.pages)
        # Page counts move when definitions or figures are added; guard against a broken
        # build (near-empty or runaway PDF) rather than pinning an exact number.
        minimum = 12 if week == 12 else 16
        assert minimum <= expected_pages <= minimum + 6, (pdf, expected_pages)
        if week == 11:
            lecture_text = ' '.join(' '.join(p.extract_text() for p in reader.pages).split())
            assert 'Hydrofoil cavitation: the machine-vision extension' in lecture_text
            assert 'native_alpha20_v6' in lecture_text
            assert 'Alpha and pressure methods on the same field' in lecture_text
            assert 'Interpreting pressure-based detection' in lecture_text
        assert all(len(p.extract_text()) > 500 for p in reader.pages)
        if render:
            from PIL import Image, ImageOps, ImageDraw
            dest = ROOT / 'tmp' / 'week11_12_pdf_review' / f'week{week}_{expected_pages}pages'
            dest.mkdir(parents=True, exist_ok=True)
            prefix = dest / f'week{week}'
            subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(prefix)],check=True)
            sheet = Image.new('RGB',(1200,450*((expected_pages+1)//2)),'#dbe3e9')
            for i, image in enumerate(sorted(dest.glob(f'week{week}-*.png'))):
                with Image.open(image) as im:
                    thumb=ImageOps.contain(im.convert('RGB'),(585,425))
                col,row=i%2,i//2
                sheet.paste(thumb,(col*600+(600-thumb.width)//2,row*450+18))
                ImageDraw.Draw(sheet).text((col*600+12,row*450+4),f'Page {i+1}',fill='black')
            sheet.save(dest / f'week{week}_review.jpg')
        print(pdf.name, f'{expected_pages} pages')
    link_files = [ROOT/'README.md', ROOT/'lectures/README.md',
                  ROOT/'results/week11_12_teaching/README.md']
    link_files += list((ROOT/'notebooks/week11').glob('*.md'))
    link_files += list((ROOT/'notebooks/week12').glob('*.md'))
    for path in link_files:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith(('#','mailto:')):
                continue
            assert (path.parent/target.split('#')[0]).exists(), (path,target)
    print('WEEK11_12_MATERIALS_PASS')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render',action='store_true')
    verify(parser.parse_args().render)
