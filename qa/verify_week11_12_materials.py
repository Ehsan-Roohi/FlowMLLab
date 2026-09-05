"""Check executed new notebooks, local links and eight-page PDF notes.

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
        path = next((ROOT / 'notebooks' / f'week{week}').glob('*.ipynb'))
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
        assert len(reader.pages) == 8, (pdf, len(reader.pages))
        assert all(len(p.extract_text()) > 500 for p in reader.pages)
        if render:
            from PIL import Image, ImageOps, ImageDraw
            dest = ROOT / 'tmp' / 'week11_12_pdf_review'
            dest.mkdir(parents=True, exist_ok=True)
            prefix = dest / f'week{week}'
            subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(prefix)],check=True)
            sheet = Image.new('RGB',(1200,1800),'#dbe3e9')
            for i, image in enumerate(sorted(dest.glob(f'week{week}-*.png'))):
                with Image.open(image) as im:
                    thumb=ImageOps.contain(im.convert('RGB'),(585,425))
                col,row=i%2,i//2
                sheet.paste(thumb,(col*600+(600-thumb.width)//2,row*450+18))
                ImageDraw.Draw(sheet).text((col*600+12,row*450+4),f'Page {i+1}',fill='black')
            sheet.save(dest / f'week{week}_review.jpg')
        print(pdf.name, '8 pages')
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
