"""Read-only consistency check for all four tasks in every published edition."""
from pathlib import Path
import json,re,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat
from pypdf import PdfReader

def normalized(text):
    return re.sub(r'\s+',' ',text).strip()

def verify():
    guide=(ROOT/'docs/TRANSFORMER_COURSE.md').read_text(encoding='utf-8')
    checked=0
    for week in range(17,23):
        spec=json.loads((ROOT/f'course/lessons/week{week}.json').read_text())
        source=(ROOT/f'lectures/source/week{week}_transformer_course.md').read_text()
        assert 'Independent review corrections and integrated investigation' not in source
        pdf=normalized(' '.join(page.extract_text() for page in PdfReader(ROOT/f'lectures/week{week}_cfd_transformer.pdf').pages))
        for index,task in enumerate(spec['tasks'],1):
            title=f'Task {index}: {task["title"]}'
            assert source.count('### '+title)==1,(week,title,'source')
            assert normalized(title) in pdf,(week,title,'PDF')
            assert task['title'] in guide,(week,title,'guide')
            assert task['prompt'] in source,(week,title,'contract')
            for edition in ('student','instructor'):
                path=ROOT/(f'notebooks/week{week}/W{week}_CFD_Transformer.ipynb' if edition=='student' else f'instructor/week{week}/W{week}_Solutions.ipynb')
                nb=nbformat.read(path,as_version=4)
                assert any(c.cell_type=='markdown' and f'## {title}\n\n{task["prompt"]}'==c.source for c in nb.cells),(week,title,edition)
            checked+=1
    print(f'PASS: {checked} task titles and contracts agree across lesson JSON, both notebooks, PDF source, PDFs and guide.')
    return checked

if __name__=='__main__':verify()
