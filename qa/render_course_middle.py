"""Render current lecture PDFs and notebook figures for individual visual review."""
from pathlib import Path
import base64,hashlib,json,sys
import fitz

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tmp/course_middle'
def sha(b):return hashlib.sha256(b).hexdigest()

def render_pdf(path):
    target=TMP/'render'/path.stem;target.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(path);pages=[];bounds=[];glyphs=[]
    for i,p in enumerate(doc):
        image=p.get_pixmap(dpi=125,alpha=False);out=target/f'page-{i+1:03d}.png';image.save(out)
        for word in p.get_text('words'):
            if not p.rect.contains(fitz.Rect(word[:4])):bounds.append({'page':i+1,'text':word[4]})
        if any(c in p.get_text() for c in ('\u25a0','\ufffd','\x00')):glyphs.append(i+1)
        pages.append({'page':i+1,'image_path':out.relative_to(ROOT).as_posix(),'image_sha256':sha(out.read_bytes()),
                      'pixel_sha256':sha(image.samples),'width':image.width,'height':image.height,'reviewer':None,'status':'pending'})
    result={'path':path.relative_to(ROOT).as_posix(),'pdf_sha256':sha(path.read_bytes()),'dpi':125,'pages':pages,'out_of_bounds_text':bounds,'glyph_candidates':glyphs}
    (target/'render.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    return result

if __name__=='__main__':
    for name in sys.argv[1:]:
        p=ROOT/'lectures'/f'{name}.pdf';r=render_pdf(p)
        print(name,len(r['pages']),'out-of-bounds',len(r['out_of_bounds_text']),'glyphs',r['glyph_candidates'],flush=True)
