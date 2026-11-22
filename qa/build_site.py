"""Build the Pages artifact from the maintained course metadata and native data."""
from pathlib import Path
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qa'))
import update_course_inventory as inventory
from build_cavity_explorer import export
from build_cavity_continuation_viewer import BUNDLE as CONTINUATION_BUNDLE, export as export_continuation

class LinkAudit(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag in ('a','link','script','img'):
            url=a.get('href',a.get('src',''))
            if url:self.links.append(url)
        if tag=='img' and not a.get('alt'):raise ValueError('Image missing accessible description')

def build(out):
    out=out.resolve()
    if ROOT not in out.parents or out==ROOT/'docs':
        raise ValueError('Build must be inside this checkout and outside docs/')
    # Copy only the small, published documentation tree; never raw result archives.
    out.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'docs',out,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.DS_Store'))
    inv=inventory.inventory()
    source=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    modules=inventory.site_modules(inv)
    if (CONTINUATION_BUNDLE/'media/preview_manifest.json').is_file():
        cards=modules.split('</article>')
        matches=[i for i,card in enumerate(cards) if '<span class="eyebrow">Week 1.2</span>' in card]
        if len(matches)!=1:raise ValueError('Expected exactly one Week 1.2 card for continuation preview')
        cards[matches[0]]+='<p><a href="cavity-continuation/">Longer cavity startup movies: physical time 0–100</a></p>'
        if (ROOT/'results/cavity_settling/index.html').is_file():
            cards[matches[0]]+='<p><a href="cavity-settling/">Final cavity results: stationarity and numerical sensitivity</a> · CPU notebook and all seven outcomes.</p>'
        modules='</article>'.join(cards)
    source=inventory.replace_block(source,'course-modules',modules)
    c=inv['counts']
    source=inventory.replace_block(source,'course-summary',f'<p class="release">Current course version v{inv["course_version"]} · Weeks 1–22</p>\n<p>{c["student_notebooks"]} student notebooks, including {c["historical_student_companions"]} historical companion · {c["active_lecture_pdfs"]} active lecture PDFs · {c["lecture_pages"]} pages. Optional supplements are separate.</p>')
    source=source.replace('<!-- site-version -->',html.escape(inv['course_version']))
    def link(match):
        prefix,url,suffix=match.groups()
        if not url.startswith(('#','http:','https:','mailto:')) and urlsplit(url).path.endswith('.md'):
            url='https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/docs/'+url
        if not url.startswith(('#', 'mailto:')) and not url.endswith('.csv'):
            return prefix+url+suffix+' target="_blank" rel="noopener noreferrer"'
        return prefix+url+suffix
    source=re.sub(r'(<a\b[^>]*?href=")([^"]+)(")',link,source)
    (out/'index.html').write_text(source,encoding='utf-8')
    (out/'course_inventory.json').write_text(json.dumps(inv,indent=2)+'\n',encoding='utf-8')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (out/'build.json').write_text(json.dumps(dict(commit=commit,version=inv['course_version'],units=len(inv['modules'])))+'\n')
    export(out/'assets/cavity')
    # Export only the reviewed movies and their receipt, never native CFD arrays.
    movie_root = ROOT/'results/cavity_startup/media'
    movie_receipt = movie_root/'movies.json'
    if movie_receipt.is_file():
        import hashlib
        movie_manifest = json.loads(movie_receipt.read_text(encoding='utf-8'))
        target = out/'cavity-startup'
        target.mkdir(parents=True, exist_ok=True)
        for entry in movie_manifest['fields'].values():
            name = entry['movie']
            if Path(name).name != name:
                raise ValueError('Movie receipt must use a plain filename')
            payload = movie_root/name
            if hashlib.sha256(payload.read_bytes()).hexdigest() != entry['movie_sha256']:
                raise ValueError('Movie hash mismatch: '+name)
            shutil.copy2(payload, target/name)
        shutil.copy2(movie_receipt, target/'movies.json')
        shutil.copy2(movie_root/'index.html', target/'index.html')
    # The longer preview has its own native-time schema and route. Never
    # replace the 0..10 assets used by the original notebook and lecture.
    if (CONTINUATION_BUNDLE/'media/preview_manifest.json').is_file():
        export_continuation(out)
    # Publish reviewed result figures, without importing notebook runtimes or
    # copying native fields/checkpoints into the browser artifact.
    settling = ROOT/'results/cavity_settling'
    if (settling/'index.html').is_file():
        import hashlib
        identities=json.loads((settling/'manifest.json').read_text())['files']
        lesson=json.loads((settling/'lesson_figures/manifest.json').read_text())['files']
        admitted=[(name, item['sha256']) for name,item in identities.items() if name.startswith('figures/')]
        admitted += [(item['file'],item['sha256']) for item in lesson]
        target=out/'cavity-settling';target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(settling/'index.html',target/'index.html')
        for name,expected in admitted:
            source_file=(settling/name).resolve();destination=(target/name).resolve()
            if not source_file.is_relative_to(settling.resolve()) or not destination.is_relative_to(target.resolve()):
                raise ValueError('Cavity result figure escapes its bundle')
            if hashlib.sha256(source_file.read_bytes()).hexdigest()!=expected:
                raise ValueError('Cavity figure hash mismatch: '+name)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source_file,destination)
    audit=LinkAudit();audit.feed(source)
    if len(audit.ids)!=len(set(audit.ids)):raise ValueError('Duplicate HTML IDs')
    for link in audit.links:
        parsed=urlsplit(html.unescape(link))
        if parsed.scheme or link.startswith('//'):continue
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in audit.ids:raise ValueError('Missing anchor '+link)
        elif not (out/unquote(parsed.path)).exists():raise ValueError('Missing local site asset '+link)
    # The complete maintained unit sequence must survive the presentation layer.
    for unit in inv['modules']:
        if inventory.REPO+unit['page'] not in source:raise ValueError('Missing unit '+unit['id'])
    (out/'.nojekyll').write_text('')
    print(f'Built {len(inv["modules"])} course units, all seven Ghia cases, frozen model comparisons: {out}')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'tmp/pages-site')
    build(parser.parse_args().output)
