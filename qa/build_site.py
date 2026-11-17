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
    source=inventory.replace_block(source,'course-modules',inventory.site_modules(inv))
    c=inv['counts']
    source=inventory.replace_block(source,'course-summary',f'<p class="release">Current course version v{inv["course_version"]} Â· Weeks 1â€“22</p>\n<p>{c["student_notebooks"]} student notebooks, including {c["historical_student_companions"]} historical companion Â· {c["active_lecture_pdfs"]} active lecture PDFs Â· {c["lecture_pages"]} pages. Optional supplements are separate.</p>')
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
