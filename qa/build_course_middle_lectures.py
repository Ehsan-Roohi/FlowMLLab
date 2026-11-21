"""Idempotently add detailed Week 7–14 teaching guides to retained protocols.

Call refresh_lecture(path) AFTER the original lecture and reading-section
builders. Only the target PDFs below are changed. A second call extracts its
retained protocol pages using PDF metadata, never stacking a guide twice.
"""
from pathlib import Path
from io import BytesIO
import hashlib,html,json,os,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lectures.source.course_middle_guides import GUIDES
from qa.lecture_typography import register_reportlab,configure_matplotlib

def sha(data):return hashlib.sha256(data).hexdigest()

def build_guide(name, retained_pages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image
    from pypdf import PdfReader
    configure_matplotlib();register_reportlab('MiddleTNR')
    module=GUIDES[name];width=A4[0]-92
    styles={
        'title':ParagraphStyle('mt',fontName='MiddleTNRB',fontSize=18,leading=21,spaceAfter=9,textColor=colors.black),
        'head':ParagraphStyle('mh',fontName='MiddleTNRB',fontSize=13,leading=16,spaceAfter=10,textColor=colors.black),
        'body':ParagraphStyle('mb',fontName='MiddleTNR',fontSize=11,leading=14.5,spaceAfter=8,textColor=colors.black),
        'small':ParagraphStyle('ms',fontName='MiddleTNR',fontSize=8.6,leading=11,spaceAfter=8,textColor=colors.black),
    }
    out=ROOT/'tmp/course_middle/guides'/f'{name}.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    story=[]
    for i,p in enumerate(module['pages']):
        if i:story.append(PageBreak())
        if i==0:
            story.append(Paragraph(html.escape(module['title']),styles['title']))
            label=module['notebook'].split('/')[-1]
            url='https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/'+module['notebook']
            story.append(Paragraph(f'Notebook: <a color="black" href="{url}">{html.escape(label)}</a>',styles['small']))
            route=f'This guide: pages 1–{len(module["pages"])}.'
            if retained_pages:route+=f' Detailed protocol and retained results: PDF pages {len(module["pages"])+1}–{len(module["pages"])+retained_pages} (original pagination retained).'
            story.append(Paragraph(route,styles['small']))
        story.append(Paragraph(html.escape(p['title']),styles['head']))
        for paragraph in p['paragraphs']:
            formatted=html.escape(paragraph)
            for old,new in (('Re_tau','Re<sub size="8">τ</sub>'),('u_tau','u<sub size="8">τ</sub>'),('nu_t','ν<sub size="8">t</sub>')):
                formatted=formatted.replace(old,new)
            story.append(Paragraph(formatted,styles['body']))
        for equation in p['equations']:
            fig=plt.figure(figsize=(7.3,.58));fig.text(.02,.5,equation,fontsize=13.2,va='center',color='black')
            buf=BytesIO();fig.savefig(buf,format='png',dpi=200,bbox_inches='tight',pad_inches=.05);plt.close(fig);buf.seek(0)
            img=Image(buf);factor=min(width/img.imageWidth,72/200)
            img.drawWidth=img.imageWidth*factor;img.drawHeight=img.imageHeight*factor
            story.extend([img,Spacer(1,5)])
        if p.get('figure'):
            target=ROOT/p['figure']
            if not target.is_file():raise FileNotFoundError(target)
            img=Image(str(target));factor=min(width/img.imageWidth,330/img.imageHeight)
            img.drawWidth=img.imageWidth*factor;img.drawHeight=img.imageHeight*factor
            story.extend([Spacer(1,5),img,Spacer(1,5),Paragraph(html.escape(p['caption']),styles['small'])])
    def footer(canvas,doc):
        canvas.setFont('MiddleTNR',8);canvas.setFillColor(colors.black)
        canvas.drawString(46,24,'FlowMLLab | Teaching guide | '+module['title'].split(' — ')[0])
        canvas.drawRightString(A4[0]-46,24,str(doc.page))
    SimpleDocTemplate(str(out),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=38,bottomMargin=45,
        title=module['title'],author='Ehsan Roohi / FlowMLLab').build(story,onFirstPage=footer,onLaterPages=footer)
    n=len(PdfReader(out).pages)
    if n!=len(module['pages']):raise RuntimeError(f'{name}: planned {len(module["pages"])} guide pages, produced {n}; repair layout rather than accept a spill.')
    return out

def refresh_lecture(path):
    from pypdf import PdfReader,PdfWriter
    path=Path(path).resolve()
    if path.stem not in GUIDES:return False
    original=path.read_bytes();reader=PdfReader(BytesIO(original));meta=reader.metadata or {}
    start=int(meta.get('/CourseMiddleGuidePages',0))
    if GUIDES[path.stem].get('replace_old'):
        retained=[]
    elif start:
        count=int(meta['/CourseMiddleRetainedPages']);retained=list(reader.pages[start:start+count])
    else:
        count=min(len(reader.pages),int(meta.get('/FlowMLLabBasePages',len(reader.pages))))
        retained=list(reader.pages[:count])
    guide=build_guide(path.stem,len(retained));guide_reader=PdfReader(guide)
    writer=PdfWriter()
    for p in guide_reader.pages:writer.add_page(p)
    for p in retained:writer.add_page(p)
    for i,p in enumerate(GUIDES[path.stem]['pages']):writer.add_outline_item(p['title'],i)
    if retained:writer.add_outline_item('Detailed protocol and retained results',len(guide_reader.pages))
    source=ROOT/'lectures/source/course_middle_guides.py'
    writer.add_metadata({'/Title':GUIDES[path.stem]['title'],'/Author':'Ehsan Roohi / FlowMLLab',
        '/CourseMiddleGuidePages':str(len(guide_reader.pages)),'/CourseMiddleRetainedPages':str(len(retained)),
        '/CourseMiddleGuideSourceSHA256':sha(source.read_bytes()),'/FlowMLLabBasePages':str(len(writer.pages))})
    buf=BytesIO();writer.write(buf);path.write_bytes(buf.getvalue())
    return dict(path=path.relative_to(ROOT).as_posix(),guide_pages=len(guide_reader.pages),retained_pages=len(retained),pages=len(writer.pages),sha256=sha(path.read_bytes()))

if __name__=='__main__':
    selected=sys.argv[1:] or list(GUIDES)
    result=[refresh_lecture(ROOT/'lectures'/f'{n}.pdf') for n in selected]
    (ROOT/'tmp/course_middle/lecture_build.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))
