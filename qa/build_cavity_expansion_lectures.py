"""Assemble the all-Re and D/W=5 reading sections into existing lecture PDFs.

Run after the base lecture builders. Source is cavity_expansion.json; a separate
metadata key preserves all earlier pages, including prior feedback extensions.
Re-running replaces this extension instead of accumulating duplicate pages.
"""
from pathlib import Path
import io,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'qa'))
from lecture_typography import register_reportlab, configure_matplotlib
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from pypdf import PdfReader,PdfWriter


def build(targets=None):
    fonts=register_reportlab('CavityExtension')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    configure_matplotlib()
    styles={
        'title':ParagraphStyle('title',fontName=fonts['bold'],fontSize=18,leading=21,spaceAfter=16),
        'heading':ParagraphStyle('heading',fontName=fonts['bold'],fontSize=12.5,leading=16,spaceBefore=10,spaceAfter=7),
        'body':ParagraphStyle('body',fontName=fonts['regular'],fontSize=11,leading=15,spaceAfter=9),
        'small':ParagraphStyle('small',fontName=fonts['regular'],fontSize=9,leading=11.5,spaceAfter=7),
        'table':ParagraphStyle('table',fontName=fonts['regular'],fontSize=9,leading=11),
    }
    for s in styles.values():s.textColor=colors.black
    source=ROOT/'lectures/source/cavity_expansion.json'
    content=json.loads(source.read_text(encoding='utf8'))
    manifest_path=ROOT/'qa/cavity_lecture_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name,pages in content.items():
        if targets and name not in targets:continue
        if name == 'week04_cavity_surrogates_deeponet':
            from week04_evidence_presentation import format_pages
            pages = format_pages(pages)
        path=ROOT/'lectures'/f'{name}.pdf';base=PdfReader(path)
        base_count=int(base.metadata.get('/FlowMLLabCavityBasePages',len(base.pages)))
        story=[]
        for page in pages:
            if story:story.append(PageBreak())
            story.append(Paragraph(page['title'],styles['title']))
            for block in page['blocks']:
                kind=block['type']
                if kind in ('title','heading','body','small'):story.append(Paragraph(block['text'],styles[kind]))
                elif kind=='math':
                    fig=plt.figure(figsize=(6.3,.52));fig.text(.5,.5,'$'+block['latex']+'$',
                        ha='center',va='center',fontsize=12.5,color='black')
                    payload=io.BytesIO();fig.savefig(payload,format='png',dpi=240,bbox_inches='tight',pad_inches=.07)
                    plt.close(fig);payload.seek(0)
                    w,h=ImageReader(payload).getSize();width=min(465,w*72/240)
                    story.extend([Image(payload,width=width,height=width*h/w),Spacer(1,8)])
                elif kind=='figure':
                    im=ImageReader(str(ROOT/block['path']));w,h=im.getSize()
                    width=min(465,float(block.get('width',465)));height=width*h/w
                    if height>float(block.get('max_height',540)):
                        height=float(block.get('max_height',540));width=height*w/h
                    story.extend([Image(str(ROOT/block['path']),width=width,height=height),Spacer(1,8),
                                  Paragraph(block['caption'],styles['small'])])
                elif kind=='table':
                    rows=[[Paragraph(str(c),styles['table']) for c in row] for row in block['rows']]
                    table=Table(rows,colWidths=block.get('widths'),hAlign='LEFT',repeatRows=1)
                    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEABOVE',(0,0),(-1,0),.7,colors.black),
                        ('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),
                        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
                    story.extend([table,Spacer(1,12)])
                else:raise ValueError(kind)
        def footer(c,doc):
            footnote=pages[doc.page-1].get('footnote')
            if footnote:
                note=Paragraph(footnote,styles['small'])
                _,height=note.wrap(475,180)
                if height>155:raise RuntimeError('Footnote exceeds reserved region')
                c.setStrokeColor(colors.black);c.line(60,50+height+7,220,50+height+7)
                note.drawOn(c,60,50)
            c.setFillColor(colors.black);c.setFont(fonts['regular'],9)
            c.drawString(60,809,'FlowMLLab | Cavity research and validation extension')
            c.line(60,799,535,799);c.drawCentredString(297.5,28,str(base_count+doc.page))
        stream=io.BytesIO();doc=SimpleDocTemplate(stream,pagesize=(595.28,841.89),
            leftMargin=60,rightMargin=60,topMargin=58,bottomMargin=51)
        doc.build(story,onFirstPage=footer,onLaterPages=footer)
        appendix=PdfReader(stream)
        if len(appendix.pages)!=len(pages):raise RuntimeError(f'{name}: overflow {len(appendix.pages)} != {len(pages)}')
        writer=PdfWriter();writer.append(base,pages=(0,base_count),import_outline=True);writer.append(appendix)
        metadata={str(k):str(v) for k,v in dict(base.metadata or {}).items()}
        metadata.update({'/FlowMLLabCavityBasePages':str(base_count),'/FlowMLLabCavitySourceSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),
                         '/FlowMLLabCavityExtensionPages':str(len(pages))})
        writer.add_metadata(metadata)
        output=io.BytesIO();writer.write(output);path.write_bytes(output.getvalue())
        check=PdfReader(path);assert len(check.pages)==base_count+len(pages)
        manifest[name]={'base_pages_preserved':base_count,'new_pages':len(pages),
                        'total_pages':len(check.pages),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name,manifest[name],flush=True)
    (ROOT/'qa/cavity_lecture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':build()
