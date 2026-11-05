from pathlib import Path
import os,re,html,json,hashlib
ROOT=Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"]=str(ROOT/".matplotlib")
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle,Preformatted
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from matplotlib import font_manager
for name,fp in [("Guide",font_manager.findfont("DejaVu Sans")),
                ("GuideBold",font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",weight="bold"))),
                ("GuideMono",font_manager.findfont("DejaVu Sans Mono"))]:
    pdfmetrics.registerFont(TTFont(name,fp))
pdfmetrics.registerFontFamily("Guide",normal="Guide",bold="GuideBold",italic="Guide",boldItalic="GuideBold")
styles=getSampleStyleSheet()
for name,size,leading in [("Body",10.3,13.8),("H1",19,23),("H2",14,18),("H3",11,15),("Caption",8.4,11),("Table",8.2,10.8),("Code",7.5,10)]:
    styles.add(ParagraphStyle(name="G"+name,fontName="GuideBold" if name.startswith("H") else "GuideMono" if name=="Code" else "Guide",fontSize=size,leading=leading,keepWithNext=name.startswith("H"),spaceBefore=4 if name.startswith("H") else 0,spaceAfter=6 if name!="Table" else 0,textColor=colors.HexColor("#125a80" if name.startswith("H") else "#23313e"),backColor=colors.HexColor("#f2f5f7") if name=="Code" else None,borderPadding=6 if name=="Code" else 0))
width=A4[0]-92
tick=chr(96)
def clean(s):
    return s.replace("\u2011","-").replace("\u2013","-").replace("\u2014"," - ")
def inline(s):
    s=html.escape(clean(s))
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',lambda m:f'<link href="{m[2]}" color="#125a80">{m[1]}</link>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(tick+r'([^'+tick+r']+)'+tick,r'<font name="GuideMono">\1</font>',s)
    s=re.sub(r'\*(.+?)\*',r'<i>\1</i>',s)
    return s
lines=(ROOT/"week01_02_tensorflow_guide.md").read_text(encoding="utf-8-sig").splitlines()
story=[]
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:
        i+=1;continue
    if line=="<!-- PAGEBREAK -->":
        story.append(PageBreak());i+=1;continue
    if line.startswith("<!--"):
        i+=1;continue
    if line.startswith((tick*3,"~~~")):
        block=[];i+=1
        while i<len(lines) and not lines[i].startswith((tick*3,"~~~")):
            s=lines[i].expandtabs(4)
            if len(s)>101:
                block.extend([s[:98]+" \\", "    "+s[98:]])
            else:
                block.append(s)
            i+=1
        story.append(Preformatted(clean("\n".join(block)),styles["GCode"]));i+=1;continue
    image=re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)",line)
    if image:
        path=ROOT/image[2]
        assert path.exists(),path
        im=Image(str(path));im._restrictSize(width,180)
        story.append(im)
        if image[1]:
            story.append(Paragraph(inline(image[1]),styles["GCaption"]))
        i+=1;continue
    if line.startswith("|"):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith("|"):
            row=[s.strip() for s in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?",s) for s in row):
                rows.append(row)
            i+=1
        n=max(map(len,rows))
        weights=[max(12,min(48,max(len(r[j]) if j<len(r) else 1 for r in rows))) for j in range(n)]
        widths=[width*w/sum(weights) for w in weights]
        data=[[Paragraph(inline(c),styles["GTable"]) for c in r+[""]*(n-len(r))] for r in rows]
        table=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5eff4")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#cad7df")),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
        story.extend([table,Spacer(1,9)]);continue
    if line.startswith("#"):
        m=re.match(r"(#{1,3})\s+(.+)",line)
        if m:
            story.append(Paragraph(inline(m[2]),styles["G"+"H"+str(len(m[1]))]));i+=1;continue
    block=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#",tick*3,"~~~","|","![","<!--")):
        if re.match(r"^\s*(?:[-*]|\d+\.)\s+",lines[i]):
            break
        block.append(lines[i].strip());i+=1
    s=" ".join(block)
    if re.match(r"^(?:[-*]|\d+\.)\s+",s):
        m=re.match(r"^([-*]|\d+\.)\s+(.+)",s)
        story.append(Paragraph(inline(m[2]),styles["GBody"],bulletText="•" if m[1] in "-*" else m[1]))
    else:
        story.append(Paragraph(inline(s),styles["GBody"]))
def page(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#8aacbf"));canvas.setLineWidth(.5)
    canvas.line(46,A4[1]-37,A4[0]-46,A4[1]-37)
    canvas.setFont("Guide",8);canvas.setFillColor(colors.HexColor("#526b7c"))
    canvas.drawString(46,A4[1]-29,"FlowMLLab  |  TensorFlow and Keras for fluid engineers")
    canvas.drawString(46,27,"Self-study lab 1, notebook 02  |  Ehsan Roohi")
    canvas.drawRightString(A4[0]-46,27,str(doc.page));canvas.restoreState()
output=ROOT.parents[1]/"week01_02_tensorflow_keras.pdf"
doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=53,bottomMargin=47,title="FlowMLLab - TensorFlow and Keras for fluid engineers",author="Ehsan Roohi")
doc.build(story,onFirstPage=page,onLaterPages=page)
import fitz
pdf=fitz.open(output)
qa=ROOT/"pdf_qa";qa.mkdir(exist_ok=True)
audit=[]
for n,p in enumerate(pdf):
    p.get_pixmap(matrix=fitz.Matrix(1.45,1.45),alpha=False).save(qa/f"page_{n+1:02d}.png")
    text=p.get_text()
    bounds=[(b[0],b[1],b[2],b[3]) for b in p.get_text("blocks")]
    outside=[b for b in bounds if b[0]<23 or b[1]<10 or b[2]>A4[0]-20 or b[3]>A4[1]-12]
    audit.append({"page":n+1,"words":len(text.split()),"outside_safe_page_bounds":outside})
assert all(x["words"]>20 and not x["outside_safe_page_bounds"] for x in audit),audit
(ROOT/"pdf_validation.json").write_text(json.dumps(dict(pages=len(pdf),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),pages_checks=audit),indent=2))
print("PDF pages",len(pdf),"size",output.stat().st_size)
