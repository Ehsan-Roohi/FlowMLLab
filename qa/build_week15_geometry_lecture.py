"""Build the Week 15 geometry-generalization lecture in the established research-notes layout."""
from pathlib import Path
import html
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'results/step_geometry_generalization/generated'
WIDTH=A4[0]-104
pdfmetrics.registerFont(TTFont('FlowSerif',font_manager.findfont('DejaVu Serif')))
pdfmetrics.registerFont(TTFont('FlowSerifBold',font_manager.findfont(font_manager.FontProperties(family='DejaVu Serif',weight='bold'))))
styles={
 'body':ParagraphStyle('body',fontName='FlowSerif',fontSize=9.7,leading=12.8,alignment=4,spaceAfter=5.5),
 'title':ParagraphStyle('title',fontName='FlowSerifBold',fontSize=23,leading=27,textColor=colors.HexColor('#17384d'),spaceAfter=8,keepWithNext=True),
 'subtitle':ParagraphStyle('subtitle',fontName='FlowSerif',fontSize=16,leading=20,spaceAfter=11,keepWithNext=True),
 'heading':ParagraphStyle('heading',fontName='FlowSerifBold',fontSize=12.5,leading=15.5,textColor=colors.HexColor('#17384d'),spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='FlowSerif',fontSize=8.8,leading=11.1,spaceAfter=8),
 'reference':ParagraphStyle('reference',fontName='FlowSerif',fontSize=7.7,leading=9.3,spaceAfter=2.5)}
def para(text,style='body'):
    return Paragraph(html.escape(text.replace('\n',' ')),styles[style])
FIGURES={
 'SPLIT':('__SPLIT__','Whole-geometry case map for the retained 130-field dataset; no bar chart.'),
 'VANILLA':('__VANILLA__','Existing V5 project comparison: each point is one seed; large markers are means. The ordinary DeepONet fails the frozen checkpoint gate.'),
 'POINTS':('generalization_case_points.png','Every held-out case and the protocol mean. Large markers are means; small markers are individual cases.'),
 'G009':('g009_Re100_medium_fields.png','Unseen geometry g009 at Re=100: CFD, ordinary DeepONet, Geo-DeepONet, and FNO in rows. Row labels report seed-17 velocity and centered-pressure errors; ordinary DeepONet is markedly worse.'),
 'G048':('g048_Re50_medium_fields.png','Unseen geometry g048 at Re=50: the fixed-domain DeepONet is shown beside geometry-aware and spectral operators, with seed-17 errors printed in each row.'),
 'FAMILY':('casebook_g049_Re50_family.png','Whole-family holdout example g049 at Re=50 from the retained three-seed casebook.')}
def figure(key,number,heading=None):
    rel,caption=FIGURES[key]
    if rel=='__SPLIT__': path=ROOT/'results/step_operator_audit/generated/dataset_split_matrix.png'
    elif rel=='__VANILLA__': path=ROOT/'results/step_operator_audit/generated/vanilla_deeponet_v5_points.png'
    else: path=E/rel
    path=path.resolve()
    pic=Image(str(path)); scale=min(WIDTH/pic.imageWidth,205/pic.imageHeight)
    pic.drawWidth*=scale; pic.drawHeight*=scale
    return KeepTogether(([] if heading is None else [heading])+[pic,Spacer(1,4),para(f'Figure {number}. {caption}','caption')])

source=(ROOT/'lectures/source/week15_geometry_generalization.md').read_text(encoding='utf-8')
story=[para('Neural Operators under Geometry Change','title'),
       para('DeepONet, Geo-DeepONet, FNO, U-FNO, and separated step flow','subtitle'),
       para('FlowMLLab | Week 15 | Ehsan Roohi | Graduate research lecture')]
fnum=0
for section in source.split('\n## ')[1:]:
    title,content=section.split('\n',1); blocks=content.strip().split('\n\n')
    first=blocks[0].strip().strip('[]') in FIGURES; heading=para(title,'heading')
    if not first: story.append(heading)
    for i,block in enumerate(blocks):
        marker=block.strip().strip('[]')
        if marker in FIGURES:
            fnum+=1; story.append(figure(marker,fnum,heading if first and i==0 else None))
        elif block.strip()!='FlowMLLab / Ehsan Roohi':
            ref=block.startswith(('Lu et al.','Li et al.'))
            story.append(para(block,'reference' if ref else 'body'))
def footer(canvas,document):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#b7c3cd')); canvas.setLineWidth(.4)
    canvas.line(52,43,A4[0]-52,43); canvas.setFont('FlowSerif',8.5)
    canvas.drawString(52,29,'FlowMLLab | Week 15 | Geometry-aware neural operators')
    canvas.drawRightString(A4[0]-52,29,str(document.page)); canvas.restoreState()
def invariant_canvas(*args,**kwargs):
    kwargs['invariant']=1; return Canvas(*args,**kwargs)
out=ROOT/'output/pdf/week15_geometry_generalization.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=45,bottomMargin=58,
    title='Neural Operators under Geometry Change',author='Ehsan Roohi').build(
        story,onFirstPage=footer,onLaterPages=footer,canvasmaker=invariant_canvas)
(ROOT/'lectures').mkdir(exist_ok=True)
(ROOT/'lectures/week15_geometry_generalization.pdf').write_bytes(out.read_bytes())
print(out)
