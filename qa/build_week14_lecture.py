"""Week 14 PDF in the exact Week 13 research-notes layout; evidence is read-only."""
from pathlib import Path
from io import BytesIO
import html, json, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'results/week14_validation'
WIDTH=A4[0]-104
pdfmetrics.registerFont(TTFont('FlowSerif',font_manager.findfont('DejaVu Serif')))
pdfmetrics.registerFont(TTFont('FlowSerifBold',font_manager.findfont(font_manager.FontProperties(family='DejaVu Serif',weight='bold'))))
# Identical type sizes, leading, margins and palette to build_week13_materials.py.
styles={
 'body':ParagraphStyle('body',fontName='FlowSerif',fontSize=9.7,leading=12.8,alignment=4,spaceAfter=5.5),
 'title':ParagraphStyle('title',fontName='FlowSerifBold',fontSize=23,leading=27,textColor=colors.HexColor('#17384d'),spaceAfter=8,keepWithNext=True),
 'subtitle':ParagraphStyle('subtitle',fontName='FlowSerif',fontSize=16,leading=20,spaceAfter=11,keepWithNext=True),
 'heading':ParagraphStyle('heading',fontName='FlowSerifBold',fontSize=12.5,leading=15.5,textColor=colors.HexColor('#17384d'),spaceBefore=9,spaceAfter=5,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='FlowSerif',fontSize=8.8,leading=11.1,spaceAfter=8),
 'cell':ParagraphStyle('cell',fontName='FlowSerif',fontSize=8,leading=9.8),
 'reference':ParagraphStyle('reference',fontName='FlowSerif',fontSize=7.7,leading=9.3,spaceAfter=2.5)}
def para(text,style='body'):
    return Paragraph(html.escape(text.replace('\n',' ')),styles[style])

EQUATIONS={
 'EQUATIONS':[r'$\nu_t=k/\omega,\qquad P_k=\nu_t\,(dU/dy)^2$',
  r'$0=P_k-0.09C_k k\omega+\frac{d}{dy}[(\nu+\nu_t/\sigma_k)dk/dy]$',
  r'$0=\frac{5}{9}\frac{\omega}{k}P_k-C_{\omega2}\omega^2+\frac{d}{dy}[(\nu+\nu_t/\sigma_\omega)d\omega/dy]$'],
 'PINN_EQUATION':[r'$r(y)=a^{\prime}(y)k^{\prime}(y)+(\nu+a(y))k^{\prime\prime}(y)+P_k(y)-\epsilon(y)$',
  r'$\mathcal{L}=\sum_i r(y_i)^2+1000\mathcal{L}_{BC}$',r'$\sigma_k(y)=\nu_t(y)/a(y)$']}
def equation(key,number):
    lines=EQUATIONS[key]
    fig=plt.figure(figsize=(7,.47*len(lines)+.18))
    for i,line in enumerate(lines):
        fig.text(.5,1-(i+.55)/len(lines),line,ha='center',va='center',fontsize=13.2)
    stream=BytesIO()
    fig.savefig(stream,format='png',dpi=300,bbox_inches='tight',pad_inches=.08,facecolor='white')
    plt.close(fig); stream.seek(0)
    picture=Image(stream)
    scale=min((WIDTH-40)/picture.imageWidth,102/picture.imageHeight)
    picture.drawWidth*=scale; picture.drawHeight*=scale
    table=Table([[picture,para(f'({number})','cell')]],colWidths=[WIDTH-28,28])
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    return table

FIGURES={
 'PROFILES':('profiles_legend_below','Classical and table-based PINN profiles against Lee-Moser DNS. Markers are fresh restarted solves, not a verified reproduction of the final PINN-NN curves in Figure 8 of the paper.'),
 'PIPELINE':('pipeline','Four stages of the attributed research workflow, each requiring a different kind of evidence.'),
 'COEFFICIENTS':('coefficients','Released spatial correction coefficients and unmodified model constants.'),
 'TRAINING':('training','Original c_k regression protocol with an explicit initialization seed. Random-point accuracy is not whole-case transfer.'),
 'GAP':('gap','PCHIP and the separately labeled tanh-MLP control over the historically inspected held-out interval.'),
 'CONVERGENCE':('convergence','Source-solver residuals. NN5200 is an assembled case; neither NN run meets the requested 1e-6 criterion.'),
 'INVERSE':('inverse','Fresh 200,000-epoch checkpoint continuation and the released diffusion profile. Minimum and final loss are distinct.')}
def figure(key,number,heading=None):
    name,caption=FIGURES[key]
    picture=Image(str(E/(name+'.png')))
    scale=min(WIDTH/picture.imageWidth,245/picture.imageHeight)
    picture.drawWidth*=scale; picture.drawHeight*=scale
    elements=[] if heading is None else [heading]
    return KeepTogether(elements+[picture,Spacer(1,4),para(f'Figure {number}. {caption}','caption')])

def results_table():
    summary=json.loads((E/'summary.json').read_text())
    rows=[['Fresh run','Iterations','Residual','Gate threshold','Original gate']]
    for name in ['baseline','pinn','nn10000','nn5200']:
        r=summary['runs'][name]
        rows.append([name,str(r['iteration']+1),f"{r['residual']:.2e}",f"{r['threshold']:.0e}",'PASS' if r['converged'] else 'NOT MET'])
    table=Table([[para(v,'cell') for v in row] for row in rows],colWidths=[WIDTH/5]*5,repeatRows=1)
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),
      ('LINEABOVE',(0,0),(-1,0),.7,colors.HexColor('#29475b')),('LINEBELOW',(0,0),(-1,0),.55,colors.HexColor('#29475b')),
      ('LINEBELOW',(0,-1),(-1,-1),.7,colors.HexColor('#29475b')),('VALIGN',(0,0),(-1,-1),'TOP'),
      ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return KeepTogether([para('Table 1. Actual iterations, achieved residual and the original stopping gate of each run; the baseline gate (1e-14) is far stricter than the others (1e-6), so its NOT MET at 3e-13 is a tolerance statement, not a sign of a worse solve.','caption'),table,Spacer(1,8)])

source=(ROOT/'lectures/source/week14_rans_pinn_nn.md').read_text(encoding='utf-8')
story=[para('Learning Turbulence Closures without Losing the Physics','title'),
       para('RANS, inverse PINN, neural regression, and reproducible evidence','subtitle'),
       para('FlowMLLab | Week 14 | Ehsan Roohi | Graduate research lecture')]
equation_number=figure_number=0
for section in source.split('\n## ')[1:]:
    title,content=section.split('\n',1)
    blocks=content.strip().split('\n\n')
    figure_first=blocks[0].strip().strip('[]') in FIGURES
    heading=para(title,'heading')
    if not figure_first: story.append(heading)
    for index,block in enumerate(blocks):
        marker=block.strip().strip('[]')
        if marker in EQUATIONS:
            equation_number+=1; story.append(equation(marker,equation_number))
        elif marker in FIGURES:
            figure_number+=1; story.append(figure(marker,figure_number,heading if index==0 and figure_first else None))
        elif marker=='RESULTS': story.append(results_table())
        elif block.strip()!='FlowMLLab / Ehsan Roohi':
            story.append(para(block,'reference' if block.startswith(('L. Davidson,','M. Lee and')) else 'body'))
def footer(canvas,document):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#b7c3cd')); canvas.setLineWidth(.4)
    canvas.line(52,43,A4[0]-52,43); canvas.setFont('FlowSerif',8.5)
    canvas.drawString(52,29,'FlowMLLab | Week 14 | RANS, PINN and neural closures')
    canvas.drawRightString(A4[0]-52,29,str(document.page)); canvas.restoreState()
def invariant_canvas(*args,**kwargs):
    kwargs['invariant']=1
    return Canvas(*args,**kwargs)
out=ROOT/'output/pdf/week14_rans_pinn_nn.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=45,bottomMargin=58,
    title='Learning Turbulence Closures without Losing the Physics',author='Ehsan Roohi').build(
        story,onFirstPage=footer,onLaterPages=footer,canvasmaker=invariant_canvas)
shutil.copy2(out,ROOT/'lectures'/out.name)
print(out)
