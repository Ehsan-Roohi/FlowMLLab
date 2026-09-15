"""Render independently authored lecture notes and computed scientific figures."""
from pathlib import Path
import html
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Image,Table,TableStyle
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'results/week14_validation'
def equation(name,lines):
    fig,ax=plt.subplots(figsize=(10,1.9))
    ax.axis('off')
    for i,line in enumerate(lines):
        ax.text(.02,.85-i*.30,line,fontsize=17,color='#166c8d',transform=ax.transAxes)
    fig.savefig(E/(name+'.png'),dpi=180,bbox_inches='tight',facecolor='#f3f7fa')
    plt.close(fig)
equation('equations',[r'$\nu_t=k/\omega,\qquad P_k=\nu_t\,(dU/dy)^2$',
 r'$0=P_k-0.09\,C_k\,k\omega+\frac{d}{dy}[(\nu+\nu_t/\sigma_k)\,dk/dy]$',
 r'$0=\frac{5}{9}\frac{\omega}{k}P_k-C_{\omega2}\omega^2+\frac{d}{dy}[(\nu+\nu_t/\sigma_\omega)\,d\omega/dy]$'])
equation('pinn_equation',[r'$r(y)=a^{\prime}(y)\,k^{\prime}(y)+(\nu+a(y))\,k^{\prime\prime}(y)+P_k(y)-\epsilon(y)$',
 r'$\mathcal{L}=\sum_i r(y_i)^2+1000\,\mathcal{L}_{BC}$',
 r'$\sigma_k(y)=\nu_t(y)/a(y)$'])
styles={
 'body':ParagraphStyle('body',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=12,textColor=colors.HexColor('#24364b')),
 'title':ParagraphStyle('title',fontName='Helvetica-Bold',fontSize=22,leading=27,spaceAfter=20,textColor=colors.HexColor('#17344e')),
 'small':ParagraphStyle('small',fontName='Helvetica',fontSize=8.3,leading=11,spaceAfter=10,textColor=colors.HexColor('#537084'))}
source=(ROOT/'lectures/source/week14_rans_pinn_nn.md').read_text(encoding='utf-8')
mapping={'EQUATIONS':'equations','PINN_EQUATION':'pinn_equation','PROFILES':'profiles',
 'PIPELINE':'pipeline','COEFFICIENTS':'coefficients','TRAINING':'training','GAP':'gap','CONVERGENCE':'convergence','INVERSE':'inverse'}
story=[]
summary=json.loads((E/'summary.json').read_text())
for i,section in enumerate(source.split('\n## ')[1:]):
    title,content=section.split('\n',1)
    if i: story.append(PageBreak())
    story.append(Paragraph('FLOWMLLAB / WEEK 14 / RANS + PINN + NEURAL CLOSURES',styles['small']))
    story.append(Paragraph(html.escape(title),styles['title']))
    for p in content.strip().split('\n\n'):
        marker=p.strip().strip('[]')
        if marker in mapping:
            path=E/(mapping[marker]+'.png')
            with PILImage.open(path) as im: w,h=im.size
            width=475; height=width*h/w
            if height>208: width*=208/height; height=208
            story.append(Image(str(path),width=width,height=height))
            story.append(Spacer(1,12))
        elif marker=='RESULTS':
            rows=[['Fresh run','Iterations','Residual','Original gate']]
            for name in ['baseline','pinn','nn10000','nn5200']:
                r=summary['runs'].get(name)
                if r: rows.append([name,str(r['iteration']+1),f"{r['residual']:.2e}", 'PASS' if r['converged'] else 'NOT MET'])
            table=Table(rows,colWidths=[145,100,115,115])
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#166c8d')),
              ('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
              ('FONTSIZE',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),10),
              ('TOPPADDING',(0,0),(-1,-1),10),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#edf3f7'),colors.white])]))
            story.extend([table,Spacer(1,14)])
        else:
            story.append(Paragraph(html.escape(p.replace('\n',' ')),styles['body']))
def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#166c8d')); c.line(60,48,535,48)
    c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#537084'))
    c.drawString(60,34,'Ehsan Roohi | FlowMLLab | Based on Davidson; DNS by Lee & Moser')
    c.drawRightString(535,34,str(doc.page))
out=ROOT/'output/pdf/week14_rans_pinn_nn.pdf'
out.parent.mkdir(parents=True,exist_ok=True)
SimpleDocTemplate(str(out),pagesize=(595,842),leftMargin=60,rightMargin=60,
 topMargin=48,bottomMargin=65,title='Week 14 - Learning turbulence closures without losing the physics',
 author='Ehsan Roohi / FlowMLLab').build(story,onFirstPage=footer,onLaterPages=footer)
shutil.copy2(out,ROOT/'lectures'/out.name)
print(out)
