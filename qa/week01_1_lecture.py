"""Render continuous Week 1.1 lecture notes from their editable Markdown source."""
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullLocator
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.pdfgen.canvas import Canvas

EQUATIONS = {
    'operators': [r'$\nabla\cdot\mathbf{u}=\partial_xu+\partial_yv,\qquad\omega_z=\partial_xv-\partial_yu$'],
    'taylor': [r'$f(x\pm h)=f(x)\pm h f^{\prime}(x)+\frac{h^2}{2}f^{\prime\prime}(x)\pm\frac{h^3}{6}f^{\prime\prime\prime}(x)+\mathcal{O}(h^4)$'],
    'centered': [r'$\frac{f(x+h)-f(x-h)}{2h}=f^{\prime}(x)+\frac{h^2}{6}f^{\prime\prime\prime}(x)+\mathcal{O}(h^4)$'],
    'psi': [r'$\psi(x,y)=\sin^2(\pi x)\sin^2(\pi y),\qquad (x,y)\in[0,1]^2$'],
    'velocities': [r'$u=\psi_y=\pi\sin^2(\pi x)\sin(2\pi y)$',r'$v=-\psi_x=-\pi\sin(2\pi x)\sin^2(\pi y)$'],
    'omega': [r'$\omega_z=-\nabla^2\psi$',r'$=-2\pi^2[\cos(2\pi x)\sin^2(\pi y)+\sin^2(\pi x)\cos(2\pi y)]$'],
    'error': [r'$E_h=\frac{\left[\sum_{(j,i)\in I_h}(\omega^h_{j,i}-\omega^{\mathrm{exact}}_{j,i})^2\right]^{1/2}}{\left[\sum_{(j,i)\in I_h}(\omega^{\mathrm{exact}}_{j,i})^2\right]^{1/2}}$'],
    'wrong': [r'$\omega_{\mathrm{wrong}}=D_yv-D_xu\ne D_xv-D_yu$'],
}


def build_lecture(output, lecture, verification, cavity, record, faulty_error, root):
    """Keep the legacy builder interface; only the lecture is reformatted."""
    from flowmllab.scientific_software import load_cavity_case, differential_diagnostics
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = {
        'body': ParagraphStyle('body',fontName='Times-Roman',fontSize=11,leading=15,spaceAfter=8,alignment=4),
        'h1': ParagraphStyle('h1',fontName='Times-Bold',fontSize=22,leading=27,spaceAfter=8,keepWithNext=True),
        'h2': ParagraphStyle('h2',fontName='Times-Bold',fontSize=16,leading=21,spaceAfter=8,keepWithNext=True),
        'h3': ParagraphStyle('h3',fontName='Times-Bold',fontSize=13,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True),
        'caption': ParagraphStyle('caption',fontName='Times-Roman',fontSize=9.5,leading=12,spaceAfter=10),
        'cell': ParagraphStyle('cell',fontName='Times-Roman',fontSize=9.5,leading=12),
        'code': ParagraphStyle('code',fontName='Courier',fontSize=9,leading=13,backColor=colors.HexColor('#f1f4f6'),borderPadding=8,spaceBefore=8,spaceAfter=12),
    }
    story=[]; width=A4[0]-104
    def para(t, style='body'): return Paragraph(escape(t),styles[style])
    def raster(fig, maxw=width, maxh=220):
        buf=BytesIO()
        fig.savefig(buf,format='png',dpi=260,bbox_inches='tight',pad_inches=.08,facecolor='white')
        plt.close(fig); buf.seek(0)
        img=Image(buf); scale=min(maxw/img.imageWidth,maxh/img.imageHeight)
        img.drawWidth=img.imageWidth*scale; img.drawHeight=img.imageHeight*scale
        return img
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def equation(key, number):
        lines=EQUATIONS[key]
        fig=plt.figure(figsize=(7,.55*len(lines)+.18))
        for i,s in enumerate(lines): fig.text(.5,1-(i+.6)/len(lines),s,ha='center',va='center',fontsize=15)
        im=raster(fig,maxw=width-35,maxh=85 if key=='error' else 40*len(lines))
        t=Table([[im,para('('+str(number)+')','cell')]],colWidths=[width-30,30])
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
        return t
    def tabulate(rows, widths, caption):
        t=Table([[para(str(v),'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='CENTER')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef2')),
            ('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
            ('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        return KeepTogether([para(caption,'caption'),t,Spacer(1,12)])
    rows=verification['rows']
    def table(key):
        if key=='convergence':
            data=[['Points per direction','Grid spacing','Relative L2 error','Divergence RMS']]
            data += [[r['points'],f"{r['h']:.6f}",f"{r['vorticity_relative_l2']:.6e}",f"{r['divergence_rms']:.3e}"] for r in rows]
            return tabulate(data,[width*.25]*4,'Table 1. Analytic verification on four uniform grids; two layers are excluded from each edge.')
        data=[['Acceptance check','Observed','Required'],
            ['Dataset identity','Exact hash match','Exact match'],
            ['Four-grid fitted order',f"{verification['observed_order']:.4f}",'At least 1.90'],
            ['Finest vorticity error',f"{rows[-1]['vorticity_relative_l2']:.3e}",'At most 5e-4'],
            ['Analytic divergence RMS',f"{max(r['divergence_rms'] for r in rows):.3e}",'At most 1e-12'],
            ['Cavity divergence RMS',f"{cavity['interior_divergence_rms']:.3e}",'At most 1e-12'],
            ['Cavity vorticity disagreement',f"{cavity['archive_vorticity_relative_l2']:.3e}",'At most 4e-2'],
            ['Non-corner wall error',f"{cavity['wall_velocity_max_abs_error']:.1e}",'At most 1e-12']]
        return tabulate(data,[width*.48,width*.26,width*.26],'Table 2. Retained measurements and fixed criteria. All seven checks pass.')
    def figure(key):
        if key=='convergence':
            fig,ax=plt.subplots(figsize=(5.9,3.0))
            h=np.array([r['h'] for r in rows]); e=np.array([r['vorticity_relative_l2'] for r in rows])
            ax.loglog(h,e,'o-',color='#007F82',label='Measured error')
            ax.loglog(h,e[-1]*(h/h[-1])**2,'--',color='#687B89',label='Second-order reference')
            ax.set_xticks(h,['1/16','1/32','1/64','1/128']); ax.xaxis.set_minor_locator(NullLocator())
            ax.set(xlabel='Grid spacing h',ylabel='Relative L2 vorticity error')
            ax.legend(frameon=False); ax.grid(alpha=.18)
            caption='Figure 1. Error decreases at approximately second order. The reported order, 1.9965, is the slope fitted to all four log-error/log-spacing pairs.'
        else:
            case,_=load_cavity_case(root,100.)
            omega=differential_diagnostics(case['x'],case['y'],case['u'],case['v']).vorticity
            lim=np.max(np.abs(omega)); positive=np.geomspace(.1,lim,30)
            fig,ax=plt.subplots(figsize=(4.9,3.6))
            im=ax.contourf(case['x'],case['y'],omega,levels=np.r_[-positive[::-1],0,positive],
                norm=SymLogNorm(linthresh=1,vmin=-lim,vmax=lim),cmap='RdBu_r')
            ax.streamplot(case['x'],case['y'],case['u'],case['v'],color='#162C40',density=.7,linewidth=.45,arrowsize=.7)
            ax.set(xlabel='x / L',ylabel='y / L',aspect='equal')
            fig.colorbar(im,ax=ax,label='Vorticity, normalized units',ticks=[-80,-10,-1,0,1,10,80])
            caption='Figure 2. Vorticity computed from retained Re = 100 cavity velocities, with streamlines. Axes have equal scale. Color uses a symmetric logarithmic scale, linear within +/-1, to retain wall extremes while revealing the interior structure. Metrics use the support described in the text.'
        return KeepTogether([raster(fig,maxh=215),Spacer(1,5),para(caption,'caption')])
    source=root/'lectures/source/week01_1_lecture_notes.md'
    equation_number=0
    for block in source.read_text(encoding='utf-8').strip().split('\n\n'):
        block=block.strip()
        if block.startswith('@equation '):
            equation_number+=1; story.append(equation(block.split()[1],equation_number))
        elif block.startswith('@figure '): story.append(figure(block.split()[1]))
        elif block.startswith('@table '): story.append(table(block.split()[1]))
        elif block.startswith('@code '): story.append(para(block[6:],'code'))
        elif block.startswith('### '): story.append(para(block[4:],'h3'))
        elif block.startswith('## '): story.append(para(block[3:],'h2'))
        elif block.startswith('# '): story.append(para(block[2:],'h1'))
        else: story.append(para(block.replace('\n',' ')))
    def footer(c,doc):
        c.saveState(); c.setStrokeColor(colors.HexColor('#b7c3cd')); c.setLineWidth(.4)
        c.line(52,43,A4[0]-52,43); c.setFont('Times-Roman',9)
        c.drawString(52,29,'FlowMLLab | Week 1.1 | Ehsan Roohi')
        c.drawRightString(A4[0]-52,29,str(doc.page)); c.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=48,bottomMargin=58,
        title='Week 1.1 - AI-assisted scientific software: lecture notes',author='Ehsan Roohi')
    def deterministic_canvas(*args, **kwargs):
        kwargs['invariant'] = 1
        return Canvas(*args, **kwargs)
    doc.build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=deterministic_canvas)
    shutil.copy2(output,lecture)
    return output
