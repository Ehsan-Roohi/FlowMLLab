"""Build original PINN/cavity reading notes; no upstream training or network access.

Run from the repository: python qa/build_week04_2_lecture.py
Requires numpy, matplotlib, reportlab. Outputs default to output/pdf.
Use --publish-copy to copy the verified generated PDF into lectures/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from io import BytesIO
from pathlib import Path
import re
import shutil
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
NAME = 'week04_2_pinn_cavity'
EQS = {
    'scales': [r'$x^*=x/L,\quad y^*=y/L,\quad (u^*,v^*)=(u,v)/U$', r'$p^*=(p-p_0)/(\rho U^2),\qquad Re=UL/\nu$'],
    'ns': [r'$r_c=u_x+v_y=0$', r'$r_u=u u_x+v u_y+p_x-Re^{-1}(u_{xx}+u_{yy})=0$', r'$r_v=u v_x+v v_y+p_y-Re^{-1}(v_{xx}+v_{yy})=0$'],
    'network': [r'$\mathbf{h}^{(0)}=(x,y),\quad\mathbf{h}^{(k+1)}=\tanh(W_k\mathbf{h}^{(k)}+\mathbf{b}_k)$', r'$(u_\theta,v_\theta,p_\theta)=W_o\mathbf{h}^{(K)}+\mathbf{b}_o$'],
    'softloss': [r'$\mathcal{L}=\lambda_m\langle r_u^2+r_v^2\rangle_\Omega+\lambda_c\langle r_c^2\rangle_\Omega$', r'$\qquad+\lambda_b\langle\|\mathbf{u}_\theta-\mathbf{u}_b\|^2\rangle_{\partial\Omega}+\lambda_g|p_\theta(\mathbf{x}_0)|^2$'],
    'streamfunction': [r'$u=\psi_y,\quad v=-\psi_x,\quad u_x+v_y=\psi_{yx}-\psi_{xy}=0$', r'$\omega_z=v_x-u_y=-\nabla^2\psi$'],
    'lifting': [r'$B=16x(1-x)y(1-y),\qquad \psi_\theta=\psi_{\mathrm{lid}}+B^2q_\theta$', r'$u=\psi_{\mathrm{lid},y}+2BB_yq_\theta+B^2q_{\theta,y}$', r'$v=-\psi_{\mathrm{lid},x}-2BB_xq_\theta-B^2q_{\theta,x}$'],
    'lid': [r'$g(x)=(1-e^{-(1-x)^2/\delta_x^2})(1-e^{-x^2/\delta_x^2})$', r'$\psi_{\mathrm{lid}}=(y-1)y^2g(x)e^{-(1-y)^2/\delta_y^2}$', r'$\delta_x=\sqrt{10^{-3}},\qquad\delta_y=\sqrt{10^{-2}}$'],
    'mask': [r'$m=(1-e^{-d_{\mathrm{TL}}^2/a^2})(1-e^{-d_{\mathrm{TR}}^2/a^2}),\quad a=0.01$', r'$\widetilde r_u=m r_u,\quad\widetilde r_v=m r_v,\quad\mathcal{L}=\langle m^2(r_u^2+r_v^2)\rangle$'],
    'kovasznay': [r'$\lambda=\frac{Re}{2}-\sqrt{\frac{Re^2}{4}+4\pi^2}$', r'$u=1-e^{\lambda x}\cos(2\pi y),\quad v=\frac{\lambda}{2\pi}e^{\lambda x}\sin(2\pi y)$', r'$p=\frac{1}{2}(1-e^{2\lambda x})$'],
    'error': [r'$E_{\mathbf{u}}=\left[\frac{\sum_j w_j\|\mathbf{u}_{\theta,j}-\mathbf{u}_{\mathrm{ref},j}\|^2}{\sum_j w_j\|\mathbf{u}_{\mathrm{ref},j}\|^2}\right]^{1/2}$', r'$p^\circ_j=p_j-\frac{\sum_k w_kp_k}{\sum_k w_k},\qquad w_j>0$'],
}
TABLES = {
    'methods': ('Table 1. Three different uses of a numerical representation.', [
        ['Method', 'What determines the field?', 'Independent evidence'],
        ['CFD solver', 'Discrete equations, boundary conditions and convergence.', 'Mesh/time refinement and matched benchmarks.'],
        ['Supervised surrogate', 'Loss against previously generated fields.', 'Complete held-out cases and a non-neural baseline.'],
        ['Forward PINN', 'PDE residual plus imposed constraints at fixed parameters.', 'Matched reference, boundary errors and unseen residual points.'],
    ]),
    'upstream': ('Table 2. Defaults read from the pinned upstream square-cavity script [1].', [
        ['Ingredient', 'Inspected value', 'Scientific implication'],
        ['Inputs / raw outputs', '(x,y) / (q,p)', 'Fixed-Re field; velocity derived from a lifted streamfunction.'],
        ['Architecture', '3 hidden layers, 50 tanh units each', 'Two linear output channels; Xavier initialization.'],
        ['Re / dtype / seed', '5000 / float64 / 12345', 'Not an executed Re=20000 evidence record.'],
        ['PDE / test points', '262143 each', 'Independent random samples, same corner exclusion.'],
        ['SOAP_STEPS', '0', 'Optional stage disabled by default.'],
        ['SSBroyden2', '10000 outer iterations; strong-Wolfe search', 'Closure evaluations can exceed outer iterations.'],
        ['Corner treatment', 'Exclusion radius 0.02; top-mask radius 0.01', 'Logged residual is not a full-domain raw residual.'],
    ]),
    'validation': ('Table 3. Complementary evidence; none replaces the others.', [
        ['Check', 'What it diagnoses', 'What it cannot establish alone'],
        ['Velocity field / centerlines', 'Agreement with a matched resolved reference.', 'Accuracy beyond the reference or at a different Re.'],
        ['Wall velocity mismatch', 'Enforcement of the stated lid and no-slip condition.', 'Correct interior momentum balance.'],
        ['Unmasked momentum residual', 'Differential balance on a declared independent support.', 'Uniqueness, stability or accurate wall conditions.'],
        ['Divergence / boundary flux', 'Local/global kinematic consistency.', 'Correct vortex strength or pressure gradients.'],
        ['Pressure after gauge alignment', 'Pressure variation relative to the reference.', 'A physically meaningful absolute pressure offset.'],
        ['Seed / grid / sampling study', 'Sensitivity to initialization and numerical resolution.', 'Universality outside the tested problem family.'],
    ]),
}


def analytic_preflight():
    """Independent analytic checks, not a training or autograd validation claim."""
    rng = np.random.default_rng(421)
    x, y = rng.uniform(0, 1, (2, 2000))
    reynolds = 40.
    lam = -4*np.pi**2/(reynolds/2+np.sqrt(reynolds**2/4+4*np.pi**2))
    a = np.exp(lam*x); c = np.cos(2*np.pi*y); s = np.sin(2*np.pi*y)
    u = 1-a*c; v = lam*a*s/(2*np.pi)
    ux = -lam*a*c; uy = 2*np.pi*a*s
    vx = lam**2*a*s/(2*np.pi); vy = lam*a*c
    lapu = (-lam**2+4*np.pi**2)*a*c
    lapv = (lam**3/(2*np.pi)-2*np.pi*lam)*a*s
    ru = u*ux+v*uy-lam*a*a-lapu/reynolds
    rv = u*vx+v*vy-lapv/reynolds
    residual = float(np.max(np.abs(np.stack([ru, rv, ux+vy]))))
    assert residual < 1e-12, residual
    # Complex-step derivatives of the entire independent lifting expression.
    def psi(x, y, coefficient):
        b = 16*x*(1-x)*y*(1-y)
        g = (1-np.exp(-(1-x)**2/.001))*(1-np.exp(-x*x/.001))
        lift = (y-1)*y*y*g*np.exp(-(1-y)**2/.01)
        return lift+b*b*coefficient*np.sin(x+2*y)
    t = np.linspace(0, 1, 1001); h = 1e-25; errors = []
    for coefficient in [-3., 0., 2.]:
        for bx, by, lid in [(t, np.ones_like(t), True),(t,t*0,False),(t*0,t,False),(t*0+1,t,False)]:
            bu = np.imag(psi(bx, by+1j*h, coefficient))/h
            bv = -np.imag(psi(bx+1j*h, by, coefficient))/h
            target = (1-np.exp(-(1-bx)**2/.001))*(1-np.exp(-bx*bx/.001)) if lid else np.zeros_like(t)
            errors.extend([np.max(np.abs(bu-target)),np.max(np.abs(bv))])
    boundary_error = float(max(errors)); assert boundary_error < 1e-12
    return {'check_type':'analytic formulas and complex-step boundary derivatives; no PINN training',
            'kovasznay_Re':reynolds,'kovasznay_max_residual':residual,
            'lifting_max_wall_error':boundary_error,'tested_correction_coefficients':[-3,0,2],
            'cfd_archive_sha256':hashlib.sha256((ROOT/'data/cavity_data.npz').read_bytes()).hexdigest(),
            'upstream_commit':'fcb1566eaa3253d4a4108fbac9d49a38fd10ad6b'}


def build(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    evidence = analytic_preflight()
    width = A4[0]-104
    styles = {
        'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10.8,leading=14.4,spaceAfter=7,alignment=4),
        'title':ParagraphStyle('title',fontName='Times-Bold',fontSize=24,leading=28,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#16354a')),
        'subtitle':ParagraphStyle('subtitle',fontName='Times-Roman',fontSize=17,leading=21,spaceAfter=12,keepWithNext=True),
        'heading':ParagraphStyle('heading',fontName='Times-Bold',fontSize=13,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True),
        'caption':ParagraphStyle('caption',fontName='Times-Roman',fontSize=9.5,leading=12,spaceAfter=9),
        'tablecaption':ParagraphStyle('tablecaption',fontName='Times-Roman',fontSize=9.5,leading=12,spaceAfter=7,keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=9.4,leading=12),
    }
    def paragraph(t, style='body'):
        t = escape(t)
        t = re.sub(r'(https://[^\s]+)',r'<link href="\1" color="#14648a">\1</link>',t)
        return Paragraph(t, styles[style])
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def raster(fig, maxw=width, maxh=235):
        buf = BytesIO(); fig.savefig(buf,format='png',dpi=320,bbox_inches='tight',pad_inches=.09,facecolor='white')
        plt.close(fig); buf.seek(0); im = Image(buf)
        scale = min(maxw/im.imageWidth,maxh/im.imageHeight)
        im.drawWidth=im.imageWidth*scale; im.drawHeight=im.imageHeight*scale
        return im
    def equation(key, num):
        lines=EQS[key]; spacing=.85 if key=='error' else .43
        fig=plt.figure(figsize=(7,spacing*len(lines)+.16))
        for i,line in enumerate(lines): fig.text(.5,1-(i+.55)/len(lines),line,ha='center',va='center',fontsize=14)
        im=raster(fig,maxw=width-35,maxh=110 if key=='error' else 31*len(lines))
        t=Table([[im,paragraph(f'({num})','cell')]],colWidths=[width-28,28])
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('ALIGN',(0,0),(0,0),'CENTER'),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
        return t
    def table(key):
        caption,rows=TABLES[key]
        rows=[[paragraph(v,'cell') for v in row] for row in rows]
        t=Table(rows,colWidths=[width*.23,width*.35,width*.42],repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f4')),('LINEABOVE',(0,0),(-1,0),.7,colors.HexColor('#29475b')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#29475b')),('LINEBELOW',(0,-1),(-1,-1),.7,colors.HexColor('#29475b')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return [paragraph(caption,'tablecaption'),t,Spacer(1,10)]
    def figure(key):
        if key=='cavity':
            with np.load(ROOT/'data/cavity_data.npz',allow_pickle=False) as d:
                i=int(np.flatnonzero(d['Re']==100)[0]); assert bool(d['accepted'][i])
                x,y,u,v=d['x'],d['y'],d['u'][i],d['v'][i]
            fig,ax=plt.subplots(figsize=(5.1,3.9),layout='constrained')
            im=ax.contourf(x,y,np.hypot(u,v),levels=np.linspace(0,1,26),cmap='viridis')
            ax.streamplot(x,y,u,v,color='white',density=.75,linewidth=.55,arrowsize=.7)
            ax.set(xlabel='x / L',ylabel='y / L',aspect='equal')
            ax.annotate('Moving lid',xy=(.82,1.025),xytext=(.2,1.025),arrowprops={'arrowstyle':'->','color':'#16354a'},annotation_clip=False)
            fig.colorbar(im,ax=ax,label='Speed / U',ticks=np.linspace(0,1,6))
            cap='Figure 1. Retained conventional CFD field at Re=100, with speed contours and streamlines. This is a classical-lid FlowMLLab case, not a prediction from the regularized-lid PINN. Equal axis scales preserve the square geometry.'
        elif key=='lid':
            x=np.linspace(0,1,1201); fig,axs=plt.subplots(1,2,figsize=(7,2.65),layout='constrained')
            for ax in axs:
                ax.plot(x,np.ones_like(x),'--',color='#7c8185',label='Ideal lid (interior)')
                for dx,c in [(np.sqrt(.001),'#0072B2'),(.06,'#D55E00')]:
                    ax.plot(x,(1-np.exp(-(1-x)**2/dx**2))*(1-np.exp(-x*x/dx**2)),color=c,label=f'delta_x = {dx:.4f}')
                ax.set(xlabel='x / L',ylabel='Prescribed lid u / U',ylim=(-.025,1.07)); ax.grid(alpha=.15)
            axs[0].set(xlim=(0,1),title='Whole lid'); axs[1].set(xlim=(0,.13),title='Left-corner detail')
            axs[0].legend(fontsize=8,loc='lower center',frameon=False)
            cap='Figure 3. The analytic regularized lid profile. Blue uses the inspected upstream width; orange is an illustrative wider regularization, not another computed flow. The corner detail exposes a boundary-condition difference that a centerline plot alone could miss.'
        else:
            fig,ax=plt.subplots(figsize=(7,2.15)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
            items=[(.08,'Coordinates\nx, y'),(.34,'Smooth MLP\nq, p'),(.62,'Known lifting\npsi = lift + B²q'),(.9,'Derivatives\nu, v; residuals')]
            for x,label in items: ax.text(x,.65,label,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=.5','fc':'#eef4f6','ec':'#7393a6'})
            for a,b in zip(items,items[1:]): ax.annotate('',xy=(b[0]-.105,.65),xytext=(a[0]+.09,.65),arrowprops={'arrowstyle':'->','color':'#16354a'})
            ax.text(.5,.12,'Spatial derivatives form the physics loss; parameter gradients update the MLP.',ha='center',fontsize=9.5)
            cap='Figure 2. The streamfunction-pressure construction. The network supplies a correction, not the prescribed wall motion. Momentum residuals require third spatial derivatives of the physical streamfunction.'
        fig.savefig(outdir/f'{NAME}_{key}.svg',bbox_inches='tight')
        return KeepTogether([raster(fig),Spacer(1,5),paragraph(cap,'caption')])
    story=[]; number=0
    for block in (ROOT/'lectures/source'/f'{NAME}.md').read_text(encoding='utf-8').strip().split('\n\n'):
        if block.startswith('@equation '):
            number+=1; story.append(equation(block.split()[1],number))
        elif block.startswith('@figure '): story.append(figure(block.split()[1]))
        elif block.startswith('@table '): story.extend(table(block.split()[1]))
        elif block.startswith('### '): story.append(paragraph(block[4:],'heading'))
        elif block.startswith('## '): story.append(paragraph(block[3:],'subtitle'))
        elif block.startswith('# '): story.append(paragraph(block[2:],'title'))
        else: story.append(paragraph(block.replace('\n',' ')))
    def footer(c,doc):
        c.saveState(); c.setStrokeColor(colors.HexColor('#b7c3cd')); c.setLineWidth(.4)
        c.line(52,43,A4[0]-52,43); c.setFont('Times-Roman',9)
        c.drawString(52,29,'FlowMLLab | Week 4.2 | PINNs and the lid-driven cavity')
        c.drawRightString(A4[0]-52,29,str(doc.page)); c.restoreState()
    def canvas(*args,**kwargs):
        kwargs['invariant']=1; return Canvas(*args,**kwargs)
    pdf=outdir/f'{NAME}.pdf'
    SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=46,bottomMargin=58,
        title='Physics-Informed Neural Networks: From differential equations to the lid-driven cavity',author='Ehsan Roohi').build(story,onFirstPage=footer,onLaterPages=footer,canvasmaker=canvas)
    (outdir/f'{NAME}_preflight.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(evidence,indent=2)); print(pdf)
    return pdf


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/pdf')
    parser.add_argument('--publish-copy',action='store_true')
    args=parser.parse_args(); pdf=build(args.output_dir)
    if args.publish_copy: shutil.copy2(pdf,ROOT/'lectures'/pdf.name)
