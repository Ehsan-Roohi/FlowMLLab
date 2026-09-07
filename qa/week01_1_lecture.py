"""Publication-style Week 1.1 lecture; figures derive from the scientific audit."""
from pathlib import Path
import shutil
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import NullLocator
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader

W, H = 960, 540
INK, TEAL, MUTED = '#162C40', '#007F82', '#53677B'
PAPER, RULE, RED = '#FAFBFC', '#DCE4E9', '#B74232'


class Deck:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H), invariant=1)
        self.c.setTitle('FlowMLLab | Week 1.1 | AI-assisted scientific software')
        self.c.setAuthor('Ehsan Roohi')
        self.n = 0

    def text(self, text, x, top, width, size=18, color=INK, bold=False, maxh=100):
        p = Paragraph(text, ParagraphStyle('t', fontName='Helvetica-Bold' if bold else 'Helvetica',
            fontSize=size, leading=size*1.32, textColor=HexColor(color)))
        _, h = p.wrap(width, 1000)
        if h > maxh:
            raise ValueError(f'Overflow on slide {self.n}: {text[:70]} ({h}>{maxh})')
        p.drawOn(self.c, x, H-top-h)
        return h

    def line(self, x1, top1, x2, top2, color=RULE, width=1):
        self.c.setStrokeColor(HexColor(color)); self.c.setLineWidth(width)
        self.c.line(x1, H-top1, x2, H-top2)

    def page(self, section, title, subtitle):
        if self.n: self.c.showPage()
        self.n += 1
        self.c.setFillColor(HexColor(PAPER)); self.c.rect(0,0,W,H,fill=1,stroke=0)
        self.text('FLOWMLLAB  /  WEEK 1.1', 44, 22, 380, 10, TEAL, True)
        self.text(section.upper(), 690, 22, 226, 10, MUTED)
        self.text(title, 44, 57, 872, 29, bold=True, maxh=77)
        self.text(subtitle, 44, 103, 865, 14, MUTED, maxh=43)
        self.line(44, 492, 916, 492)
        self.text('Ehsan Roohi  |  AI in Fluid Mechanics',44,507,620,9,MUTED)
        self.text(f'{self.n:02d} / 13',865,507,70,9,MUTED)

    def takeaway(self, text):
        self.line(44,447,916,447,TEAL,2)
        self.text(text,44,461,865,13,TEAL,True,maxh=30)

    def figure(self, fig, x, top, width, height):
        buffer = BytesIO(); fig.savefig(buffer,format='png',dpi=240,facecolor=PAPER,bbox_inches='tight',pad_inches=.08)
        plt.close(fig); buffer.seek(0)
        self.c.drawImage(ImageReader(buffer),x,H-top-height,width,height,preserveAspectRatio=True,anchor='c',mask='auto')

    def equation(self, text, x, top, width, height=58):
        fig = plt.figure(figsize=(width/90,height/90))
        fig.text(.01,.5,text,fontsize=21,color=INK,va='center')
        self.figure(fig,x,top,width,height)

    def table(self, rows, widths, top=164, rowh=37):
        x0=44
        for j,row in enumerate(rows):
            y=top+j*rowh
            if j==0:
                self.c.setFillColor(HexColor(INK)); self.c.rect(x0,H-y-rowh,872,rowh,fill=1,stroke=0)
            else: self.line(x0,y+rowh,916,y+rowh)
            x=x0
            for value,width in zip(row,widths):
                self.text(str(value),x+10,y+10,width-20,12 if j else 11,'#FFFFFF' if j==0 else INK,j==0,maxh=rowh-12)
                x+=width


def build_lecture(output, lecture, verification, cavity, record, faulty_error, root):
    from flowmllab.scientific_software import load_cavity_case, manufactured_incompressible_field
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':MUTED})
    output.parent.mkdir(parents=True,exist_ok=True)
    d=Deck(output)
    case,_=load_cavity_case(root,100.)
    x,y,u,v,exact=manufactured_incompressible_field(129)
    d.page('Scientific computing','AI-assisted scientific software','From a proposed implementation to a defensible scientific result')
    d.text('Can you trust a<br/>plausible flow field?',44,178,425,33,bold=True,maxh=105)
    d.text('A worked investigation of derivatives, conservation and reproducibility using the Re = 100 cavity.',44,305,390,18,maxh=100)
    fig,ax=plt.subplots(figsize=(4.2,3.2))
    speed=np.hypot(case['u'],case['v'])
    ax.contourf(case['x'],case['y'],speed,32,cmap='viridis')
    ax.streamplot(case['x'],case['y'],case['u'],case['v'],color='white',density=.8,linewidth=.6,arrowsize=.7)
    ax.set(xlabel='x / L',ylabel='y / L',aspect='equal',title='Retained cavity velocity field')
    d.figure(fig,496,152,410,272)
    d.takeaway('Learning objective: write a contract, challenge the code, and defend a bounded claim.')

    d.page('01 / Scientific claims','Four questions before accepting a result','Each question requires a different kind of evidence.')
    for i,(a,b,c) in enumerate([
        ('Execution','Does the program run?','Imports, finite outputs, expected shapes.'),
        ('Verification','Does it solve the equations correctly?','A known solution and a mesh-convergence study.'),
        ('Validation','Does the model represent the physical system?','Independent physical or experimental reference data.'),
        ('Reproducibility','Can the result be reconstructed?','Identified inputs, environment, commands and outputs.')]):
        top=164+i*65
        d.text(a,44,top,160,17,TEAL,True)
        d.text(b,220,top,690,17,bold=True)
        d.text(c,220,top+26,690,14,MUTED)
    d.takeaway('This lab verifies a post-processor; the cavity check audits consistency with a retained field.')

    d.page('02 / Specification','Make the scientific contract executable','Define semantics before asking a person or an agent to implement the diagnostic.')
    d.table([['Contract item','Explicit choice','Why it matters'],
        ['Coordinates','u[j, i] = u(y[j], x[i])','Axis swaps can preserve array shape.'],
        ['Quantities','Divergence and out-of-plane vorticity','Signs determine physical interpretation.'],
        ['Support','Exclude two layers for the interior audit','Boundary stencils have different support.'],
        ['Precision','Float64; fixed tolerances','Small residuals depend on numerical precision.'],
        ['Identity','Exact SHA-256 of the cavity archive','A changed input invalidates the record.'],
        ['Decision','All seven gates must pass','One failure rejects the current claim.']], [150,352,370],rowh=38)
    d.takeaway('Deliverable: SCIENTIFIC_SPEC.md plus a machine-readable acceptance record.')

    d.page('03 / Differential operators','Axes and signs are part of the mathematics','Cartesian coordinates; u is the x-velocity and v is the y-velocity.')
    d.equation(r'$\nabla\!\cdot\!\mathbf{u}=\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}$',50,157,390,65)
    d.equation(r'$\omega_z=\frac{\partial v}{\partial x}-\frac{\partial u}{\partial y}$',520,157,380,65)
    d.text('Positive vorticity means counterclockwise rotation in the x-y plane.',44,250,390,18,maxh=80)
    d.text('axis=1 differentiates along x.<br/>axis=0 differentiates along y.<br/>Coordinates supply the spacing.',520,250,380,18,maxh=100)
    d.equation(r'$\left.\frac{\partial f}{\partial x}\right|_{j,i}=\frac{f_{j,i+1}-f_{j,i-1}}{2\Delta x}+\mathcal{O}(\Delta x^2)$',65,355,800,62)
    d.takeaway('The centered interior stencil has second-order truncation error on a uniform grid.')

    d.page('04 / Exact reference','Construct a field with a known answer','The streamfunction makes incompressibility an analytic identity.')
    d.equation(r'$\psi=\sin^2(\pi x)\sin^2(\pi y),\qquad (x,y)\in[0,1]^2$',50,151,835,58)
    d.equation(r'$u=\psi_y,\qquad v=-\psi_x,\qquad \nabla\!\cdot\!\mathbf{u}=0$',50,219,835,58)
    d.equation(r'$\omega_z=-\nabla^2\psi=-2\pi^2[\cos(2\pi x)\sin^2(\pi y)+\sin^2(\pi x)\cos(2\pi y)]$',50,290,850,67)
    d.text('All wall velocities vanish. Sample the analytic expressions directly; compare numerical derivatives with the analytic vorticity.',50,381,850,17,maxh=50)
    d.takeaway('A zero divergence residual alone cannot establish that vorticity is correct.')

    d.page('05 / Convergence','Demonstrate accuracy across four grids','Use the same two-layer interior support at every resolution.')
    rows=verification['rows']
    fig,ax=plt.subplots(figsize=(5.2,3.5))
    hs=np.array([r['h'] for r in rows]); errors=np.array([r['vorticity_relative_l2'] for r in rows])
    ax.loglog(hs,errors,'o-',color=TEAL,label='Measured error',lw=2)
    ax.loglog(hs,errors[-1]*(hs/hs[-1])**2,'--',color=MUTED,label='Second-order reference')
    ax.set(xlabel='Grid spacing h',ylabel='Relative L2 vorticity error'); ax.grid(alpha=.2,which='both'); ax.legend(frameon=False,fontsize=10)
    ax.set_xticks(hs, ['1/16','1/32','1/64','1/128']); ax.xaxis.set_minor_locator(NullLocator())
    d.figure(fig,40,156,475,270)
    d.text(f"p = {verification['observed_order']:.4f}",555,173,345,32,TEAL,True)
    d.text('Measured from the two finest grids.',555,225,335,16,MUTED)
    d.equation(r'$p=\frac{\log(E_{2h}/E_h)}{\log 2}$',555,270,325,65)
    d.text(f"Finest-grid error: {errors[-1]:.3e}<br/>Grids: 17, 33, 65, 129",555,354,345,17,maxh=65)
    d.takeaway('The measured rate agrees with the expected second-order stencil over this grid sequence.')

    d.page('06 / Real field','Audit the retained Re = 100 cavity','The cavity has a moving lid; the manufactured field on the previous slides does not.')
    # Use the same derivative definition as the evidence record.
    omega=np.gradient(case['v'],case['x'],axis=1,edge_order=2)-np.gradient(case['u'],case['y'],axis=0,edge_order=2)
    fig,ax=plt.subplots(figsize=(4.7,3.5))
    lim=np.max(np.abs(omega))
    positive=np.geomspace(.1,lim,30)
    levels=np.r_[-positive[::-1],0,positive]
    im=ax.contourf(case['x'],case['y'],omega,levels=levels,norm=SymLogNorm(linthresh=1,vmin=-lim,vmax=lim),cmap='RdBu_r')
    ax.streamplot(case['x'],case['y'],case['u'],case['v'],color=INK,density=.65,linewidth=.45,arrowsize=.6)
    ax.set(xlabel='x / L',ylabel='y / L',aspect='equal',title='Vorticity; symmetric log color scale')
    fig.colorbar(im,ax=ax,label='Vorticity (archive units)',shrink=.85,ticks=[-80,-10,-1,0,1,10,80])
    d.figure(fig,35,152,467,282)
    for i,(a,b) in enumerate([('Grid','65 x 65'),('Interior divergence RMS',f"{cavity['interior_divergence_rms']:.2e}"),('Vorticity disagreement',f"{cavity['archive_vorticity_relative_l2']:.3%}"),('Wall maximum error',f"{cavity['wall_velocity_max_abs_error']:.1e}")]):
        t=170+i*61; d.text(a,548,t,340,13,MUTED); d.text(b,548,t+21,340,22,TEAL,True)
    d.takeaway('The 2.902% value compares a derivative with archived vorticity; it is not error against exact flow.')

    d.page('07 / Acceptance','Read the evidence against fixed thresholds','These tolerances belong to this float64 diagnostic and dataset.')
    d.table([['Check','Observed','Required'],
        ['Dataset identity','Exact SHA-256 match','Exact match'],
        ['Observed convergence order',f"{verification['observed_order']:.4f}",'At least 1.90'],
        ['Finest analytic vorticity error',f'{errors[-1]:.3e}','At most 5e-4'],
        ['Analytic divergence RMS',f"{max(r['divergence_rms'] for r in rows):.3e}",'At most 1e-12'],
        ['Cavity divergence RMS',f"{cavity['interior_divergence_rms']:.3e}",'At most 1e-12'],
        ['Cavity/archive vorticity error',f"{cavity['archive_vorticity_relative_l2']:.3e}",'At most 4e-2'],
        ['Wall maximum error','0.0','At most 1e-12']], [385,230,257],top=150,rowh=35)
    d.takeaway('All seven gates pass. A passed gate supports only the claim that the gate actually tests.')

    d.page('08 / Adversarial test','A plausible axis swap survives a shape test','Compare the correct vorticity with the intentionally wrong expression.')
    d.equation(r'$\omega_{\mathrm{wrong}}=D_y v-D_x u$',48,157,410,70)
    d.text('Same shape. Finite values.<br/>Different mathematical quantity.',48,249,405,21,bold=True,maxh=100)
    d.text('The analytic reference detects the error before the cavity is used.',48,345,405,17,maxh=70)
    d.text(f'{faulty_error:.3f}',550,165,345,55,RED,True)
    d.text('Relative L2 error',550,245,345,18,MUTED)
    d.line(550,290,896,290)
    d.text('Required: at most 0.0005<br/><b>Decision: REJECT</b>',550,316,345,21,RED,maxh=95)
    d.takeaway('Keep this failing example: it shows what the acceptance procedure can detect.')

    d.page('09 / AI workflow','Give the assistant a reviewable task','A scientific specification becomes the implementation brief.')
    d.text('Example implementation brief',44,159,850,17,TEAL,True)
    d.text('Implement divergence and z-vorticity for u[y,x], v[y,x] on increasing Cartesian coordinates. Use second-order finite differences. Preserve the sign convention and the two-layer audit support. Return diagnostics; reject invalid inputs.',44,197,850,20,maxh=115)
    for i,(a,b) in enumerate([('1. Propose','Code and tests'),('2. Execute','Frozen acceptance gates'),('3. Review','Physics, diff and evidence')]):
        xx=44+i*296; d.line(xx,339,xx+262,339,TEAL,2); d.text(a,xx,355,262,19,bold=True); d.text(b,xx,389,262,15,MUTED)
    d.takeaway('Record AI assistance and human corrections. Scientific responsibility stays with the investigator.')

    d.page('10 / Reproducibility','Make the result reconstructable','The record must identify both the input and the reasoning behind acceptance.')
    d.text('Data identity',44,167,255,21,TEAL,True)
    d.text('SHA-256 binds this result to the retained cavity archive. A hash mismatch rejects the record.',44,212,255,17,maxh=115)
    d.text('Execution context',347,167,255,21,TEAL,True)
    d.text('Keep code revision, dependency versions and the exact command. Re-run from a clean checkout.',347,212,255,17,maxh=115)
    d.text('Decision evidence',650,167,266,21,TEAL,True)
    d.text('Retain thresholds, measured values and failures. Inspect the record together with the implementation.',650,212,266,17,maxh=115)
    d.line(44,350,916,350)
    d.text('python qa/build_week01_1_materials.py --execute',44,374,870,18,bold=True)
    d.takeaway('Numerical reproducibility and byte-identical rendering are separate checks; both depend on environment.')

    d.page('11 / Assessment','Defend the claim with an evidence package','Submit the specification, implementation, test results and a short scientific review.')
    d.table([['Assessment','Weight','Evidence'],['Specification','20%','Axes, units, sign, support and thresholds'],['Verification','25%','Analytic reference and measured convergence'],['Physical audit','20%','Boundaries and retained-field consistency'],['Manual review','15%','A scientific error found or ruled out'],['Provenance','10%','Input hash, revision, environment, command'],['Claim boundary','10%','Limits and retained rejection example']], [220,100,552],rowh=38)
    d.takeaway('Discussion: what new reference and test are needed before extending this diagnostic to an unstructured mesh?')

    d.page('12 / Reading','What the evidence supports','A verified Cartesian post-processor, with a consistency audit on one retained cavity case.')
    d.text('The next claim needs new evidence',44,155,860,21,TEAL,True)
    d.text('Stretched meshes, unstructured grids and other boundary treatments require their own verification. Physical validation of a solver also requires an independent physical reference.',44,193,865,18,maxh=75)
    refs=[('Wilson et al. (2017)','Good Enough Practices in Scientific Computing','10.1371/journal.pcbi.1005510'),('Sandve et al. (2013)','Ten Simple Rules for Reproducible Computational Research','10.1371/journal.pcbi.1003285'),('Roache (1998)','Verification and Validation in Computational Science and Engineering','Book: Hermosa Publishers')]
    for i,(a,b,c) in enumerate(refs):
        t=286+i*47; d.text(a,44,t,190,12,bold=True); d.text(b,248,t,668,12); d.text(c,248,t+19,668,10,MUTED)
    d.takeaway('Original FlowMLLab material. CS146S inspired the syllabus topic: themodernsoftware.dev')
    d.c.save(); shutil.copy2(output,lecture)
    return output
