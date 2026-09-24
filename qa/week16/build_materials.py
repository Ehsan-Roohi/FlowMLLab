"""Build the Week 16 lecture and execute its data-based teaching notebook."""
from pathlib import Path
import json,html,sys,os
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'results/week16_lowboom'


def notebook():
    """Execute the curated Week 16 teaching notebook without regenerating its content.

    The notebook itself is the maintained instructional source. Keeping execution
    separate prevents the build script from overwriting the pedagogical Markdown,
    equations, exercises, and lightweight no-output source committed to the repo.
    """
    target=ROOT/'notebooks/week16/W16_Supersonic_Shape_Optimization.ipynb'
    nb=nbf.read(target,as_version=4)
    assert nb.nbformat==4
    assert len(nb.cells)>=30
    assert any(c.cell_type=='markdown' and 'Why this is not yet a ground sonic-boom calculation' in ''.join(c.source) for c in nb.cells)
    assert any(c.cell_type=='markdown' and 'Proper Orthogonal Decomposition' in ''.join(c.source) for c in nb.cells)
    source_nb=nbf.from_dict(json.loads(nbf.writes(nb)))
    if os.environ.get('FLOWMLLAB_INPROCESS')=='1':
        from execute_inprocess import execute
        executed=execute(source_nb)
        method='IPython in-process'
    else:
        executed=NotebookClient(
            source_nb,
            timeout=600,
            kernel_name='python3',
            resources={'metadata':{'path':str(ROOT)}}
        ).execute()
        method='nbclient with a real Jupyter kernel'
    # Keep the committed notebook lightweight: validate execution but do not
    # persist generated figures/tables as embedded output blobs.
    for cell in source_nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    nbf.write(source_nb,target)
    print('Validated educational notebook:',len(source_nb.cells),'cells')
    report=json.loads((E/'release_check.json').read_text())
    report['notebook_executed']=True
    report['notebook_execution_method']=method
    report['notebook_source_kept_lightweight']=True
    (E/'release_check.json').write_text(json.dumps(report,indent=2))

def lecture():
    pdfmetrics.registerFont(TTFont('FlowSerif',font_manager.findfont('DejaVu Serif')))
    pdfmetrics.registerFont(TTFont('FlowSerifBold',font_manager.findfont(font_manager.FontProperties(family='DejaVu Serif',weight='bold'))))
    st={
     'body':ParagraphStyle('body',fontName='FlowSerif',fontSize=10,leading=14,alignment=4,spaceAfter=7),
     'title':ParagraphStyle('title',fontName='FlowSerifBold',fontSize=24,leading=29,textColor=colors.HexColor('#17384d'),spaceAfter=14),
     'h':ParagraphStyle('h',fontName='FlowSerifBold',fontSize=13,leading=17,textColor=colors.HexColor('#17384d'),spaceBefore=13,spaceAfter=7,keepWithNext=True),
     'small':ParagraphStyle('small',fontName='FlowSerif',fontSize=8.5,leading=11,spaceAfter=7),
     'cell':ParagraphStyle('cell',fontName='FlowSerif',fontSize=8,leading=10)}
    def para(t,k='body'):return Paragraph(html.escape(t),st[k])
    width=A4[0]-104
    def table(rows):
        obj=Table([[para(str(v),'cell') for v in row] for row in rows],colWidths=[width/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f0f4')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#17384d')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return obj
    figs={'4.':('reference/clean_cfd_fields','Actual accepted SU2 8.0.1 pressure fields for the baseline and retained optimized geometry. Both panels use identical Cp limits, exported cell connectivity and equal physical coordinate scales.'),'5.':('numerical_verification','Historical SU2 8.5.0 mesh and residual comparisons. These numerical checks did not detect the later identified tip enthalpy defect.'),'7.':('reference/clean_model_learning','Clean SU2 8.0.1 geometry split and new retained neural model: aggregate split errors, including poor extrapolation. The waveform compares the actual clean test CFD with the saved model.'),'8.':('reference/weakwall_design_validation','New SU2 8.0.1 CFD at the original candidate geometries, both design meshes and two off-design Mach numbers. All plotted cases pass full-field physical and convergence checks.'),'9.':('condition_checks','Historical SU2 8.5.0 observation-line and Mach comparisons. Accepted new 8.0.1 off-design results are reported separately.')}
    reference_figures={
        '11.':('reference/seeb_geometry','NASA SEEB-ALR as-built geometry. The nose and sting must be preserved when constructing the axisymmetric computational domain.'),
        '17.':('reference/weakwall_checkpoint_validation','Unchanged retained neural checkpoint against eight physically checked SU2 8.0.1 calculations on the same archived meshes and configurations. The older 8.5.0 physical audit remains failed.'),
        '13.':('reference/independent_neural_test','Recovered frozen neural predictions compared with eight finer-mesh CFD signatures. These predictions are not replaced by notebook refitting.')}
    figs.update({key:value for key,value in reference_figures.items() if (E/(value[0]+'.png')).is_file()})
    summary=json.loads((E/'summary.json').read_text())
    design=json.loads((E/'reference/weakwall_design_audit.json').read_text())
    cone=json.loads((E/'reference/cone_refinement_v801.json').read_text())
    accepted=json.loads((E/'reference/weakwall_checkpoint_audit.json').read_text())
    clean=json.loads((E/'reference/clean_model_audit_v801.json').read_text())
    assert design['passed'] and cone['passed'] and accepted['passed'] and clean['passed']
    source=(ROOT/'lectures/source/week16_supersonic_shape_optimization.md').read_text()
    story=[para('Supersonic Shape Optimization with Verified CFD','title'),para('FlowMLLab | Week 16 | Ehsan Roohi | MIE 690A','small'),para('Gmsh meshes, axisymmetric Euler simulations, learned models and independently recomputed designs.','body')]
    nfig=0
    equations={
      '2.':[r'$r(s)=c\sin(\pi s)\exp[a(2s-1)+b\cos(2\pi s)]$',r'$c=\sqrt{\frac{V}{\pi L\int_0^1 f(s)^2\,ds}}$'],
      '3.':[r'$\frac{\partial(rU)}{\partial t}+\frac{\partial(rF_x)}{\partial x}+\frac{\partial(rF_r)}{\partial r}=(0,0,p,0)^T$'],
      '6.':[r'$D_p=2\pi\int_0^L (p-p_\infty)r\frac{dr}{dx}\,dx,\qquad C_{D,p}=\frac{D_p}{q_\infty L^2}$'],
      '7.':[r'$C_p(x;\theta)\approx\overline{C}_p(x)+\sum_{k=1}^{K}z_k(\theta)\phi_k(x)$']}
    for section in source.split('\n## ')[1:]:
        title,content=section.split('\n',1);story.append(para(title,'h'))
        for block in content.strip().split('\n\n'):story.append(para(block.replace('\n',' '),'small' if title=='References' else 'body'))
        key=title.split(' ')[0]
        if key in equations:
            for eq in equations[key]:
                fig=plt.figure(figsize=(7,.58));fig.text(.5,.5,eq,ha='center',va='center',fontsize=14);buf=BytesIO();fig.savefig(buf,format='png',dpi=220,bbox_inches='tight',pad_inches=.08);plt.close(fig);buf.seek(0)
                im=Image(buf);factor=min(width/im.imageWidth,55/im.imageHeight);im.drawWidth*=factor;im.drawHeight*=factor;story.extend([Spacer(1,5),im,Spacer(1,5)])
        if key in figs:
            name,caption=figs[key];im=Image(str(E/(name+'.png')));scale=min(width/im.imageWidth,(500 if key in reference_figures or key in ['4.','7.'] else 250)/im.imageHeight);im.drawWidth*=scale;im.drawHeight*=scale;nfig+=1
            story.append(KeepTogether([Spacer(1,6),im,Spacer(1,4),para(f'Figure {nfig}. {caption}','small')]))
        if key=='5.':
            rows=[['SU2 8.0.1 check','Observed difference','Criterion'],['Finest cone Cp',f"{100*cone['levels'][-1]['analytical_comparison']['relative_error']:.2f}%",'<3%'],['Cone last-two mean Cp',f"{100*cone['last_two_cp_relative_change']:.2f}%",'<3%']]
            for m in design['mesh_sensitivity']:
                rows.extend([[m['design']+' peak, mesh',f"{100*m['peak_relative_change']:.2f}%",'<5%'],[m['design']+' drag, mesh',f"{100*m['drag_relative_change']:.2f}%",'<3%']])
            story.extend([Spacer(1,6),table(rows)])
        if key=='7.':
            rows=[['Clean neural model','Wave L2','Peak error','Drag error','Worst wave']]
            for name,row in [*clean['same_mesh'].items(),('Finer CFD test',clean['finer_mesh'])]:
                rows.append([name,*[f"{100*row[k]:.2f}%" for k in ['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error','worst_case_wave_relative_l2']]])
            story.extend([Spacer(1,6),table(rows)])
        if key=='8.':
            a,b=design['design_pairs']
            rows=[['New SU2 8.0.1 result','Mesh level 1.5','Mesh level 2'],['Peak reduction',f"{100*a['peak_reduction']:.2f}%",f"{100*b['peak_reduction']:.2f}%"],['Drag change',f"{100*a['drag_change']:.2f}%",f"{100*b['drag_change']:.2f}%"]];story.extend([Spacer(1,6),table(rows)])

    def page(c,doc):
        c.setFont('FlowSerif',8);c.setFillColor(colors.HexColor('#5c6c76'));c.drawString(52,29,'FlowMLLab | Week 16 | Supersonic shape optimization');c.drawRightString(A4[0]-52,29,str(doc.page))
    target=ROOT/'lectures/week16_supersonic_shape_optimization.pdf'
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=43,bottomMargin=47,title='FlowMLLab Week 16: Supersonic Shape Optimization',author='Ehsan Roohi').build(story,onFirstPage=page,onLaterPages=page)
    print('Lecture:',target)
if __name__=='__main__':
    notebook();lecture()
