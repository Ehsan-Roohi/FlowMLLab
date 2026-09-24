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
    nb=nbf.v4.new_notebook();cells=[]
    def md(t):cells.append(nbf.v4.new_markdown_cell(t))
    def code(t):cells.append(nbf.v4.new_code_cell(t))
    md(r'''# Week 16 — Supersonic Shape Optimization with Verified CFD
**FlowMLLab · Ehsan Roohi · MIE 690A**

[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week16/W16_Supersonic_Shape_Optimization.ipynb)

This complete worked lab uses newly generated Gmsh/SU2 data. You will inspect numerical verification, compare learned models, propose a fixed-volume shape, and examine fresh CFD checks.

**Claim boundary:** the computed target is near-field peak pressure at $r/L=0.5$. This notebook does not compute atmospheric propagation or PLdB, and its axisymmetric body is not TMS-10. Read [the assignment](ASSIGNMENT.md), [the model validation guide](MODEL_VALIDATION_GUIDE.md), and [the NASA reference walkthrough](NASA_REFERENCE_GUIDE.md) for assessed extensions and the distinction between analytical, experimental and learned-model checks.

Run from a clone of FlowMLLab. Dependencies for the data-based lab: numpy, scipy, scikit-learn, pandas, matplotlib. Gmsh and SU2 are required only for regenerating CFD.''')
    code('''# FLOWMLLAB_COLAB_BOOTSTRAP_V1: sparse checkout for Week 16
from pathlib import Path
import sys, json, hashlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p/'qa/week16/learning.py').exists()), None)
if ROOT is None:
    import importlib.util, subprocess
    if importlib.util.find_spec('google.colab') is not None:
        target=Path('/content/FlowMLLab_week16')
        if not target.exists():
            subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/Ehsan-Roohi/FlowMLLab.git',str(target)],check=True)
            subprocess.run(['git','-C',str(target),'sparse-checkout','set','qa/week16','results/week16_lowboom','notebooks/week16','cases/week16_lowboom/reference'],check=True)
        ROOT=target
    else:
        raise RuntimeError('Clone FlowMLLab, then open this notebook inside the checkout.')
sys.path.insert(0, str(ROOT/'qa/week16'))
from cfd import radius, VOLUME
from learning import Surrogate, optimize
E = ROOT/'results/week16_lowboom'
data = np.load(E/'dataset.npz')
summary = json.loads((E/'summary.json').read_text())
assert hashlib.sha256((E/'dataset.npz').read_bytes()).hexdigest() == summary['dataset_sha256']
print('Dataset verified:', summary['dataset_sha256'])
print('CFD cases:', len(data['parameters']))
display(pd.Series(json.loads((E/'software_versions.json').read_text()), name='Recorded generation environment'))''')
    md(r'''## 1. Geometry and volume
For $s=x/L$, define $f(s)=\sin(\pi s)\exp[a(2s-1)+b\cos(2\pi s)]$ and $r(s)=c f(s)$.
The constraint gives $c=\sqrt{V/(\pi L\int_0^1 f(s)^2\,ds)}$.

These are three-dimensional bodies of revolution. The two-dimensional mesh is a meridional representation of the axisymmetric equations, including their radial source terms.''')
    code('''x = np.linspace(0,1,2001)
fig,ax = plt.subplots(figsize=(9,2.5))
checks=[]
for a,b in [(0,0),(.3,.15),(-.3,-.15)]:
    r=radius(x,a,b)
    line,=ax.plot(x,r,label=f'a={a}, b={b}')
    ax.plot(x,-r,color=line.get_color())
    volume=np.pi*np.trapezoid(r*r,x)
    checks.append({'a':a,'b':b,'volume':volume,'relative_error':abs(volume/VOLUME-1)})
ax.set(xlabel='x/L',ylabel='r/L',aspect='equal');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.5));plt.show()
display(pd.DataFrame(checks))
assert max(r['relative_error'] for r in checks)<1e-5''')
    md(r'''## 2. Accepting the CFD data
Pressure is reconstructed as $p=(\gamma-1)[\rho E-(\rho u)^2/(2\rho)-(\rho v)^2/(2\rho)]$.
The pressure coefficient is $C_p=(p-p_\infty)/(\tfrac12\gamma p_\infty M^2)$.

We separately check an independent cone solution, discrete convergence, mesh sensitivity and domain sensitivity. The cone benchmark is not a complete sonic-boom validation.''')
    code('''display(pd.Series(summary['benchmark']))
display(pd.Series(summary['validation']))
cases = pd.read_csv(E/'case_metrics.csv')
assert len(cases)==44 and cases['converged'].all()
display(cases[['name','cells','iterations','residual_drop','drag_tail_relative_range']].head(8))
display(Image(filename=str(E/'numerical_verification.png')))''')
    md(r'''### Recompute the Taylor-Maccoll reference
The conical similarity equations are integrated from an oblique shock to the cone wall. A scalar root search adjusts the shock angle until the polar velocity vanishes at the 7-degree wall. This ODE calculation is independent of the SU2 finite-volume discretization. It verifies cone wall pressure, not an off-body boom waveform.''')
    code('''from benchmark import cone_exact
exact=cone_exact(mach=1.8,theta_deg=7.)
display(pd.Series(exact,name='Independently integrated cone solution'))
assert abs(exact['cp']-summary['benchmark']['exact']['cp'])<1e-8
print('SU2 cone pressure relative error [%]:',100*summary['benchmark']['relative_error'])''')
    md(r'''### NASA SEEB-ALR: read the independent reference first
The NASA body is a separate validation geometry at Mach 1.6. Its original STEP geometry, two wind-tunnel records, coordinate-transform macros and NASA-hosted LAVA computation are retained under `cases/week16_lowboom/reference/`. See [the NASA guide](NASA_REFERENCE_GUIDE.md) for nose geometry, units, sampling position and mesh requirements.

The following cell recomputes the **retained LAVA-versus-experiment** comparison. It is not a new SU2 result or experimental validation of the neural model. The pressure variable is $\Delta p/p_\infty$, related to $C_p$ by $\Delta p/p_\infty=\gamma M^2 C_p/2$. At Mach 1.6 this multiplier is 1.792. Source-prescribed shifts align the coordinates; no fit is used to improve agreement.''')
    code('''from nasa_reference_data import report as nasa_report, load_reference
nasa=nasa_report(write=False)
print(nasa['comparison'])
print('Fixed comparison window [inches]:',nasa['window_inches'])
display(pd.DataFrame(nasa['metrics']).T)
experiments,lava=load_reference()
fig,ax=plt.subplots(figsize=(9,4))
for label,curve in experiments.items():
    plot_mask=(curve['x']>=25)&(curve['x']<=46)
    curve={key:value[plot_mask] for key,value in curve.items()}
    line,=ax.plot(curve['x'],curve['pressure'],label=label)
    ax.fill_between(curve['x'],curve['pressure']-curve['uncertainty'],curve['pressure']+curve['uncertainty'],color=line.get_color(),alpha=.15)
ax.plot(lava['x'],lava['pressure'],'k--',label='NASA-hosted LAVA CFD')
ax.set(xlim=(25,46),xlabel='Source-aligned x [inches]',ylabel='delta p / p infinity')
ax.legend();fig.tight_layout();plt.show()
print(nasa['alignment'])
print(nasa['interpolation'])''')
    md(r'''### Our SU2 calculation against the NASA measurements
This result uses the supplied as-built meridian at Mach 1.6, independently generated Gmsh meshes, and axisymmetric Euler in SU2. The plotted pressure difference, three-mesh comparison and numerical gates come from the retained run evidence. This validates this CFD benchmark within the stated tolerances; the neural network has not been trained or experimentally validated on this geometry.''')
    code('''nasa_cfd=json.loads((E/'reference/seeb_validation.json').read_text())
rows=[]
for run in nasa_cfd['runs']:
    for label,metrics in run['experimental_metrics'].items():
        rows.append(dict(level=run['level'],cells=run['cells'],experiment=label,
                         wave_error_percent=100*metrics['wave_relative_l2'],
                         peak_error_percent=100*metrics['peak_relative_error']))
display(pd.DataFrame(rows))
display(pd.DataFrame([{key:run['metadata'][key] for key in ['level','cells','iterations','density_residual_log10','residual_drop','drag_tail_relative_range','min_pressure','min_density']} for run in nasa_cfd['runs']]))
display(Image(filename=str(E/'reference/seeb_mesh.png')))
display(Image(filename=str(E/'reference/seeb_convergence.png')))
print('Last two mesh waveform difference [%]:',100*nasa_cfd['last_two_mesh_wave_relative_l2'])
print('Declared gates:',nasa_cfd['checks'])
assert nasa_cfd['passed']
display(Image(filename=str(E/'reference/seeb_geometry.png')))
display(Image(filename=str(E/'reference/seeb_validation.png')))
print(nasa_cfd['scope'])''')
    md(r'''## 3. Frozen geometry split
POD and scalers are fitted on training cases only. Validation selects the design model. The test cases remain outside fitting and selection. Extrapolation extends parameter $b$ beyond the training interval. All splits belong to the same two-parameter geometry family.''')
    code('''display(pd.Series(data['splits']).value_counts().rename('Cases'))
fig,ax=plt.subplots(figsize=(7,4))
for split,marker in [('train','o'),('validation','s'),('test','^'),('extrapolation','x')]:
    mask=data['splits']==split
    ax.scatter(*data['parameters'][mask].T,label=split,marker=marker)
ax.set(xlabel='a',ylabel='b');ax.legend();plt.show()
assert len(set(data['names']))==44
assert len(np.unique(data['parameters'],axis=0))==44''')
    md(r'''## 4. Compare ridge, a neural network and a Gaussian process
The POD expansion is $C_p(x;\theta)\approx\overline C_p(x)+\sum_{k=1}^{K}z_k(\theta)\phi_k(x)$.
Each model predicts the same modal coefficients and log pressure drag. This notebook reruns model fitting, not the expensive CFD campaign.

The MLP has two 32-neuron tanh hidden layers and a fixed MLP seed. The PCA automatic solver is not seeded, so refitting can change the complete pipeline; see the model validation guide. This is a teaching comparison, not a statistical architecture ranking. Record any optimizer warnings rather than suppressing them.''')
    code('''train=data['splits']=='train'
models={};scores=[]
for kind in ['ridge','mlp','gp']:
    model=Surrogate(kind).fit(data['parameters'][train],data['waveforms'][train],data['cd'][train])
    models[kind]=model
    for split in ['validation','test','extrapolation']:
        mask=data['splits']==split
        w,cd=model.predict(data['parameters'][mask]);truth=data['waveforms'][mask]
        scores.append(dict(model=kind,split=split,
            waveform_error=np.linalg.norm(w-truth)/np.linalg.norm(truth),
            peak_error=np.mean(abs(w.max(axis=1)/truth.max(axis=1)-1)),
            drag_error=np.mean(abs(cd/data['cd'][mask]-1))))
    print(kind,'training seconds:',round(model.seconds,3),'warnings:',model.fit_warnings)
scores=pd.DataFrame(scores)
display(scores)
val=scores[scores.split=='validation'].copy()
selected=val.loc[(val.peak_error+val.drag_error).idxmin(),'model']
print('Validation-selected model:',selected)''')
    code('''test=np.where(data['splits']=='test')[0]
fig,axs=plt.subplots(2,2,figsize=(10,6))
for ax,i in zip(axs.flat,test[:4]):
    ax.plot(data['x'],data['waveforms'][i],'k',lw=2,label='SU2')
    for kind,model in models.items():
        w,_=model.predict(data['parameters'][i]);ax.plot(data['x'],w[0],label=kind)
    ax.set(title=str(data['names'][i]),xlabel='x/L',ylabel='Cp')
axs[0,0].legend();fig.tight_layout();plt.show()''')
    md(r'''### Independent audit of the archived prediction snapshot
The following comparison uses recovered **frozen predictions**, not the models refitted above. Eight held-out geometries were evaluated against archived finer-mesh CFD waveforms. The script verifies the original evidence hashes, geometry identities and coordinates, and exact equality of the two stored copies of the predictions.

It reports the aggregate waveform error and the worst individual geometry separately. Passing an aggregate 10% threshold does not mean every geometry is below 10%. The compact arrays allow numerical comparisons to be checked; recovering them does not independently recover the original solver logs or prove the chronology of the runs. This is a refit of the published architecture, not the original author's full-aircraft neural network.''')
    code('''from independent_audit_report import build_report
audit=build_report(write=False)
keys=['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error','worst_case_wave_relative_l2']
display(pd.DataFrame({label:{key:100*audit[label][key] for key in keys} for label in ['paired_coarse_mesh','finer_mesh']}).rename_axis('Error [%]'))
display(pd.DataFrame({'geometry':audit['names'],'finer_wave_error_percent':100*np.asarray(audit['finer_mesh']['per_case_wave_relative_l2'])}))
print('Coarse/finer CFD waveform difference [%]:',100*audit['coarse_to_finer_wave_relative_l2'])
print('Aggregate acceptance checks:',audit['checks'])
assert audit['passed']
display(Image(filename=str(E/'reference/independent_neural_test.png')))
print('These archived predictions remain unchanged when the notebook refits its demonstration models.')''')
    md(r'''### Use the retained portable neural checkpoint
The saved scaler, POD basis and network arrays form a separate deterministic refit. NumPy-only inference makes it usable without retraining. Its finer-CFD evaluation is retrospective because those reference labels were already available. Do not attribute the historical optimized design or the frozen-prediction audit above to this different fitted model. Inspect extrapolation errors as well as interpolation.''')
    code('''from freeze_model import FrozenSurrogate, audit as checkpoint_audit
frozen=FrozenSurrogate()
checkpoint=checkpoint_audit(write=False)
assert checkpoint['passed']
wave,drag=frozen.predict(data['parameters'][test])
print('Checkpoint identity:',checkpoint['checkpoint'])
scalar_errors=['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error','worst_case_wave_relative_l2']
error_table=pd.DataFrame({split:{key:100*values[key] for key in scalar_errors} for split,values in checkpoint['original_mesh'].items()}).T
error_table.columns=[key+' [%]' for key in error_table.columns]
display(error_table)
display(pd.Series({key+' [%]':100*checkpoint['finer_mesh'][key] for key in scalar_errors},name='Retrospective finer-CFD evaluation'))
fig,ax=plt.subplots(figsize=(9,3))
ax.plot(data['x'],data['waveforms'][test[0]],'k',label='Original CFD test case')
ax.plot(data['x'],wave[0],label='Retained checkpoint')
ax.set(xlabel='x/L',ylabel='Cp');ax.legend();plt.show()
print('This cell loads fixed weights. It does not fit or select a model.')''')
    md(r'''### Recomputed CFD with complete raw evidence
The eight known test geometries were also solved again at mesh level 2 with the retained checkpoint unchanged. Each prediction was saved before its solver run, and the complete meshes, configurations, histories and fields are archived. These repeated known geometries strengthen numerical reproducibility; they are not a new blind generalization set. The following cell independently recomputes errors from the compact distributed arrays.''')
    code('''from independent_audit_report import metrics as error_metrics
recomputed=json.loads((E/'reference/recomputed_checkpoint_audit.json').read_text())
path=E/'reference/recomputed_checkpoint_test.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==recomputed['compact_arrays_sha256']
with np.load(path,allow_pickle=False) as labels:
    assert list(labels['names'])==recomputed['names']
    predicted,predicted_drag=frozen.predict(labels['parameters'])
    np.testing.assert_allclose(predicted,labels['predicted'],rtol=1e-12,atol=1e-14)
    measured=error_metrics(labels['predicted'],labels['actual'],labels['predicted_cd'],labels['actual_cd'])
for key in scalar_errors:
    assert np.isclose(measured[key],recomputed['metrics'][key],rtol=1e-10,atol=1e-12)
display(pd.Series({key+' [%]':100*measured[key] for key in scalar_errors},name='New CFD / unchanged checkpoint'))
display(pd.DataFrame([dict(case=row['name'],iterations=row['checks']['iterations'],residual_log10=row['checks']['density_residual_log10'],residual_drop=row['checks']['residual_drop']) for row in recomputed['run_evidence']]))
assert recomputed['passed']
display(Image(filename=str(E/'reference/recomputed_checkpoint_validation.png')))
print(recomputed['scope'])''')
    md(r'''## 5. Propose a design with the surrogate
The objective is minimum peak $C_p$ at $r/L=0.5$, with pressure drag at most 2% above baseline. Volume is enforced by parameterization. The retained optimization code uses a conservative search margin and checks feasibility explicitly after CFD. A predicted optimum is only a candidate.

Inspect the search implementation in `qa/week16/learning.py`; then compare the actual proposals below. The retained candidate is not replaced by a new unverified design during notebook execution.''')
    code('''new_proposal=optimize(models[selected],write=False)
print('New refit proposal (not independently CFD-verified):',new_proposal['candidates'][0])
candidates=json.loads((E/'design_candidates.json').read_text())
print(candidates['objective'])
display(pd.DataFrame(candidates['candidates']))
print('Surrogate evaluations:',candidates['surrogate_evaluations'])
display(pd.read_csv(E/'design_comparison.csv'))
display(Image(filename=str(E/'design_shapes_signatures.png')))''')
    md(r'''## 6. Recompute and refine
The proposed shapes were meshed and solved again using SU2. The principal candidate was also evaluated on a finer mesh. Compare its achieved drag constraint and peak reduction at both resolutions. Similar peak objectives do not imply identical pressure waveforms or unique inverse geometries.''')
    code('''display(Image(filename=str(E/'cfd_fields.png')))
checks=json.loads((E/'release_check.json').read_text())
display(pd.Series(checks['scientific_checks'],name='Scientific acceptance check'))
assert checks['scientific_pass']
print('Fine mesh peak reduction [%]:',100*summary['validation']['peak_reduction_fine'])
print('Finer mesh peak reduction [%]:',100*summary['validation']['peak_reduction_finer'])''')
    md(r'''## 7. Off-design checks
The Mach 1.7 and 1.9 cases are fresh CFD runs. The surrogate was trained only at Mach 1.8. Differences between extraction radii describe different observation locations; they do not quantify atmospheric uncertainty.''')
    code('''display(pd.DataFrame(summary['mach_stress_tests']).T)
display(Image(filename=str(E/'condition_checks.png')))''')
    md(r'''## 8. Optional: regenerate actual CFD
Install the pinned SU2 and Gmsh versions, set `SU2_CFD`, and follow `cases/week16_lowboom/README.md`. Regeneration commands are deliberately not launched by Run All because they replace retained data and may require substantial CPU time.

```bash
python qa/week16/benchmark.py
python qa/week16/campaign.py
python qa/week16/learning.py
python qa/week16/verify_designs.py
python qa/week16/report.py
```

For a new design, create a unique case name and use `cfd.py --a ... --b ... --level 1.5`, followed by `analyze.py CASE_NAME`. Check convergence and recompute on level 2 before accepting a claim.

## Reflection
1. Why is fixed volume necessary?
2. Which model gives the lowest validation error, and does it also have the lowest extrapolation error?
3. Does a small waveform L2 error guarantee a correct peak?
4. Did the proposed improvement survive grid refinement?
5. What additional equations and data are needed before claiming a quieter sonic boom on the ground?

**References:** Zheng et al. (2026), DOI 10.1016/j.ast.2026.113218; SU2 8.5.0 source and governing-equation documentation; Gmsh reference manual; Taylor and Maccoll (1933); NASA/AIAA Sonic Boom Prediction Workshops. Detailed links and the independently authored theory are in the lecture source.''')
    nb.cells=cells;nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    target=ROOT/'notebooks/week16/W16_Supersonic_Shape_Optimization.ipynb'
    nbf.write(nb,target)
    if os.environ.get('FLOWMLLAB_INPROCESS')=='1':
        from execute_inprocess import execute
        nb=execute(nb)
    else:
        client=NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
    nbf.write(nb,target)
    print('Executed notebook',len(cells),'cells')
    report=json.loads((E/'release_check.json').read_text());report['notebook_executed']=True;report['notebook_execution_method']='IPython in-process' if os.environ.get('FLOWMLLAB_INPROCESS')=='1' else 'nbclient with a real Jupyter kernel';(E/'release_check.json').write_text(json.dumps(report,indent=2))


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
    figs={'4.':('cfd_fields','Actual SU2 Cp fields in the meridional plane. The dashed line marks r/L=0.5; the body is shown at its true aspect ratio.'),'5.':('numerical_verification','Mesh sensitivity and density-residual convergence. Residuals and discretization differences support distinct claims.'),'7.':('dataset_learning','Frozen geometry identities and validation-selected model predictions. Extrapolation extends the second shape parameter.'),'8.':('design_shapes_signatures','Different volume-preserving shapes and their freshly recomputed pressure signatures. All curves in the lower panel are CFD.'),'9.':('condition_checks','Pressure signatures at three distances, and changes in peak Cp and pressure drag at independently recomputed Mach numbers.')}
    reference_figures={
        '11.':('reference/seeb_geometry','NASA SEEB-ALR as-built geometry. The nose and sting must be preserved when constructing the axisymmetric computational domain.'),
        '12.':('reference/seeb_validation','Actual SU2 results against unchanged NASA experiments and NASA-hosted LAVA. The lower panels show signed errors and three-mesh sensitivity, with no fitted alignment.'),
        '15.':('reference/seeb_convergence','Actual residual and pressure-drag histories for the three NASA grids. Limiter freeze and residual acceptance are separate from experimental agreement.'),
        '17.':('reference/recomputed_checkpoint_validation','Unchanged retained neural checkpoint against eight newly recomputed CFD cases. Every raw mesh, solver configuration, field and history is retained separately.'),
        '16.':('reference/seeb_mesh','The actual exported Gmsh mesh, with equal coordinate scales in each view. The finite cap is retained; the sampling inset shows the pressure extraction location.'),
        '13.':('reference/independent_neural_test','Recovered frozen neural predictions compared with eight finer-mesh CFD signatures. These predictions are not replaced by notebook refitting.')}
    figs.update({key:value for key,value in reference_figures.items() if (E/(value[0]+'.png')).is_file()})
    summary=json.loads((E/'summary.json').read_text());val=summary['validation']
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
            name,caption=figs[key];im=Image(str(E/(name+'.png')));scale=min(width/im.imageWidth,(500 if key in reference_figures else 250)/im.imageHeight);im.drawWidth*=scale;im.drawHeight*=scale;nfig+=1
            story.append(KeepTogether([Spacer(1,6),im,Spacer(1,4),para(f'Figure {nfig}. {caption}','small')]))
        if key=='5.':
            rows=[['Numerical check','Observed difference','Criterion'],['Cone pressure',f"{100*summary['benchmark']['relative_error']:.2f}%",'<3%'],['Baseline peak, mesh',f"{100*val['baseline_peak_mesh_difference']:.2f}%",'<5%'],['Baseline drag, mesh',f"{100*val['baseline_drag_mesh_difference']:.2f}%",'<3%'],['Baseline peak, domain',f"{100*val['baseline_domain_peak_difference']:.2f}%",'<1%']]
            story.extend([Spacer(1,6),table(rows)])
        if key=='7.':
            rows=[['Model','Split','Wave L2','Peak error','Drag error']]
            for row in summary['models']['comparison']:
                rows.append([row['model'],row['split'],f"{100*row['wave_relative_l2']:.1f}%",f"{100*row['peak_mean_relative_error']:.1f}%",f"{100*row['cd_mean_relative_error']:.1f}%"])
            story.extend([Spacer(1,6),table(rows)])
        if key=='8.':
            rows=[['Quantity','45k mesh','81k mesh'],['Peak reduction',f"{100*val['peak_reduction_fine']:.2f}%",f"{100*val['peak_reduction_finer']:.2f}%"],['Drag change',f"{100*val['drag_change_fine']:.2f}%",f"{100*val['drag_change_finer']:.2f}%"]];story.extend([Spacer(1,6),table(rows)])
    def page(c,doc):
        c.setFont('FlowSerif',8);c.setFillColor(colors.HexColor('#5c6c76'));c.drawString(52,29,'FlowMLLab | Week 16 | Supersonic shape optimization');c.drawRightString(A4[0]-52,29,str(doc.page))
    target=ROOT/'lectures/week16_supersonic_shape_optimization.pdf'
    SimpleDocTemplate(str(target),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=43,bottomMargin=47,title='FlowMLLab Week 16: Supersonic Shape Optimization',author='Ehsan Roohi').build(story,onFirstPage=page,onLaterPages=page)
    print('Lecture:',target)
if __name__=='__main__':
    notebook();lecture()
