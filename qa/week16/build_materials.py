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

The main student dataset contains 44 newly computed SU2 8.0.1 cases with numerical and full-field physical checks. A separately identified neural model uses only its 24 training cases. Historical SU2 8.5.0 data and weights remain available for an audit lesson: their small residuals did not reveal local total-enthalpy defects. You will distinguish analytical verification, experimental CFD validation, neural prediction accuracy and direct design checks.

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
historical_data = np.load(E/'dataset.npz')
data = np.load(E/'reference/clean_dataset_v801.npz')
clean_campaign = json.loads((E/'reference/clean_campaign_audit.json').read_text())
assert clean_campaign['passed'] and clean_campaign['cases']==44
assert hashlib.sha256((E/'reference/clean_dataset_v801.npz').read_bytes()).hexdigest()==clean_campaign['dataset_sha256']
summary = json.loads((E/'summary.json').read_text())
assert hashlib.sha256((E/'dataset.npz').read_bytes()).hexdigest() == summary['dataset_sha256']
print('Clean student dataset verified:', clean_campaign['dataset_sha256'])
print('CFD cases:', len(data['parameters']))
print('Clean solver version:', clean_campaign['solver_version'])
print('Clean solver binary SHA-256:', clean_campaign['identity']['solver_binary_sha256'])
display(pd.Series(json.loads((E/'software_versions.json').read_text()), name='Historical SU2 8.5.0 generation environment'))''')
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
    code('''print('Historical SU2 8.5.0 numerical checks; physical acceptance is assessed separately below.')
display(pd.Series(summary['benchmark']))
display(pd.Series(summary['validation']))
cases = pd.read_csv(E/'case_metrics.csv')
assert len(cases)==44 and cases['converged'].all()
display(cases[['name','cells','iterations','residual_drop','drag_tail_relative_range']].head(8))
display(Image(filename=str(E/'numerical_verification.png')))''')
    md(r'''### Recompute the Taylor-Maccoll reference
The conical similarity equations are integrated from an oblique shock to the cone wall. A scalar root search adjusts the shock angle until the polar velocity vanishes at the 7-degree wall. This ODE calculation is independent of the SU2 finite-volume discretization. The SU2 8.0.1 comparison retains all three mesh levels, including any coarse-grid failure. It verifies cone wall pressure, not an off-body boom waveform.''')
    code('''from benchmark import cone_exact
exact=cone_exact(mach=1.8,theta_deg=7.)
display(pd.Series(exact,name='Independently integrated cone solution'))
assert abs(exact['cp']-summary['benchmark']['exact']['cp'])<1e-8
cone=json.loads((E/'reference/cone_refinement_v801.json').read_text())
display(pd.DataFrame([dict(level=r['level'],cells=r['mesh']['cells'],Cp=r['analytical_comparison']['cfd_cp_mean'],Cp_error_percent=100*r['analytical_comparison']['relative_error'],physical_pass=r['physical']['passed'],converged=r['numerical']['passed']) for r in cone['levels']]))
print('Last two mean-Cp change [%]:',100*cone['last_two_cp_relative_change'])
print('Refined-family gates:',cone['checks'])
assert cone['passed']
print('Coarse-grid Cp failure, if present, is retained; acceptance uses the declared refined-family criteria.')''')
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
    md(r'''### Primary retained neural model: clean CFD labels
All 44 original geometries were recomputed with official SU2 8.0.1 and passed the declared numerical and full-field physical checks. A new model was fitted once on the 24 training cases, with 12 full-SVD POD modes and the fixed 32–32 tanh architecture. The following cell reloads its portable weights without fitting. Validation, test and extrapolation cases do not enter this fit. Previously examined geometries and existing finer solutions make this a retrospective computational check, not a new blind experiment.''')
    code('''from clean_model_v801 import audit as clean_model_audit
from freeze_model import FrozenSurrogate
clean_audit=clean_model_audit(write=False)
assert clean_audit['passed']
clean_model=FrozenSurrogate(E/'reference/clean_model_v801.npz')
keys=['wave_relative_l2','peak_mean_relative_error','drag_mean_relative_error','worst_case_wave_relative_l2']
display(pd.DataFrame({split:{k:100*v[k] for k in keys} for split,v in clean_audit['same_mesh'].items()}).T)
display(pd.Series({k:100*clean_audit['finer_mesh'][k] for k in keys},name='Clean neural model versus finer CFD [%]'))
print('Checkpoint SHA-256:',clean_audit['checkpoint_sha256'])
print('Training warnings:',clean_audit['training_log']['warnings'])
with np.load(E/'reference/weakwall_checkpoint_test.npz') as fine:
    np.testing.assert_array_equal(fine['parameters'],data['parameters'][test])
    predicted,_=clean_model.predict(fine['parameters'])
    fig,axs=plt.subplots(4,2,figsize=(11,12))
    for j,ax in enumerate(axs.flat):
        ax.plot(data['x'],data['waveforms'][test[j]],label='Clean training-resolution CFD')
        ax.plot(fine['x'],fine['actual'][j],label='Finer CFD')
        ax.plot(fine['x'],predicted[j],'--',label='New retained neural model')
        ax.set(title=str(fine['names'][j]),xlabel='x/L',ylabel='Cp')
    axs.flat[0].legend(fontsize=7);fig.tight_layout();plt.show()
print('The historical optimizer candidate below was proposed by a different model.')''')
    md(r'''### Independent audit of the archived prediction snapshot
The following comparison uses recovered **frozen predictions**, not the models refitted above. Eight held-out geometries were evaluated against archived finer-mesh CFD waveforms. The script verifies the original evidence hashes, geometry identities and coordinates, and exact equality of the two stored copies of the predictions.

It reports the aggregate waveform error and the worst individual geometry separately. Passing an aggregate 10% threshold does not mean every geometry is below 10%. The compact arrays allow numerical comparisons to be checked; recovering them does not independently recover the original solver logs or prove the chronology of the runs. This is a refit of this teaching repository's architecture, not the original author's full-aircraft neural network.''')
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
ax.plot(historical_data['x'],historical_data['waveforms'][test[0]],'k',label='Historical SU2 8.5.0 test case')
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
    md(r'''### The physical audit changes what the numerical error means
Small residuals and accurate neural predictions do not establish physically reliable CFD labels. The archived 8.5.0 runs fail the full-field maximum total-enthalpy criterion at pointed body/axis nodes. The original training data remain unchanged. An official SU2 8.0.1 comparison uses the exact same eight meshes and configurations with the same retained neural weights; its full-field checks are reported separately. This is a controlled solver-version comparison at known geometries, not a new blind generalization test. See [the pointed-body investigation](POINTED_BODY_PHYSICS_AUDIT.md).''')
    code('''historical_physics=json.loads((E/'reference/neural_physical_plausibility.json').read_text())
assert not historical_physics['passed']
display(pd.DataFrame([dict(case=r['case'],max_h0_error_percent=100*max(f['max_h0_relative_deviation'] for f in r['fields']),physical_pass=r['passed']) for r in historical_physics['runs']]))
accepted=json.loads((E/'reference/weakwall_checkpoint_audit.json').read_text())
path=E/'reference/weakwall_checkpoint_test.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==accepted['compact_arrays_sha256']
with np.load(path,allow_pickle=False) as labels:
    assert list(labels['names'])==accepted['names']
    assert np.array_equal(labels['parameters'],data['parameters'][test])
    assert np.array_equal(labels['x'],data['x'])
    pw,pc=frozen.predict(labels['parameters'])
    np.testing.assert_allclose(pw,labels['predicted'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(pc,labels['predicted_cd'],rtol=1e-12,atol=1e-14)
    accepted_metrics=error_metrics(pw,labels['actual'],pc,labels['actual_cd'])
for key in scalar_errors:
    assert np.isclose(accepted_metrics[key],accepted['metrics'][key],rtol=1e-10,atol=1e-12)
display(pd.Series({key+' [%]':100*accepted_metrics[key] for key in scalar_errors},name='Unchanged checkpoint versus SU2 8.0.1'))
display(pd.DataFrame([dict(case=r['name'],max_h0_error_percent=100*max(f['max_h0_relative_deviation'] for f in r['checks']['physical']['fields']),max_density_over_bound=max(f['max_density_over_stagnation_bound'] for f in r['checks']['physical']['fields']),passed=r['checks']['passed']) for r in accepted['run_evidence']]))
assert accepted['passed'] and accepted['physical_and_convergence_pass']
display(Image(filename=str(E/'reference/weakwall_checkpoint_validation.png')))
print('Worst individual error remains above 10%; the declared neural criteria are aggregate criteria.')''')
    md(r'''## 5. Propose a design with the surrogate
The objective is minimum peak $C_p$ at $r/L=0.5$, with pressure drag at most 2% above baseline. Volume is enforced by parameterization. The retained optimization code uses a conservative search margin and checks feasibility explicitly after CFD. A predicted optimum is only a candidate.

Inspect the search implementation in `qa/week16/learning.py`; then compare the actual proposals below. The retained candidate is not replaced by a new unverified design during notebook execution.''')
    code('''baseline_evidence=json.loads((E/'reference/weakwall_design_audit.json').read_text())['runs'][0]
new_proposal=optimize(models[selected],write=False,baseline={'peak_cp':baseline_evidence['peak_cp'],'cd_pressure':baseline_evidence['cd_pressure']})
print('New refit proposal (not independently CFD-verified):',new_proposal['candidates'][0])
candidates=json.loads((E/'design_candidates.json').read_text())
print(candidates['objective'])
display(pd.DataFrame(candidates['candidates']))
print('Surrogate evaluations:',candidates['surrogate_evaluations'])
print('The following fresh CFD section evaluates these retained geometries directly; the portable checkpoint did not generate the historical optimizer proposal.')''')
    md(r'''## 6. Recompute and refine the retained candidate
Ten new SU2 8.0.1 runs check the original candidate geometries without changing the training data or retraining. The baseline and optimized candidate are solved at two mesh levels. Alternative shapes and Mach 1.7/1.9 cases have their own convergence and physical checks. The candidate must reduce the near-field peak by more than 5% and satisfy the actual pressure-drag constraint at both design-point meshes. Mesh differences are observed sensitivity, not a formal GCI.''')
    code('''design=json.loads((E/'reference/weakwall_design_audit.json').read_text())
path=E/'reference/weakwall_design_test.npz'
assert hashlib.sha256(path.read_bytes()).hexdigest()==design['compact_arrays_sha256']
with np.load(path,allow_pickle=False) as evidence:
    assert list(evidence['names'])==[r['specification']['name'] for r in design['runs']]
    for i,row in enumerate(design['runs']):
        assert np.isclose(evidence['waveforms'][i].max(),row['peak_cp'])
        assert np.isclose(evidence['cd_pressure'][i],row['cd_pressure'])
    for pair in design['design_pairs']:
        i,j=pair['baseline_index'],pair['optimized_index']
        assert np.isclose(1-evidence['waveforms'][j].max()/evidence['waveforms'][i].max(),pair['peak_reduction'])
        assert np.isclose(evidence['cd_pressure'][j]/evidence['cd_pressure'][i],pair['drag_ratio'])
display(pd.DataFrame([dict(level=p['level'],peak_reduction_percent=100*p['peak_reduction'],drag_change_percent=100*p['drag_change'],drag_ratio=p['drag_ratio']) for p in design['design_pairs']]))
display(pd.DataFrame([dict(design=m['design'],peak_mesh_change_percent=100*m['peak_relative_change'],drag_mesh_change_percent=100*m['drag_relative_change'],wave_mesh_change_percent=100*m['wave_relative_l2_change']) for m in design['mesh_sensitivity']]))
display(pd.Series(design['checks'],name='Fresh design acceptance'))
assert design['passed']
display(Image(filename=str(E/'reference/weakwall_design_validation.png')))
print('The waveform mesh difference is also reported; the declared mesh gates apply to peak and drag.')''')
    md(r'''## 7. Off-design checks
The Mach 1.7 and 1.9 results below are new SU2 8.0.1 calculations of the fixed baseline and retained optimized geometry. The surrogate was trained only at Mach 1.8. These off-design values are CFD results, not surrogate predictions or atmospheric uncertainty estimates. Off-design and alternative runs pass physical and convergence checks; their performance is reported without assuming design-point acceptance.''')
    code('''display(pd.DataFrame([dict(mach=p['mach'],peak_reduction_percent=100*p['peak_reduction'],drag_change_percent=100*p['drag_change']) for p in design['offdesign_pairs']]))
display(pd.DataFrame([dict(case=r['specification']['name'],mach=r['specification']['mach'],cells=r['cells'],iterations=r['checks']['iterations'],max_h0_error_percent=100*max(f['max_h0_relative_deviation'] for f in r['checks']['physical']['fields']),passed=r['checks']['passed']) for r in design['runs']]))''')
    md(r'''## 8. Optional: regenerate actual CFD
Install the pinned SU2 and Gmsh versions, set `SU2_CFD`, and follow `cases/week16_lowboom/README.md`. Regeneration commands are deliberately not launched by Run All because they require substantial CPU time. The new generators refuse to overwrite existing run folders or trained weights.

```bash
python qa/week16/install_su2_801.py
# Run in a new checkout/output location to preserve published evidence.
# For each batch index 0 through 10:
python qa/week16/clean_campaign_v801.py --batch 0
# After all eleven batches finish:
python qa/week16/clean_campaign_v801.py --assemble
python qa/week16/clean_model_v801.py --train
python qa/week16/clean_model_v801.py --check-only
```

For a new design, create a unique case name and use `cfd.py --a ... --b ... --level 1.5`, followed by `analyze.py CASE_NAME`. Check convergence and recompute on level 2 before accepting a claim.

## Reflection
1. Why is fixed volume necessary?
2. Which model gives the lowest validation error, and does it also have the lowest extrapolation error?
3. Does a small waveform L2 error guarantee a correct peak?
4. Did the proposed improvement survive grid refinement?
5. What additional equations and data are needed before claiming a quieter sonic boom on the ground?

**References:** Zheng et al. (2026), DOI 10.1016/j.ast.2026.113218; SU2 8.0.1 and 8.5.0 official sources and governing-equation documentation; Gmsh reference manual; Taylor and Maccoll (1933); NASA/AIAA Sonic Boom Prediction Workshops. Detailed links and the independently authored theory are in the lecture source.''')
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
    figs={'4.':('reference/clean_cfd_fields','Actual accepted SU2 8.0.1 pressure fields for the baseline and retained optimized geometry. Both panels use identical Cp limits, exported cell connectivity and equal physical coordinate scales.'),'5.':('numerical_verification','Historical SU2 8.5.0 mesh and residual comparisons. These numerical checks did not detect the later identified tip enthalpy defect.'),'7.':('reference/clean_model_learning','Clean SU2 8.0.1 geometry split and new retained neural model: aggregate split errors, including poor extrapolation. The waveform compares the actual clean test CFD with the saved model.'),'8.':('reference/weakwall_design_validation','New SU2 8.0.1 CFD at the original candidate geometries, both design meshes and two off-design Mach numbers. All plotted cases pass full-field physical and convergence checks.'),'9.':('condition_checks','Historical SU2 8.5.0 observation-line and Mach comparisons. Accepted new 8.0.1 off-design results are reported separately.')}
    reference_figures={
        '11.':('reference/seeb_geometry','NASA SEEB-ALR as-built geometry. The nose and sting must be preserved when constructing the axisymmetric computational domain.'),
        '12.':('reference/seeb_validation','Actual SU2 results against unchanged NASA experiments and NASA-hosted LAVA. The lower panels show signed errors and three-mesh sensitivity, with no fitted alignment.'),
        '15.':('reference/seeb_convergence','Actual residual and pressure-drag histories for the three NASA grids. Limiter freeze and residual acceptance are separate from experimental agreement.'),
        '17.':('reference/weakwall_checkpoint_validation','Unchanged retained neural checkpoint against eight physically checked SU2 8.0.1 calculations on the same archived meshes and configurations. The older 8.5.0 physical audit remains failed.'),
        '16.':('reference/seeb_mesh','The actual exported Gmsh mesh, with equal coordinate scales in each view. The finite cap is retained; the sampling inset shows the pressure extraction location.'),
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
