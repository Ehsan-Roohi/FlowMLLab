"""Idempotent late-course teaching updates; preserve frozen research evidence.

Canonical transformer lessons/builders own Weeks17--22. This updater owns the
curated Weeks15/16 and optional-bridge notebooks, whose existing cells remain.
"""
from pathlib import Path
import json
import nbformat as nbf
if __package__:
    from .late_course_support import bootstrap_source
else:
    from late_course_support import bootstrap_source

ROOT=Path(__file__).resolve().parents[1]


def source(cell):
    return cell.source


def replace_code(cell, new):
    if cell.source!=new:
        cell.source=new
        cell.execution_count=None
        cell.outputs=[]


def add_before(book, anchor, cells):
    additions=[]
    for new in cells:
        old=next((c for c in book.cells if c.get("id")==new.get("id")),None)
        if old is None:
            old=next((c for c in book.cells if c.cell_type==new.cell_type and c.source==new.source),None)
        if old is not None:
            old["id"]=new["id"]
            if old.cell_type=="code":replace_code(old,new.source)
            else:old.source=new.source
        else:additions.append(new)
    if additions:
        where=next((i for i,c in enumerate(book.cells) if anchor in c.source),len(book.cells))
        book.cells[where:where]=additions


def md(key,text):
    return nbf.v4.new_markdown_cell(text,id='late-'+key+'-text')


def code(key,text):
    return nbf.v4.new_code_cell(text,id='late-'+key+'-code')


def entry_setup(book, marker, paths, root_name):
    first=next(c for c in book.cells if c.cell_type=='code')
    original=first.source
    if '# Late-course checkout setup' in original:
        end=f'print("Course checkout:", {root_name})'
        if end not in original:raise ValueError("Unrecognized checkout setup; preserve it for review")
        original=original.split(end,1)[1].lstrip('\n')
        if original.startswith(f'os.chdir({root_name})'):
            original=original.split('\n',1)[1].lstrip('\n')
    prefix=bootstrap_source(marker,paths,root_name)+f'\nos.chdir({root_name})\n'
    replace_code(first,'# Late-course checkout setup\n'+prefix+'\n'+original)


def transolver_cells(folder,default_case,kind):
    return [
      md(kind+'-fields',"""## Native fields, signs and seed variation

Aim: find where a prediction changes the separated flow, rather than treating a global error as a complete description. The controls below select one **previously inspected retained case**, one physical component and the same three training seeds. They neither select a new checkpoint nor retrain a model.

The first plot fills the CFD field and overlays each seed's prediction at the same physical contour levels. Its companion shows prediction minus CFD with a shared linear signed scale. The second figure compares a native downstream row, with solids masked. Pressure is centered over the same fluid cells for every field before comparison; this removes only the arbitrary constant gauge. A negative streamwise velocity indicates reverse flow, not by itself a complete vortex.

Choose the case before judging the errors. Repeat at Re=25, 50 and 100 where that same geometry exists; preserve all three seeds and report both a gain and a failure."""),
      code(kind+'-fields',f'''import hashlib
import numpy as np
import matplotlib.pyplot as plt
from flowmllab.late_teaching_plots import field_profiles
CASE = {default_case!r}
COMPONENT = "u"  # u, v or pressure; no checkpoint selection
CUT_Y = .12
folder = ROOT / "results/{folder}"
receipt = json.loads((folder/"verification.json").read_text())
available = sorted({{r["case"] for r in receipt["records"]}})
print("Retained cases:", available)
if CASE not in available: raise ValueError("Choose a case printed above")
if COMPONENT not in ("u","v","pressure"): raise ValueError("Choose u, v or pressure")
fields=[]
for seed in (17,29,43):
    row=next(r for r in receipt["records"] if r["seed"]==seed and r["case"]==CASE)
    path=folder/f"seed_{{seed}}"/f"{{CASE}}_prediction.npz"
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=row["prediction_sha256"]: raise ValueError(f"Changed retained field: {{path}}")
    with np.load(path,allow_pickle=False) as z: fields.append({{k:z[k].copy() for k in z.files}})
ref=fields[0]; shape=tuple(map(int,ref["shape"]))
mask=ref["mask"].reshape(shape).astype(bool)
xy=ref["coordinates"].reshape(*shape,2)
channel={{"u":0,"v":1,"pressure":2}}[COMPONENT]
truth=ref["truth"].reshape(*shape,3)[:,:,channel].astype(float)
predictions={{}}
for seed,f in zip((17,29,43),fields):
    np.testing.assert_array_equal(f["mask"],ref["mask"])
    np.testing.assert_allclose(f["truth"],ref["truth"],rtol=0,atol=0)
    value=f["prediction"].reshape(*shape,3)[:,:,channel].astype(float)
    if COMPONENT=="pressure":value=value-value[mask].mean()
    predictions[f"Transolver seed {{seed}}"]=value
if COMPONENT=="pressure":truth=truth-truth[mask].mean()
figures,diagnostics=field_profiles(xy[0,:,0],xy[:,0,1],truth,predictions,
    mask=mask,quantity=f"{{COMPONENT}} (stored units)",case=f"{{CASE}}; retained fields",
    cut_y=CUT_Y)
display(pd.DataFrame(diagnostics["metrics"]))
print("Native profile row:",diagnostics["native_profile_y"],"; no checkpoint replay in this cell.")
plt.show()
'''),
      md(kind+'-experiments',"""## A controlled student investigation

1. **Locate a missed feature.** At a fixed geometry/Re and seed, report the native coordinate of the largest signed streamwise-velocity discrepancy, and compare the reverse-flow region with CFD. State whether the main error is strength, location or missing connected regions.
2. **Check repeatability.** Keep that case fixed and compare all three seeds. Report their numerical range, and explain why three runs do not establish a universal architecture ranking.
3. **Change the quantity.** Repeat the same-case comparison for pressure after common fluid-mean centering. Explain why a change in the pressure zero cannot correct the velocity error.
4. **Compare Reynolds regimes.** Use every available Re for the chosen geometry. Submit the matched field/signed-profile figures and a compact table with case, seed, metric definition and native grid. Do not use these retrospective cases to claim a new blind test.

Full checkpoint replay and campaign training are separate research procedures. They require the original dependencies, frozen input/source identities and an isolated output workspace. Default Run All verifies the displayed saved fields and recomputes their diagnostics; it does not rerun those campaigns.""")
    ]


def refresh_book(book,name):
    for c in book.cells:
        if c.cell_type=='markdown':
            c.source=c.source.replace('The split map below is deliberately **not a bar chart**. Each square is one accepted',
                                      'Each square in the split map below is one accepted')
            c.source=c.source.replace('The waveform itself is the physical object of interest, so line plots are used rather than bar charts.',
                                      'Compare pressure at matched physical coordinates: peak strength, shock location and expansion shape can change independently. Signed prediction errors distinguish a shifted wave from a uniformly biased pressure level.')
    if name=='W15_Complete_Geometry_Generalization.ipynb':
        entry_setup(book,'results/week15_postaudit',
                    ['flowmllab','qa','notebooks/week15','results/week15_postaudit','results/family_transolver'],
                    'REPO')
        add_before(book,'## 1. Reproducible setup',[
          md('w15-route',"""## Choose the laboratory route

| Route | What runs on a CPU | What you should produce |
|---|---|---|
| Core, this notebook | Geometry/SDF, branch-trunk and Fourier demonstrations; a genuine bounded CPU POD-DeepONet fit; accessible retained tables/figures | Change depth, optimizer, rank or loss weights; compare unseen manufactured geometries with exact reference and linear interpolation, then interpret the separate CFD evidence |
| Original Transolver companions | Hash-checked retained fields for the single-step or double-step protocol | Same-case CFD/prediction/signed-error plots across three seeds |
| Larger networks | Arithmetic on retained capacity summaries and linked physical fields | A matched small-versus-large comparison; keep protocols and missing controls separate |
| External archive extension | Optional independent recomputation of 342 retained predictions when the complete archive is present | A reproducibility audit, not newly trained or newly generated CFD |

Start with setup, the two mechanism demonstrations, the **Local CPU laboratory**, and its four controlled experiments. Then read Sections 5–6 and 13–17 for the retained CFD comparison. Sections 3–4 and 7–12 are clearly marked **optional archive extension**; their code runs only when the external research archive is present. Section 18 separates archive exercises from interpretation questions. The core path is complete without that archive.

The setup cell locates an existing checkout or obtains the dependencies in Colab. It reports archive availability once; unavailable archive checks do not count as passes. The seven-row galleries and their tables remain retained, retrospective research evidence.""")])
        if __package__:
            from .week15_operator_section import cells as operator_cells
        else:
            from week15_operator_section import cells as operator_cells
        add_before(book,'## 3. Split reconstruction',operator_cells())
        archive_sections=(3,4,7,8,9,10,11,12)
        for c in book.cells:
            if c.cell_type=='code':
                new=c.source.replace("    print('SKIPPED optional external-archive investigation; continue the local core.')",
                                     '    pass  # Optional archive extension; availability is reported once in setup.')
                new=new.replace("Core route ready. Optional 342-field archive checks will be SKIPPED, not passed.",
                                "Core CPU route ready. Optional archive extension is unavailable and will not execute; no archive checks are counted as passed.")
                replace_code(c,new)
            else:
                for number in archive_sections:
                    prefix=f'## {number}. '
                    if c.source.startswith(prefix) and 'Optional archive extension' not in c.source.splitlines()[0]:
                        c.source=c.source.replace(prefix,prefix+'Optional archive extension — ',1)
                if c.get('id')=='w15-complete-03':
                    c.source="""## Core route: understand, train, then evaluate

The next two cells isolate geometry encoding, branch–trunk contraction and Fourier filtering. Their weights are chosen for explanation; they are not fitted CFD surrogates. Next, the **Local CPU laboratory** trains a POD-DeepONet and compares two held-out manufactured geometries with exact fields and linear interpolation. Complete its four controlled experiments before moving to the retained research comparisons. The lecture's local-CPU section defines every variable and the train/validation/test contract."""
                if c.get('id')=='w15-complete-37' and 'Exercises 1–5' not in c.source:
                    c.source=c.source.replace('## 18. Student exercises','## 18. Research-extension exercises\n\nExercises 1–5 require the optional archive and its original case identities. Questions 6–7 can be answered from the retained tables and figures. The fully runnable CPU experiments are in the Local CPU laboratory above.')
                if c.source.startswith('## Original Transolver:'):
                    c.source="""## Original Transolver: same double-step split

The [double-step companion](W15_Original_Transolver_Double_Step.ipynb) verifies the saved prediction-file hashes and plots native CFD/prediction/signed-error fields for three seeds. Change the case, Reynolds number and quantity controls to inspect failure modes. Full checkpoint replay and campaign training are separate research procedures; ordinary Run All does not rerun them."""
    elif name=='W15_Geometry_Operators_Step_Audit.ipynb':
        entry_setup(book,'results/step_operator_audit/wake_predictions.tgz',
            ['flowmllab','qa','notebooks/week15','results/step_operator_audit',
             'results/step_architecture_v5','results/step_geometry_generalization',
             'results/week15_ordinary_deeponet'], 'REPO_ROOT')
        for c in book.cells:
            if c.cell_type=='code':
                s=c.source.replace("Upload wake_predictions.tgz next to the notebook, then rerun.",
                    "This historical audit requires the complete lesson checkout; rerun the setup cell.")
                s=s.replace("Upload dataset.npz next to the notebook, then rerun.",
                    "The historical CFD dataset is missing from the lesson checkout.")
                replace_code(c,s)
        book.cells[0].source=book.cells[0].source.replace(
            'In Colab, clone the repository and change into it before running this notebook.',
            'The setup cell obtains these lesson directories automatically in Colab.')
    elif name=='W15_Original_Transolver_Double_Step.ipynb':
        entry_setup(book,'results/family_transolver/comparison.csv',
                    ['flowmllab','qa','notebooks/week15','results/family_transolver'], 'ROOT')
        for c in book.cells:
            if c.cell_type=='code' and 'RUN_TRAINING = False' in c.source:
                replace_code(c,'''RUN_TRAINING = False
if RUN_TRAINING:
    raise RuntimeError("This notebook is a retained-evidence lab. Campaign training must use an isolated output workspace; do not overwrite results/family_transolver.")
print("Default route: current saved-field identity checks and physical diagnostics; no training.")
''')
            elif c.cell_type=='code' and "subprocess.run([sys.executable,'qa/verify_family_transolver.py']" in c.source:
                replace_code(c,'''import json
from IPython.display import display
retained_verification=json.loads((ROOT/"results/family_transolver/verification.json").read_text())
print("Historical replay:",retained_verification["checks"])
print("Current displayed-field hashes are checked in the physical comparison below; full replay is not run here.")
''')
        add_before(book,'## Same-case comparison',[
            md('double-aim',"""## Aim and model input

Learn to evaluate a geometry-conditioned field operator on a **whole withheld geometry**, while separating checkpoint replay, training and physical assessment. The original Transolver receives spatial coordinates together with geometry/flow features and groups query information through learned physical-state attention. Changing the query set can affect that representation; it is not the same pointwise branch-trunk contraction introduced in the main notebook.

The retained double-step split has 100 training, 8 validation and 19 retrospectively inspected test cases. Validation selected checkpoints; the plots below do not select them again. The same 19,200-update cap does not equalize training FLOPs or all tuning choices across architecture families.""")])
        add_before(book,'__never_a_real_anchor__',transolver_cells('family_transolver','g049_Re100_medium','double'))
    elif name=='W15_Original_Transolver_Step.ipynb':
        first=next(c for c in book.cells if c.cell_type=='code')
        if '# Late-course checkout setup' not in first.source:
            replace_code(first,'''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/"results/step_transolver/comparison.csv").exists())
OUT=ROOT/"results/step_transolver"
retained_verification=json.loads((OUT/"verification.json").read_text())
print("Retained replay receipt:",retained_verification["checks"])
print("Default Run All reads retained data; it does not rewrite verification or figure files.")
''')
        entry_setup(book,'results/step_transolver/comparison.csv',
                    ['flowmllab','qa','notebooks/week15','results/step_transolver'], 'ROOT')
        book.cells[0].source=book.cells[0].source.replace(
            "Execution note: cells were checked sequentially in-process with IPython; a separate Jupyter kernel could not start because this execution environment prohibits socket binding.",
            "The default CPU route reads retained results. Current checks of the displayed arrays are separate from the historical checkpoint-replay receipt.")
        add_before(book,'__never_a_real_anchor__',transolver_cells('step_transolver','g011_Re50_medium','single'))
    elif name=='W15_Larger_Networks_Analysis.ipynb':
        entry_setup(book,'results/week15_capacity_study/native_capacity_results.csv',
                    ['notebooks/week15','weeks/week15','qa','results/week15_capacity_study',
                     'results/family_transolver','results/step_transolver'], 'root')
    elif name=='W16_Supersonic_Shape_Optimization.ipynb':
        first=next(c for c in book.cells if c.cell_type=='code')
        if '# Late-course checkout setup' not in first.source:
            start=first.source.index('ROOT = next(')
            end=first.source.index('E = ROOT')
            prefix=bootstrap_source('results/week16_lowboom',
                ['flowmllab','qa/week16','notebooks/week16','results/week16_lowboom'])
            replace_code(first,'# Late-course checkout setup\n'+prefix+'\n'+first.source[:start]+first.source[end:])
        entry_setup(book,'results/week16_lowboom',
                    ['flowmllab','qa/week16','notebooks/week16','results/week16_lowboom'],'ROOT')
        add_before(book,'## 1. From supersonic',[
            md('w16-map',"""## Laboratory map and execution boundary

| Sections | Aim and algorithm | Evidence produced now |
|---|---|---|
| 1–4 | Normalize pressure, preserve body volume and solve the Taylor–Maccoll ODE | Fresh scalar/geometry/ODE calculations; no SU2 run |
| 5–9 | Freeze shape-level splits, fit training-only POD and compare ridge, GP and MLP regression | Fresh small CPU fits, validation selection and physical waveform comparisons |
| 10 | Replay the separately retained neural checkpoint | Current inference checked against recorded metrics |
| 11–13 | Explore a surrogate proposal and assess the previously evaluated design | New surrogate grid search; retained independent CFD and mesh/off-design checks |

All displayed CFD was computed before this notebook run. A new shape proposed by your fitted surrogate is **not** verified by the old design-CFD files. Its acceptance would require its own separate CFD calculation.

Keep the test/extrapolation cases out of parameter choice. For a fair capacity exercise, hold the training cases, POD rank, scaling, seed and optimizer fixed, vary only one MLP control below, select on validation, then report the predetermined test and extrapolation cases."""),
            code('w16-controls',"""POD_RANK = 12
MLP_WIDTH = 32
MLP_DEPTH = 2
MLP_MAX_ITER = 3000
MODEL_SEED = 16
# These preserve the lesson's original defaults. Keep the saved checkpoint separate.
assert 1 <= POD_RANK <= 23 and MLP_WIDTH > 0 and MLP_DEPTH > 0 and MLP_MAX_ITER > 0
""")])
        for c in book.cells:
            if c.cell_type=='code':
                s=c.source.replace('K = min(12, len(S))','K = min(POD_RANK, len(S))')
                s=s.replace('hidden_layer_sizes=(32,32)','hidden_layer_sizes=(MLP_WIDTH,)*MLP_DEPTH')
                s=s.replace('max_iter=3000','max_iter=MLP_MAX_ITER').replace('random_state=16','random_state=MODEL_SEED')
                s=s.replace('Fresh CFD verification: pressure-signature reshaping','Retained independent CFD: pressure-signature reshaping')
                replace_code(c,s)
            else:
                c.source=c.source.replace('## 12. Fresh CFD verification of the retained optimized design',
                    '## 12. Retained independent CFD verification of the optimized design')
        add_before(book,'## 10. Retained audited neural model',[
          md('w16-all-models',"""### Compare all three fitted methods on the same validation shape

A table can conceal a displaced shock or an error in the negative-pressure part of the signature. The following predeclared validation case is used for diagnosis; it is not chosen because a model looks good. The upper plot retains signed Cp, and the lower plot shows each method minus the same CFD waveform. Change MODEL_CASE to another validation case while developing a model; reserve the existing test cases for the final report."""),
          code('w16-all-models',"""validation_ids = np.flatnonzero(data["splits"] == "validation")
MODEL_CASE = int(validation_ids[0])
assert data["splits"][MODEL_CASE] == "validation"
fig,axs = plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
truth_wave = Y[MODEL_CASE]
axs[0].plot(data["x"],truth_wave,color="black",lw=2,label="SU2 CFD")
for (name,model),style in zip(models.items(),("--","-.",":")):
    wave,_=predict(model,X[MODEL_CASE]);wave=wave[0]
    axs[0].plot(data["x"],wave,lw=1.6,ls=style,label=name)
    axs[1].plot(data["x"],wave-truth_wave,lw=1.6,ls=style,label=name)
axs[0].set(ylabel="Cp",title=f"Validation shape {data['names'][MODEL_CASE]}; M=1.8")
axs[1].set(xlabel="x/L",ylabel="surrogate minus CFD, Cp")
axs[1].axhline(0,color=".5",lw=.7)
axs[0].legend(loc="upper center",bbox_to_anchor=(.5,1.45),ncol=4,frameon=False)
for ax in axs:ax.grid(alpha=.2)
plt.show()
print("Fresh CPU models; separately retained neural-model evidence follows.")
""")])
        for c in book.cells:
            if c.cell_type!='code':continue
            new=c.source
            if c.get('id')=='w16-07':
                new=new.replace('ax.legend(ncol=3)','ax.legend(ncol=3,loc="lower center",bbox_to_anchor=(.5,1.45),frameon=False)')
            elif c.get('id')=='w16-12':
                new=new.replace('ax.legend()','ax.legend(loc="lower center",bbox_to_anchor=(.5,1.15),ncol=4,frameon=False)')
            elif c.get('id')=='w16-25':
                new=new.replace('c=peak_g,s=12)','c=peak_g,s=12,cmap="jet")')
                new=new.replace('marker="*",s=180,label=', 'marker="*",s=180,facecolor="white",edgecolor="black",label=')
                new=new.replace('ax.legend()','ax.legend(loc="lower center",bbox_to_anchor=(.5,1.15),frameon=False)')
            elif c.get('id')=='w16-30':
                new=new.replace('"o-",label="peak reduction"','"o",label="peak reduction"')
                new=new.replace('"s-",label="magnitude of drag reduction"','"s",label="magnitude of drag reduction"')
            replace_code(c,new)
    elif name in ('Geometry_Time_Operator_Bridge.ipynb','Geometry_Time_Bridge_Solutions.ipynb'):
        entry_setup(book,'flowmllab/geometry_time_bridge.py',
                    ['flowmllab','notebooks/optional_geometry_time_bridge'], 'root')
        add_before(book,'## Your four exercises',[
          md('bridge-feedback-plot',"""## See feedback and phase errors before implementing them

These manufactured signals isolate two mechanisms. In the first panel, teacher forcing receives a new observed state at each step; autonomous rollout must feed back its own prediction. Their difference is an input-information difference, not proof that one algorithm is intrinsically more accurate. In the second example, an amplitude error and a wrapped phase offset are both present. Compare the signed signal error as well as the fitted phase; a small phase statistic alone can conceal a poor amplitude."""),
          code('bridge-feedback-plot',"""fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
steps=np.arange(1,len(auto)+1)
ax.plot(steps,observed[2:,0],"ko-",label="manufactured observations")
ax.plot(steps,tf[:,0],"s--",label="teacher forced")
ax.plot(steps,auto[:,0],"d-.",label="autonomous")
ax.set(xlabel="predicted step",ylabel="manufactured state")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.3),ncol=3,frameon=False)
ax.grid(alpha=.2)
plt.show()
angle=2*np.pi*.2*(t-t[0])
reference=2*np.sin(angle+np.deg2rad(179))
prediction=np.sin(angle+np.deg2rad(-179))
fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,constrained_layout=True)
axes[0].plot(t,reference,color="black",label="manufactured reference")
axes[0].plot(t,prediction,"--",label="half amplitude, wrapped phase offset")
axes[1].plot(t,prediction-reference)
axes[1].axhline(0,color=".5",lw=.7)
axes[0].set(ylabel="signal")
axes[1].set(xlabel="time",ylabel="prediction minus reference")
axes[0].legend(loc="upper center",bbox_to_anchor=(.5,1.3),ncol=2,frameon=False)
for ax in axes:ax.grid(alpha=.2)
plt.show()
""")])
    book.metadata.setdefault('flowmllab',{})['teaching_refresh_scope']='late-course CPU instruction; retained research evidence preserved'
    nbf.validate(book)
    return book


def refresh_one(path):
    book=refresh_book(nbf.read(path,as_version=4),path.name)
    path.write_text(nbf.writes(book),encoding='utf-8',newline='\n')


def refresh_notebooks(paths=None):
    if paths is None:
        paths=[p for w in ('week15','week16','optional_geometry_time_bridge')
               for p in (ROOT/'notebooks'/w).glob('*.ipynb')]
    for p in paths:refresh_one(Path(p))


if __name__=='__main__':refresh_notebooks()
