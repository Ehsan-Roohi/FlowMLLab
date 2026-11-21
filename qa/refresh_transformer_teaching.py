"""Add concrete experiment maps and native physical diagnostics to canonical lessons."""
from pathlib import Path
import json
from late_course_support import bootstrap_source, TRANSFORMER_PATHS
ROOT=Path(__file__).resolve().parents[1]

MAPS={
17:("Attention algebra -> physical sensor arrangement",
"""| Notebook stage | What enters the calculation | What to inspect |
|---|---|---|
| Random Q/K/V demonstration | Sixteen v/U observations and their x/D,y/D coordinates; random fixed projections | Weight rows sum to one; record tensor shapes and the temperature effect |
| Geometry versus content kernels | The same coordinates and three flow frames | A changed sensor value changes content attention; moving a coordinate changes the physical input |
| Retained trained heads | Saved Sensor-set weights and saved normalization, clean Re110 frames | Head differences are conditional weights, not causal influence |
| Tasks 1–4 | Your stable softmax, permutation, temperature and masked-attention functions | Algebraic tests plus a controlled perturbation report |""",
"""Start with the default 16 sensors. Keep observations, coordinates, projection seed and head width fixed when changing temperature. The source POD basis used to place sensors is fitted on Re90 frames 0–159. The displayed Re110 frame is an already inspected mechanism example, not a new held-out accuracy claim. A dot on the wake plot is an observation location; the rectangular ROI boundary is not a wall."""),
18:("Causal masks -> honest next-position supervision",
"""| Notebook stage | Input/output | Required observation |
|---|---|---|
| Character-model fit | Integer context IDs to logits at every position | Training loss and whole-validation-document NLL; vocabulary excludes validation-only characters |
| POD prefix perturbation | Four continuous rank-eight state vectors to four shifted targets | Earlier masked outputs stay fixed when future inputs change |
| Tasks 1–2 | Stride-one windows and log-softmax likelihood | Check one batch and a hand-computed NLL before using perplexity |
| Tasks 3–4 | Prefix invariance and per-position likelihood | A final-position success must not conceal errors at earlier positions |""",
"""Change LM_STEPS, MODEL_WIDTH, CONTEXT or SAMPLE_TEMPERATURE one at a time. Width must remain divisible by four attention heads. Temperature changes sampling rather than fitted weights. The 160-update default is a small CPU teaching fit; generated prose is unvalidated. The POD masked/unmasked curves below come from the same randomly initialized decoder and the same perturbed input, so the experiment isolates information access rather than comparing trained forecast accuracy."""),
19:("Representation -> sensor-conditioned field reconstruction",
"""| Notebook stage | Frozen information | Physical output |
|---|---|---|
| Patching | One Re110 v/U frame | Exact inverse reshaping; no representation loss from patching alone |
| Source POD | Re90 training frames, rank eight | Reconstructed native field and its representation floor |
| QR versus lattice | Sixteen sensor locations, one declared noise realization | Both reconstructed wakes, shared-scale signed errors and the same native-row profiles |
| Tasks 1–4 | Your patch inversion, projection error and gappy solve | Identify representation error separately from observation/conditioning error |""",
"""Use the same POD basis, target frame, sensor count and noise amplitude when comparing layouts. The two layouts observe different physical coordinates; the numerical noise vector is paired by sensor order and is only one realization. Repeat independently seeded noise realizations before making a statistical statement. Report the actual x/D,y/D positions and condition number. A lower residual at sensor locations need not mean a better field away from them."""),
20:("Fitting a set model -> assessing the reconstructed wake",
"""| Notebook stage | Development data | Evaluation/output |
|---|---|---|
| 120-update teaching fit | Re90/Re110 training; Re100 validation | Validation curve and a fresh Re100 validation-field prediction |
| Retained all/half-sensor comparison | Original frozen source basis and checkpoints | Same Re105 frame for Sensor-set, POD-DeepONet, ridge-imputation and gappy POD |
| Tasks 1–3 | Dropout consistency, parameter count and budget diagnosis | Distinguish missing values from missing sensor tokens |
| Task 4 | Validation history and checkpoint states | Select without reading test metrics; report boundary and step-zero choices |""",
"""The fresh validation picture is not a replay of the published 1,200-update results. The retained Re105 comparison is retrospective and uses the full saved protocol. Set FRAME and CONDITION before inspecting the field. Under drop-half, fixed-input methods use their documented imputation while variable-set and gappy methods receive available observations. Describe this as a system comparison, with the paired same-model augmentation study kept separate."""),
21:("Autonomous dynamics -> phase and physical field errors",
"""| Notebook stage | Time information | Diagnostic |
|---|---|---|
| Primary rollout | Observed frames 156–159 only; predict 160–280 | No reset at frame 210; show error over the complete trajectory |
| Native snapshot | One predeclared predicted frame | CFD, Transformer, history-MLP, DMD and persistence; POD oracle is a representation control |
| Fixed physical probe | Nearest native location to x/D=4,y/D=0.5 | Signed vorticity versus physical time, then frequency/amplitude/phase fit |
| Tasks 1–4 | History update, orthogonal decomposition, frequency fit, rollout audit | Keep truth for scoring only after each autonomous prediction |""",
"""FRAME=240 is a reproducible interior test snapshot, not the easiest frame selected after inspection. Change it to inspect temporal drift without selecting a different checkpoint. The POD oracle uses the target field for projection and is not a deployable forecaster. A small representation floor does not guarantee stable feedback dynamics. Fit phase over the declared full 121-frame window; a constant signal has no identifiable phase."""),
22:("Transfer claims -> matched-case evidence",
"""| Notebook stage | Information/cost contract | Physical output |
|---|---|---|
| Label accounting | Training, validation and observed initialization frames | Union of all target labels, including overlaps only once |
| Retained budget sweep | Same target, label count and seed across arms | Full-field and in-subspace error; retain step-zero and budget-boundary flags |
| Matched native comparison | TARGET_RE, LABELS, SEED and FRAME fixed before plotting | All five arms against the same CFD frame and signed native-row profile |
| Tasks 1–4 | Label union, total updates, floor ratio and paired gains | Preserve negative gains; identify unmatched or untrained controls |""",
"""The default physical comparison uses Re110, 84 target labels, seed17 and frame240. Change one control at a time. Pretrained versus matched-steps is not perfectly matched in source information or validation-horizon coverage; target-POD-MLP also changes the basis. Equal update caps are not equal wall time or FLOPs. The retained cases are already inspected, and plotting them does not create a new blind transfer test.""")
}

DEMOS={
17:[dict(title="Locate the observations in the actual wake",prose="The field is native LBM v/U, Re110, frame100. Overlay the exact frozen sensor locations; sensor indices in the heatmaps refer to these physical points.",
code="""fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
case=cases[110];frame=100
field=case["v"][frame];bound=float(np.max(np.abs(field)))
im=ax.contourf(case["x"],case["y"],field,levels=np.linspace(-bound,bound,25),cmap="jet")
yy,xx=np.meshgrid(case["y"],case["x"],indexing="ij")
ax.scatter(xx.ravel()[ids],yy.ravel()[ids],facecolor="white",edgecolor="black",s=35,label="frozen sensor positions")
ax.set(xlabel="x/D",ylabel="y/D",aspect="equal",title=f"Native wake, Re=110; tU/D={case['t'][frame]:.3f}")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.35),frameon=False)
fig.colorbar(im,ax=ax,label="v/U")
plt.show()
""")],
18:[dict(title="Plot the controlled future-information test",prose="Use the same decoder weights before and after perturbing future input positions. A zero prefix response confirms the masking dependency; it is not a claim of learned CFD accuracy.",
code="""with torch.no_grad():
    masked_difference=(decoder(changed)-decoder(x))[0,:,0].numpy()
    for block in decoder.blocks:block.causal=False
    unmasked_difference=(decoder(changed)-decoder(x))[0,:,0].numpy()
    for block in decoder.blocks:block.causal=True
positions=np.arange(x.shape[1])
fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
ax.plot(positions,masked_difference,"o-",label="causal mask")
ax.plot(positions,unmasked_difference,"s--",label="unmasked control")
ax.axvline(1.5,color=".5",ls=":",label="future inputs changed")
ax.axhline(0,color=".7",lw=.7)
ax.set(xlabel="history position",ylabel="change in first predicted POD coefficient")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.3),ncol=3,frameon=False)
ax.set_xticks(positions);ax.grid(alpha=.2)
plt.show()
""")],
19:[dict(title="Inspect both sensor reconstructions in physical space",prose="A printed condition number does not reveal the lost wake structures. Display the native source-POD projection and both noisy gappy reconstructions against the same CFD frame. All errors retain their signs and full extrema.",
code="""from flowmllab.late_teaching_plots import field_profiles
field=cases[110]["v"][100]
reconstructions={"Source-POD projection":reduced}
for name,sensors in [("QR/D-optimal",qr_ids),("Lattice",grid_ids)]:
    observed=field.ravel()[sensors]+noise
    coefficient=np.linalg.lstsq(representation.modes[sensors],
        observed-representation.mean[sensors],rcond=None)[0]
    reconstructions[name]=(coefficient@representation.modes.T+representation.mean).reshape(field.shape)
figures,diagnostics=field_profiles(cases[110]["x"],cases[110]["y"],field,reconstructions,
    quantity="v/U",case=f"Native wake Re=110, frame100; tU/D={cases[110]['t'][100]:.3f}",
    cut_y=.5,output=RUN_OUTPUT,stem="week19_sensing")
display(pd.DataFrame(diagnostics["metrics"]))
plt.show()
""")],
20:[dict(title="Inspect the fresh teaching fit on validation data",prose="This is the small model just fitted above, evaluated at a fixed Re100 validation frame. It is a development diagnostic and must not be described as a new held-out result or the retained full-budget model.",
code="""from flowmllab.late_teaching_plots import field_profiles
VALIDATION_FRAME=120
fresh_input=lab.sensor_tokens(cases[100]["v"][VALIDATION_FRAME:VALIDATION_FRAME+1],ids,cases[90],mean,scale)
fresh_field=rep.decode(lab.infer(model,fresh_input))[0].reshape(cases[100]["v"].shape[1:])
figures,diagnostics=field_profiles(cases[100]["x"],cases[100]["y"],cases[100]["v"][VALIDATION_FRAME],
    {"Fresh 120-update Sensor-set":fresh_field},quantity="v/U",
    case=f"Native wake validation Re=100, frame{VALIDATION_FRAME}",cut_y=.5)
display(pd.DataFrame(diagnostics["metrics"]))
plt.show()
"""),
dict(title="Compare retained methods on the same observed wake",prose="Set the frame and missing-sensor condition before looking at the answers. These are frozen full-protocol Re105 predictions, not the short fresh fit. Keep all four methods, even where a simpler method is better.",
code="""FRAME=120
CONDITION="drop-half"  # all or drop-half
SEED=17
if CONDITION not in ("all","drop-half"):raise ValueError("Choose all or drop-half")
retained_record=json.loads((EVIDENCE/"sensors.json").read_text())
matched=[r for r in retained_record["rows"] if r["condition"]==CONDITION and r["seed"] in (None,SEED)]
native_shape=cases[105]["v"].shape[1:]
fields={r["method"]:lab.prediction(retained,"sensors__"+r["key"])[FRAME].reshape(native_shape) for r in matched}
figures,diagnostics=field_profiles(cases[105]["x"],cases[105]["y"],cases[105]["v"][FRAME],fields,
    quantity="v/U",case=f"Native wake retained test Re=105, frame{FRAME}, {CONDITION}",cut_y=.5,
    output=RUN_OUTPUT,stem="week20_retained")
display(pd.DataFrame(diagnostics["metrics"]))
plt.show()
""")],
21:[dict(title="Compare an actual forecast field and its probe trace",prose="Field and temporal plots answer different questions. The native snapshot shows where a wake is displaced; the fixed probe reveals phase/amplitude drift. The oracle projection is explicitly a representation control using reference data.",
code="""from flowmllab.late_teaching_plots import field_profiles
FRAME=240
if not 160<=FRAME<281:raise ValueError("Choose a predicted frame in [160,281)")
chosen=[r for r in record["rows"] if r["seed"] in (None,17)]
shape=cases[110]["omega"].shape[1:]
fields={r["method"]:lab.prediction(retained,"forecast__"+r["key"])[FRAME-160].reshape(shape) for r in chosen}
figures,diagnostics=field_profiles(cases[110]["x"],cases[110]["y"],cases[110]["omega"][FRAME],fields,
    quantity="vorticity D/U",case=f"Native wake Re=110 forecast frame{FRAME}; no reset",cut_y=.5,
    output=RUN_OUTPUT,stem="week21_forecast")
display(pd.DataFrame(diagnostics["metrics"]))
plt.show()
iy=int(np.argmin(abs(cases[110]["y"]-.5)));ix=int(np.argmin(abs(cases[110]["x"]-4.)))
probe=np.ravel_multi_index((iy,ix),shape)
time_axis=cases[110]["t"][160:281]
fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
ax.plot(time_axis,cases[110]["omega"][160:281,iy,ix],color="black",lw=2,label="CFD")
for row in chosen:
    ax.plot(time_axis,lab.prediction(retained,"forecast__"+row["key"])[:,probe],
            lw=1.3,label=row["method"])
ax.set(xlabel="tU/D",ylabel="vorticity D/U",
       title=f"Native probe x/D={cases[110]['x'][ix]:.4g}, y/D={cases[110]['y'][iy]:.4g}")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.45),ncol=3,frameon=False,fontsize=8)
ax.grid(alpha=.2);plt.show()
""")],
22:[dict(title="Hold target, labels, seed and frame fixed in physical space",prose="Compare all five retained arms at one exact target/budget/seed/frame. Target-POD-MLP changes the representation; the matched-step arm changes the training initialization and information history. Keep those distinctions when explaining the flow.",
code="""from flowmllab.late_teaching_plots import field_profiles
TARGET_RE=110
LABELS=84
SEED=17
FRAME=240
matched=[r for r in rows if r["target_re"]==TARGET_RE and r["labels"]["total_target_labels"]==LABELS and r["seed"]==SEED]
if len(matched)!=5:raise ValueError("Choose a recorded target, label count and seed")
if not 160<=FRAME<281:raise ValueError("Choose a predicted frame in [160,281)")
shape=cases[TARGET_RE]["omega"].shape[1:]
fields={r["arm"]:lab.prediction(retained,"transfer__"+r["key"])[FRAME-160].reshape(shape) for r in matched}
figures,diagnostics=field_profiles(cases[TARGET_RE]["x"],cases[TARGET_RE]["y"],
    cases[TARGET_RE]["omega"][FRAME],fields,quantity="vorticity D/U",
    case=f"Native wake Re={TARGET_RE}, {LABELS} labels, seed{SEED}, frame{FRAME}",cut_y=.5,
    output=RUN_OUTPUT,stem="week22_transfer")
display(pd.DataFrame(diagnostics["metrics"]))
print("Retained comparison only; source/target information and representation differ between arms.")
plt.show()
""")]
}

def main():
    for w in range(17,23):
        p=ROOT/f'course/lessons/week{w}.json';spec=json.loads(p.read_text(encoding='utf-8'))
        title,table,text=MAPS[w]
        spec['reading_map']=table+'\n\n'+text
        for item in DEMOS[w]:
            i=next((i for i,d in enumerate(spec['demos']) if d['title']==item['title']),None)
            if i is None:spec['demos'].append(item)
            else:spec['demos'][i]=item
        if w==18:
            d=spec['demos'][0]
            if 'LM_STEPS=' not in d['code']:
                d['code']='LM_STEPS=160\nMODEL_WIDTH=24\nCONTEXT=24\nSAMPLE_TEMPERATURE=.8\nassert MODEL_WIDTH%4==0 and CONTEXT>1 and SAMPLE_TEMPERATURE>0\n'+d['code']
                d['code']=d['code'].replace('context=24','context=CONTEXT').replace('range(160)','range(LM_STEPS)')
                d['code']=d['code'].replace('lab.TinyLM(len(vocab),context)','lab.TinyLM(len(vocab),context,width=MODEL_WIDTH)')
                d['code']=d['code'].replace('[:,-1]/.8','[:,-1]/SAMPLE_TEMPERATURE')
        if w==21:
            d=spec['demos'][0]
            d['code']=d['code'].replace('figsize=(9,4)','figsize=(11,5),constrained_layout=True')
            d['code']=d['code'].replace("ax.legend(fontsize=8)","ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.5),ncol=2,frameon=False)")
        if w==22:
            d=spec['demos'][1]
            d['code']=d['code'].replace(";axes[j,col].legend(fontsize=6)","")
            if 'handles,labels=axes[0,0].get_legend_handles_labels()' not in d['code']:
                d['code']=d['code'].replace("fig.tight_layout();plt.show()",
                    "handles,labels=axes[0,0].get_legend_handles_labels()\nfig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.12),ncol=3,fontsize=8,frameon=False)\nfig.tight_layout();plt.show()")
        p.write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
        md=ROOT/f'lectures/source/week{w}_transformer_course.md'
        old=md.read_text(encoding='utf-8')
        old=old.replace('the connection requested for the LLM extension','the connection between sequence learning and fluid prediction')
        marker='\n## Laboratory route: from inputs to physical conclusions\n'
        old=old.split(marker)[0]
        old+=marker+'\n'+table+'\n\n'+text+'\n'
        md.write_text(old,encoding='utf-8',newline='\n')
        readme=ROOT/f'notebooks/week{w}/README.md'
        guide=readme.read_text(encoding='utf8')
        link=f'[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week{w}/W{w}_CFD_Transformer.ipynb)'
        if link not in guide:
            guide=guide.replace(f'[Student laboratory](W{w}_CFD_Transformer.ipynb)',
                                f'[Student laboratory](W{w}_CFD_Transformer.ipynb) | '+link)
        old_access='This published edition includes a complete local/ZIP setup route.'
        if old_access in guide:
            guide=guide[:guide.index(old_access)].rstrip()+'\n\nThe setup cell locates a checkout or fetches the public lesson dependencies in Colab. Run it before the examples. Keep the declared case, time window and sensor identities when comparing physical fields; the retained research results remain separate from the short classroom fits.\n'
        readme.write_text(guide,encoding='utf8',newline='\n')

    p=ROOT/'qa/build_transformer_course.py';s=p.read_text(encoding='utf-8')
    a=s.index("SETUP=");b=s.index("\n\ndef notebook(",a)
    setup=bootstrap_source('flowmllab/transformer_course.py',TRANSFORMER_PATHS)+'''
import tempfile, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from flowmllab import transformer_course as lab
lab.seed_all(17)
cases,input_manifest=lab.load_data(ROOT)
EVIDENCE=ROOT/"results/transformer_course_v3"
_student_parent=ROOT/"student_runs/transformer_labs"
_student_parent.mkdir(parents=True,exist_ok=True)
workspace=tempfile.TemporaryDirectory(prefix="classroom-",dir=_student_parent)
RUN_OUTPUT=Path(workspace.name)
task_status={}
print("Loaded checksummed native wake fields. Optional student work stays in:",RUN_OUTPUT)
'''
    s=s[:a]+"SETUP="+repr(setup)+"\n"+s[b:]
    anchor="    for demo in spec['demos']:"
    addition="    if spec.get('reading_map'):\n        cells.insert(1,nbf.v4.new_markdown_cell('## Laboratory route and reading map\\n\\n'+spec['reading_map']))\n"
    if addition not in s:s=s.replace(anchor,addition+anchor)
    s=s.replace("nbf.write(nb,destination)","destination.write_text(nbf.writes(nb),encoding='utf-8',newline='\\n')")
    p.write_text(s,encoding='utf-8',newline='\n')
    p=ROOT/'qa/refresh_late_course.py';s=p.read_text(encoding='utf-8')
    s=s.replace('nbf.write(book,path)',"path.write_text(nbf.writes(book),encoding='utf-8',newline='\\n')")
    p.write_text(s,encoding='utf-8',newline='\n')

if __name__=='__main__':main()
