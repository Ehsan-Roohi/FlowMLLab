"""Preserve the existing modal lesson and insert worked, editable sensing sections."""
from pathlib import Path
import json
from course_refresh_early_notebooks import read,text,find,after,before,replace,md,code,save,MARK

def build():
 p,nb=read('notebooks/week05_06/W5_Lab2_Sparse_Sensing_Dynamics.ipynb')
 if any(MARK in text(c) for c in nb['cells']):return
 replace(nb,'Worked two-sensor','(after Week 7 prerequisites)','(before real wake measurements)')
 replace(nb,'Worked two-sensor',"Study this lab after Week 7's modal dynamics; its directory does not impose an\nadditional mandatory pre-project dynamics unit.","The sensing sections require POD and least squares. The autonomous dynamics extension additionally uses Week 7's modal dynamics; complete it after that lesson.")
 after(nb,'# Week 5 companion',md(r'''## Aim and route

The required CPU experiment reconstructs the transverse-velocity field of a cylinder wake from a small number of point measurements. It asks whether sensor placement, basis rank and measurement noise explain the reconstruction error. The second, optional part identifies a sparse differential equation in modal coordinates. A reconstruction uses new measurements at each frame; an autonomous forecast does not. Do not mix these information budgets.

Read the worked conditioning example, verify an exactly low-rank matrix, freeze the full-case protocol, fit the basis on Re=90/110 only, select sensor count using Re=100, and finally report Re=105. Those public cases have already been inspected in this course; this is a reproducible held-out teaching study, not a new blind research claim. The optional dynamics branch uses one Re=110 trajectory with nonoverlapping time windows.'''))
 # Replace shared renderer dependency by a scoped presentation helper; no shared evidence writes.
 replace(nb,'tracked_evidence =','from run_modal_labs import figures','from flowmllab.early_sensing import sensor_experiment as run_sensor_study, sensing_figures')
 after(nb,'## 1. Warm-up',md(r'''### Derive the inverse problem before choosing sensors

Flatten one field into $q\in\mathbb R^n$ and approximate it by $q=\bar q+\Phi a+r$. The training mean is $\bar q$, the $n\times r_m$ POD basis is $\Phi$, the modal coefficients are $a$, and $r$ is the unresolved field. A row-selection matrix $C$ picks $m$ sensor coordinates. Given observations $y=Cq+\epsilon$, the coefficient estimate solves

$$\widehat a=\arg\min_a\|(C\Phi)a-(y-C\bar q)\|_2^2,\qquad\widehat q=\bar q+\Phi\widehat a.$$

Here $\epsilon$ is measurement noise; $m$ must be at least the modal rank, and $C\Phi$ must have full column rank. The code uses least squares rather than explicitly inverting a matrix. A small singular value amplifies noise. Good conditioning cannot remove $r$, so compare with a **full-field POD projection** as a representation floor. That projection uses every target value and is an oracle, not a competing sensor algorithm.

Pivoted QR first chooses rank-many linearly independent spatial rows. Extra sensors are added by a D-optimal greedy criterion using the selected Gram matrix. The random comparison uses exactly the same sensor budget and paired noise realization. Merely adding adjacent, redundant sensors need not improve information.'''))
 after(nb,'assert np.allclose(recovered',code(r'''# A hand-checkable observation system: no CFD or learned model is involved.
true_coefficients = np.array([1.,2.]); perturbation = np.array([0.,.01])
rows=[]
for separation in [.01,.1,1.]:
    A=np.array([[1.,0.],[1.,separation]])
    observed=A@true_coefficients
    estimate=np.linalg.lstsq(A,observed+perturbation,rcond=None)[0]
    rows.append({"row_separation":separation,"condition_number":np.linalg.cond(A),
        "noise_norm":np.linalg.norm(perturbation),"coefficient_error":np.linalg.norm(estimate-true_coefficients)})
display(pd.DataFrame(rows))
fig,ax=plt.subplots(figsize=(6.5,3.8),layout="constrained")
ax.loglog([r["condition_number"] for r in rows],[r["coefficient_error"] for r in rows],"o-")
ax.set(xlabel="Condition number of A",ylabel="Coefficient error norm",title="Same measurement perturbation; different row independence")
ax.grid(alpha=.2);plt.show()
'''))
 after(nb,'## 2. Real-data protocol',md(r'''### Editable controls and what stays fixed

`SENSOR_RANK` is the number of centered POD modes, not the number of sensors. `SENSOR_BUDGETS` are candidate point counts, each at least that rank. `NOISE_FRACTION` multiplies the RMS of **training** velocity to set observation-noise standard deviation. `NOISE_SEEDS` gives replicated measurement noise and random layouts; it does not generate independent CFD realizations. `VALIDATION_TARGET` is the predeclared relative-L2 acceptance target used to choose the smallest qualifying QR budget, or the best validation budget if none qualify.

The code fits the basis only once per chosen rank on complete Re=90 and110 trajectories. Freeze rank before opening the final case; a rank sweep requires validation-based selection too. Test profiles, global errors and extrema are evaluated only after that decision. The native ROI is downstream of the cylinder; its edges are not physical walls and its scalar integrals are not mass-conservation proofs.'''),code('''SENSOR_RANK=8
SENSOR_BUDGETS=(8,16,32)
NOISE_FRACTION=.01
NOISE_SEEDS=(10,11,12,13,14)
VALIDATION_TARGET=.05
RUN_DYNAMICS_BRIDGE=False  # Optional Week7 extension; required sensing remains CPU-only.
'''))
 c=find(nb,'# Fresh CPU fits');c['source']='''# Fresh sensor fits on the retained LBM fields, with all controls passed explicitly.
import os,time,json
scratch=Path(os.environ.get("FLOWMLLAB_EARLY_OUTPUT",str(ROOT/"student_runs/week05_sensing")))
scratch.mkdir(parents=True,exist_ok=True)
started=time.perf_counter()
with threadpool_limits(limits=1):
    sensing,examples=run_sensor_study(cases,rank=SENSOR_RANK,budgets=SENSOR_BUDGETS,
        noise_fraction=NOISE_FRACTION,seeds=NOISE_SEEDS,validation_target=VALIDATION_TARGET)
print(f"Sensor reconstruction CPU time: {time.perf_counter()-started:.2f} s")
(scratch/"sensor_metrics.json").write_text(json.dumps(sensing,indent=2)+"\\n")
forecast=predictions=None
if RUN_DYNAMICS_BRIDGE:
    with threadpool_limits(limits=1):
        forecast,predictions=forecast_experiment(cases)
'''
 c=find(nb,"rows = pd.DataFrame(sensing");c['source']='''rows=pd.DataFrame(sensing['records'])
display(rows[rows['split']=='validation'][['method','budget','seed','relative_l2']])
print('Validation-selected sensor budget:',sensing['selected_budget'])
display(rows[rows['split']=='test'][['method','budget','seed','condition_number','relative_l2','edge_relative_l2']])
for index,figure in enumerate(sensing_figures(cases,sensing,examples)):
    figure.savefig(scratch/f"sensor_diagnostic_{index+1}.png",dpi=180)
    plt.show()
'''
 c=find(nb,"display(pd.DataFrame(forecast['sindy_candidates'])");s=text(c);s=s.replace("display(Image(filename=str(scratch/'forecast_audit.png')))",'''fig,ax=plt.subplots(figsize=(8,4),layout="constrained")
truth=cases[110]['omega'][160:]
for name in [forecast['selected_dmd'],forecast['selected_sindy'],'POD-r2-oracle']:
    if name in predictions:
        errors=np.linalg.norm((predictions[name]-truth).reshape(len(truth),-1),axis=1)/np.linalg.norm(truth.reshape(len(truth),-1),axis=1)
        ax.plot(cases[110]['t'][160:],100*errors,label=name)
ax.axvline(cases[110]['t'][210],color="black",ls=":",label="Assessment boundary")
ax.set(xlabel="t U/D",ylabel="Frame relative L2 (%)")
ax.legend(loc="upper left",bbox_to_anchor=(1.02,1));ax.grid(alpha=.2);plt.show()''')
 c['source']='if RUN_DYNAMICS_BRIDGE:\n'+''.join('    '+line+'\n' for line in s.splitlines())+'else:\n    print("Optional dynamics bridge not run. Read its derivation, then enable it after Week7.")\n'
 after(nb,'## 3. Separate representation',md(r'''### What the optional sparse-dynamics fit actually solves

The two retained POD coefficients are divided by their training standard deviations to give $z_1,z_2$. A polynomial library $\Theta(z)$ contains the constant, linear, quadratic and cubic terms. Rather than differentiating noisy observations directly, the integral formulation fits

$$z(t_{k+w})-z(t_k)\approx\int_{t_k}^{t_{k+w}}\Theta(z(t))\,dt\;\Xi.$$

The window length is four sample intervals and the integral uses trapezoidal quadrature. Sequential thresholded least squares fits coefficients, removes terms below each candidate threshold, and refits the remaining terms. Only the first160 Re110 frames define the basis, normalization and ODE fit. Frames160-209 select the threshold; the final71 frames assess the frozen rollout without a reset. An unstable candidate is a failed autonomous prediction, not a reason to clip its state or silently replace it.

A near-periodic training orbit constrains the vector field near that orbit. It cannot identify arbitrary off-orbit cubic dynamics. Separate the rank-two oracle error from dynamical error and from coarse-grid LBM error. The optional helper also computes DMD and a nonlinear history baseline for the Week7 comparison; its full cost is more than fitting a two-state ODE.'''))
 before(nb,'## Sources, provenance',md(r'''## Engineering exercises and decision table

1. **Conditioning:** repeat the algebraic two-sensor example with perturbations of magnitude0.001,0.01,0.1. Report coefficient error divided by observation error and compare it with the smallest singular value. Explain why the condition number is a bound/diagnostic rather than an exact prediction for every noise direction.
2. **Measurement budget:** keep rank8 and the five seeds fixed. Compare8,16,32 sensors with the same1% noise. Use the validation rule before examining Re105; report every seeded error, the selected budget, native profile and full-field oracle. If the target is not reached, state that explicitly.
3. **Noise versus representation:** at the frozen rank and layout compare noise fractions0,0.01,0.03. Inspect signed spatial discrepancies and the difference from the oracle. Zero sensor noise does not remove an unresolved mode. Treat these public-case repetitions as post-hoc sensitivity, not repeated untouched tests.
4. **Rank:** use rank4 and8 with sensor budgets at least the rank. Select by Re100 only. Record offline SVD cost, online reconstruction cost and condition number. More captured energy can worsen conditioning or increase the measurement requirement.
5. **Optional dynamics:** enable the Week7 bridge and list each retained ODE term with its coefficient. Compare continuous rollout, the rank-two oracle and DMD. Describe which candidate fails first and why a small one-step fit need not be stable over many steps.

Submit a table with one row per configuration: rank, sensor count, noise fraction, seed, selection role, condition number, field error, worst-frame error, runtime and your physical interpretation. Include the reconstruction/profiles, not only that table. Preserve the original reference data and the default result before changing a control.'''))
 save(p,nb)
if __name__=='__main__':build()