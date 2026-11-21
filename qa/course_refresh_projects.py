import json
"""Repair concrete P3/P4 diagnostic gaps while retaining their trained algorithms."""
from course_refresh_early_notebooks import read,text,find,after,md,code,save,MARK

def build():
 p,nb=read('notebooks/week05_06/P3_POD_Study.ipynb')
 if not any(MARK in text(c)for c in nb['cells']):
  c=find(nb,'LOW_RANK = 1');s=text(c);s=s.replace('else:\n    print("Requested low-rank/basis pair', '''    low_figure=w5_common.plot_case_evidence(data,VAL_RE,
        {"Low-rank coefficient prediction":low_pred,"Low-rank full-field oracle":low_rec},
        f"Validation diagnosis: rank {LOW_RANK}")
    plt.show()
else:
    print("Requested low-rank/basis pair''');c['source']=s
  after(nb,'LOW_RANK = 1',md('''The low-rank figure uses the validation case. Its oracle projects the entire reference field and therefore has more information than the coefficient predictor. A shared missed vortex or pressure feature points to representation loss; an extra predictor discrepancy indicates coefficient-map error. Compare the native profiles as well as the global score. Neither panel measures CFD discretization error.'''))
  save(p,nb)
 p,nb=read('notebooks/week05_06/P4_Uncertainty_Study.ipynb')
 if not any(MARK in text(c)for c in nb['cells']):
  for c in nb['cells']:
   s=text(c)
   if c['cell_type']=='code' and ('seed_predictions[seed]={}'in s or 'size_predictions[label]={}'in s):
    s=s.replace('        seed_predictions[seed]={}','        fit_seconds=time.time()-t0\n        seed_predictions[seed]={}')
    s=s.replace('        size_predictions[label]={}','        fit_seconds=time.time()-t0\n        size_predictions[label]={}')
    s=s.replace('"training_seconds":time.time()-t0','"training_seconds":fit_seconds');c['source']=s
  c=find(nb,'stack_u=np.stack');s=text(c)
  s=s.replace('a=ax[0].contourf(X,Y,error,28)','limits=np.linspace(0,max(float(error.max()),float(spread.max()),1e-12),29)\n    a=ax[0].contourf(X,Y,error,levels=limits,cmap="turbo")')
  s=s.replace('a=ax[1].contourf(X,Y,spread,28)','a=ax[1].contourf(X,Y,spread,levels=limits,cmap="turbo")')
  s=s.replace('    plt.tight_layout();plt.show()','''    for panel in ax[:2]:panel.set(xlabel="x/L",ylabel="y/L",aspect="equal")
    fig.suptitle(f"Re = {r}; {len(SEEDS)} training seeds; full linear error/spread range")
    plt.tight_layout();plt.show()''',1)
  s=s.replace('for a in ax:a.grid(.3);a.legend();a.set_xlabel("blind Re")','for a in ax:a.grid(alpha=.3);a.legend(loc="upper left",bbox_to_anchor=(1.02,1));a.set_xlabel("public held-out Re")')
  c['source']=s
  after(nb,'stack_u=np.stack',md(r'''### The ensemble must still reproduce a physical field

Sample spread across training seeds measures variability of this particular training procedure at fixed data, architecture and stopping rule. It is neither observation noise nor a calibrated posterior uncertainty. The ensemble can be confidently wrong if all members share representation or optimization bias. Compare the ensemble mean with CFD in velocity, pressure and streamlines before interpreting a spread/error correlation.

The profile band below is one **sample standard deviation** (ddof=1) across trained members, not a 95% prediction or confidence interval. It does not include source-CFD discretization error. Report member count and each seed; do not drop an unfavorable member after viewing the final case. Training time is recorded immediately after fitting and excludes later prediction and metric evaluation.'''),code(r'''if VARIANT=="4A":
    physical=w5_common.plot_case_evidence(data,r,{"Ensemble mean":ensemble},"Training-seed ensemble")
    plt.show()
    x,y=data['x'],data['y'];ix=int(np.argmin(abs(x-.5)));iy=int(np.argmin(abs(y-.5)))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    mu=stack_u[:,:,ix].mean(axis=0);sd=stack_u[:,:,ix].std(axis=0,ddof=1)
    axes[0].plot(mu,y,color="tab:blue",label="Ensemble mean")
    axes[0].fill_betweenx(y,mu-sd,mu+sd,color="tab:blue",alpha=.2,label="One sample SD across seeds")
    axes[0].plot(data['u'][idx,:,ix],y,color="black",label="CFD")
    axes[0].set(xlabel="u/U_lid",ylabel="y/L",title=f"Re={r}; x/L={float(x[ix]):.3f}")
    mu=stack_v[:,iy,:].mean(axis=0);sd=stack_v[:,iy,:].std(axis=0,ddof=1)
    axes[1].plot(x,mu,color="tab:blue",label="Ensemble mean")
    axes[1].fill_between(x,mu-sd,mu+sd,color="tab:blue",alpha=.2,label="One sample SD across seeds")
    axes[1].plot(x,data['v'][idx,iy,:],color="black",label="CFD")
    axes[1].set(xlabel="x/L",ylabel="v/U_lid",title=f"Re={r}; y/L={float(y[iy]):.3f}")
    for a in axes:a.legend(loc="upper center",bbox_to_anchor=(.5,-.20),fontsize=9);a.grid(alpha=.2)
    plt.show()
else:
    physical=w5_common.plot_case_evidence(data,275,
        {label:predictions[275] for label,predictions in size_predictions.items()},
        "Training-case-count comparison")
    plt.show()
'''))
  save(p,nb)

def kinetic_plots():
 p,nb=read('notebooks/week05_06/P5_Rarefied_Cavity.ipynb')
 if any(MARK in text(c)for c in nb['cells']):return
 c=find(nb,'# Plot one complete held-out physical case.')
 c['source']='''# Plot the same complete held-out case with shared physical levels.
from w5_kinetic_figures import plot_kinetic_evidence
eval_data=D_single if VARIANT=="5A" else D_kn
i=eval_ids[-1]
shown={name:examples[(name,i)]for name in model_specs}
for figure in plot_kinetic_evidence(eval_data,i,shown):
    plt.show()
'''
 after(nb,'# Plot the same complete held-out case',md('''### Interpret a finite-window kinetic comparison

The reference is the sampled mini-DSMC teaching field. The noisy temperature contour is omitted; its native centerline and signed discrepancy remain visible. Velocity error uses a common full linear range, and centerlines interpolate to x/L=0.5 while retaining native y locations. No physical wall values are manufactured from cell-center samples. The pressure and energy budgets of a production kinetic solver are not established by these figures.

For 5A, the reference has seed 33; neither seed 11 nor seed 22 provides its target values. This tests independent stochastic labels at the declared conditions, not a new Knudsen regime. A smooth network may conceal sampling fluctuations and may also miss genuine local structure. Compare the signed profiles with the seed-spread table before interpreting a better-looking streamline as higher accuracy.'''))
 save(p,nb)

def other_project_plot_style():
 for relative in ['notebooks/week05_06/P2_Physics_Guided_DNN.ipynb','notebooks/week05_06/P6_FP_Cavity_Closure.ipynb']:
  p,nb=read(relative);changed=False
  for c in nb['cells']:
   if c['cell_type']!='code':continue
   source=text(c);updated=source
   if 'plt.hist(wall_distance' in source:
    updated=updated.replace('plt.hist(wall_distance(Xtr),bins=30)','plt.hist(wall_distance(Xtr),bins=30,histtype="step",linewidth=1.6)')
   if 'ax.hist(train[' in source and 'q_norm' in source:
    updated=updated.replace('bins=50, alpha=0.7','bins=50, histtype="step", linewidth=1.6').replace('bins=50, alpha=0.6','bins=50, histtype="step", linewidth=1.6')
   if 'Validation tradeoff'in source or ('ax.hist(train[' in source and 'q_norm' in source):
    updated=updated.replace('ax.legend()','ax.legend(loc="upper left",bbox_to_anchor=(1.02,1))')
   if updated!=source:c['source']=updated;changed=True
  if changed:p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')

if __name__=='__main__':
 build()
 kinetic_plots()
 other_project_plot_style()
