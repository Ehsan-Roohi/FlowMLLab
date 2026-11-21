"""Parametrized Week5 sensing lab; source LBM and prior retained results are read-only."""
import numpy as np
from .modal_tools import fit_pod,select_sensors,reconstruct_sensors,reconstruct_pod,project_pod
from .modal_experiments import score

def sensor_experiment(cases, rank=8, budgets=(8,16,32), noise_fraction=.01,
                      seeds=(10,11,12,13,14), validation_target=.05):
    """Train on Re90/110; select budget on Re100; report Re105 only afterwards.

    Settings are controls, not a claim of new blind CFD. Paired noise prevents
    different layouts at the same seed from receiving unrelated noise draws.
    """
    budgets=tuple(int(b) for b in budgets); seeds=tuple(int(s) for s in seeds)
    if rank < 1 or not budgets or not seeds or min(budgets) < rank:
        raise ValueError("Require positive rank, nonempty seeds, and every budget >= rank")
    if len(set(budgets)) != len(budgets) or len(set(seeds)) != len(seeds):
        raise ValueError("Budgets and seeds must be distinct")
    if not np.isfinite(noise_fraction) or noise_fraction < 0 or validation_target < 0:
        raise ValueError("Noise and validation target must be finite and nonnegative")
    train=np.concatenate([cases[r]['v'] for r in (90,110)]);shape=train.shape[1:]
    flat=train.reshape(len(train),-1);pod=fit_pod(flat,rank=rank)
    sigma=noise_fraction*float(np.sqrt(np.mean(flat**2)));records=[];positions={}
    # All locations frozen before validation/test fields are evaluated.
    for budget in budgets:
        positions[f'QR-{budget}']=select_sensors(pod.modes,budget)
        for seed in seeds:
            positions[f'random-{budget}-{seed}']=np.random.default_rng(seed).choice(flat.shape[1],budget,replace=False)
    examples={}
    for re,split in ((100,'validation'),(105,'test')):
        truth=cases[re]['v'];f=truth.reshape(len(truth),-1)
        # Paired full-grid artificial measurement noise, not additional CFD noise.
        for seed in seeds:
            noisy=f+np.random.default_rng(seed+1000*re).normal(0,sigma,f.shape)
            for budget in budgets:
                for method in ('QR','random'):
                    key=f'QR-{budget}' if method=='QR' else f'random-{budget}-{seed}'
                    ids=positions[key];pred=reconstruct_sensors(pod,ids,noisy[:,ids]).reshape(truth.shape)
                    records.append({'split':split,'case':re,'method':method,'budget':budget,'seed':seed,
                        'condition_number':float(np.linalg.cond(pod.modes[ids])),**score(pred,truth,cases[re])})
                    if re==105 and method=='QR' and seed==seeds[0]:examples[budget]=pred
        oracle=reconstruct_pod(pod,project_pod(pod,f)).reshape(truth.shape)
        mean=np.broadcast_to(pod.mean.reshape(shape),truth.shape)
        for name,pred in (('POD-oracle',oracle),('training-mean',mean)):
            records.append({'split':split,'case':re,'method':name,'budget':None,'seed':None,**score(pred,truth,cases[re])})
        if split=='validation':
            errors={b:float(np.mean([r['relative_l2'] for r in records if r['split']==split and r['method']=='QR' and r['budget']==b])) for b in budgets}
            meets=[b for b in errors if errors[b]<=validation_target]
            selected=min(meets) if meets else min(errors,key=errors.get)
    return {'selected_budget':selected,'noise_sigma':sigma,'validation_means':errors,'records':records,
        'positions':{k:v.tolist() for k,v in positions.items()},
        'settings':{'rank':rank,'budgets':list(budgets),'noise_fraction':noise_fraction,'seeds':list(seeds),'validation_target':validation_target},
        'replicate_scope':'Artificial measurement-noise/layout seeds on one held Reynolds case; not independent CFD realizations.'},examples

def sensing_figures(cases, sensing, examples):
    """Physical reconstruction, native profiles and conditioning evidence."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    c=cases[105];truth=c['v'];budget=sensing['selected_budget'];estimate=examples[budget]
    ids=np.array(sensing['positions'][f'QR-{budget}']);iy,ix=np.unravel_index(ids,truth.shape[1:])
    field=truth[-1];pred=estimate[-1];error=abs(pred-field)
    lo=min(float(field.min()),float(pred.min()));hi=max(float(field.max()),float(pred.max()))
    levels=np.linspace(lo,hi,33);lines=levels[4:-4:4]
    fig,axes=plt.subplots(2,1,figsize=(10.5,6.7),layout='constrained')
    im=axes[0].contourf(c['x'],c['y'],field,levels=levels,cmap='turbo')
    axes[0].contour(c['x'],c['y'],field,levels=lines,colors='black',linewidths=.65)
    axes[0].contour(c['x'],c['y'],pred,levels=lines,colors='magenta',linestyles='dashed',linewidths=.9)
    axes[0].scatter(c['x'][ix],c['y'][iy],s=20,facecolors='none',edgecolors='white',linewidths=1)
    axes[0].legend(handles=[Patch(facecolor=plt.get_cmap('turbo')(.5),label='LBM'),
        Line2D([],[],color='black',label='LBM contours'),Line2D([],[],color='magenta',ls='--',label='Sensor reconstruction'),
        Line2D([],[],color='white',marker='o',markerfacecolor='none',markeredgecolor='black',ls='',label='Sensors')],
        loc='upper center',bbox_to_anchor=(.5,1.28),ncol=4,fontsize=8)
    fig.colorbar(im,ax=axes[0],label='v/U')
    im=axes[1].contourf(c['x'],c['y'],error,levels=np.linspace(0,max(float(error.max()),1e-12),25),cmap='turbo')
    fig.colorbar(im,ax=axes[1],label='Absolute error in v/U (linear)')
    for ax in axes:ax.set(xlabel='x/D',ylabel='y/D',aspect='equal')
    axes[1].set_title(f'Re = 105; {budget} sensors; noise seed {sensing["settings"]["seeds"][0]}; native 32 x 78 ROI')
    fig2,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    j=int(np.argmin(abs(c['y']-.5)))
    axs[0].plot(c['x'],field[j],color='black',label='LBM')
    axs[0].plot(c['x'],pred[j],ls='--',color='magenta',label='Sensor reconstruction')
    axs[0].set(xlabel='x/D',ylabel='v/U',title=f'Final frame; native row y/D = {float(c["y"][j]):.3f}')
    frame_errors=np.linalg.norm((estimate-truth).reshape(len(truth),-1),axis=1)/np.linalg.norm(truth.reshape(len(truth),-1),axis=1)
    axs[1].plot(c['t'],100*frame_errors,color='tab:blue')
    axs[1].set(xlabel='t U/D',ylabel='Frame relative L2 (%)',title='Every retained Re105 time frame')
    axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
    for ax in axs:ax.grid(alpha=.2)
    fig3,axs=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    records=sensing['records'];budgets=sensing['settings']['budgets']
    for method,color in [('QR','tab:blue'),('random','tab:orange')]:
        means=[]
        for b in budgets:
            rows=[r for r in records if r['split']=='test' and r['method']==method and r['budget']==b]
            values=[100*r['relative_l2'] for r in rows]
            axs[0].scatter([b]*len(values),values,s=24,color=color,alpha=.5)
            means.append(np.mean(values))
            axs[1].scatter([r['condition_number'] for r in rows],values,s=24,color=color,alpha=.6)
        axs[0].plot(budgets,means,'o-',color=color,label=method)
        axs[1].plot([],[],marker='o',ls='',color=color,label=method)
    oracle=next(r['relative_l2'] for r in records if r['split']=='test' and r['method']=='POD-oracle')
    axs[0].axhline(100*oracle,color='black',ls='--',label='Full-field POD oracle')
    axs[0].set(xlabel='Sensor count',ylabel='Relative L2 (%)',yscale='log',xticks=budgets,title='Re105; every declared noise/layout seed')
    axs[1].set(xlabel='Condition number of sensor matrix',ylabel='Relative L2 (%)',xscale='log',yscale='log',title='Conditioning is one part of the error')
    for ax in axs:ax.grid(alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
    return fig,fig2,fig3