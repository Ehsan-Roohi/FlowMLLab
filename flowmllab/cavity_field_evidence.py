"""Physical diagnostics for the frozen high-Re full-field neural study."""
from pathlib import Path
import json
import numpy as np
from . import cavity_field_operator as op
from .cavity_highre import audit_field,file_hash
from . import ghia

COLORS={'CFD':'black','Adam':'#999999','Selected NN':'#D55E00','Linear interpolation':'#0072B2'}

def load(root):
    root=Path(root);out=root/'results/cavity_field_deeponet'
    info=json.loads((out/'frozen_selection.json').read_text())
    receipt=json.loads((out/'scoring_receipt.json').read_text())
    if file_hash(out/'frozen_selection.json')!=receipt['frozen_selection_sha256']:raise ValueError('Selection changed after scoring')
    if file_hash(out/'frozen_predictions.npz')!=info['frozen_predictions_sha256']:raise ValueError('Frozen predictions changed')
    if file_hash(out/'trunk.npz')!=info['trunk_sha256']:raise ValueError('Frozen POD changed')
    trunk=dict(np.load(out/'trunk.npz',allow_pickle=False))
    frozen=dict(np.load(out/'frozen_predictions.npz',allow_pickle=False))
    trains=[dict(np.load(root/f'results/cavity_highre_ghia/n512/re{re}.npz',allow_pickle=False)) for re in op.TRAIN]
    for re,f in zip(op.TRAIN,trains):
        if file_hash(root/f'results/cavity_highre_ghia/n512/re{re}.npz')!=info['data_read_during_training'][str(re)]:raise ValueError('Training field changed')
    cases={}
    adam=[j for j,m in enumerate(info['candidate_models']) if m['configuration']=='adam_2']
    for i,re in enumerate(op.TEST):
        path=root/f'results/cavity_highre_ghia/n512/re{re}.npz'
        if file_hash(path)!=receipt['heldout_field_sha256'][str(re)]:raise ValueError('Test field changed')
        truth=dict(np.load(path,allow_pickle=False))
        baseline_key='log_Re_coefficients' if info['selected_baseline_coordinate']=='log' else 'linear_Re_coefficients'
        coefficients={'Selected NN':frozen['candidate_coefficients'][info['selected_indices'],i].mean(axis=0),
                      'Linear interpolation':frozen[baseline_key][i],
                      'Adam':frozen['candidate_coefficients'][adam,i].mean(axis=0)}
        predictions={'CFD':truth}
        for name,coef in coefficients.items():
            flat=op.reconstruct(trunk,coef);n=truth['u'].size
            mix=op.mixing_weights(trunk,coef)
            f={k:truth[k] for k in ('x','y')}
            f.update(u=flat[:n].reshape(truth['u'].shape),v=flat[n:].reshape(truth['v'].shape))
            for k in ('psi','omega'):f[k]=sum(w*t[k] for w,t in zip(mix,trains))
            predictions[name]=f
        cases[re]=predictions
    return cases,info

def diagnostics(cases):
    rows=[]
    for re,fields in cases.items():
        truth=fields['CFD']
        for name,f in fields.items():
            e=np.hypot(f['u']-truth['u'],f['v']-truth['v'])
            norm=np.hypot(truth['u'],truth['v'])
            row=dict(Re=re,method=name,relative_velocity_l2=float(np.linalg.norm(e)/np.linalg.norm(norm)),
                     vector_rms_U=float(np.sqrt(np.mean(e**2))),**audit_field(f,re))
            for component in ('u','v'):
                coord=f['y'] if component=='u' else f['x']
                vals=f['u'][:,len(coord)//2] if component=='u' else f['v'][len(coord)//2,:]
                score=ghia.profile_errors(re,component,coord,vals)
                row.update({component+'_'+k:v for k,v in score.items()})
            rows.append(row)
    return rows

def plot_fields(fields,re):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12.4,8.4))
    truth=fields['CFD']
    top=['CFD','Selected NN','Linear interpolation']
    speeds=[np.hypot(fields[name]['u'],fields[name]['v']) for name in top]
    vmax=max(float(a.max()) for a in speeds)
    errors={name:np.hypot(fields[name]['u']-truth['u'],fields[name]['v']-truth['v']) for name in top[1:]}
    emax=max(float(e.max()) for e in errors.values())
    for ax,name,speed in zip(axes[0],top,speeds):
        im=ax.pcolormesh(truth['x'],truth['y'],speed,shading='nearest',cmap='turbo',vmin=0,vmax=vmax,rasterized=True)
        ax.set_title(name)
    fig.colorbar(im,ax=axes[0,:].tolist(),location='right',fraction=.024,pad=.025,label=r'$|\mathbf{u}|/U$')
    f=fields['Selected NN']
    axes[1,0].streamplot(f['x'],f['y'],f['u'],f['v'],color='black',density=2.1,linewidth=.6,arrowsize=.6)
    axes[1,0].set_title('Selected NN streamlines')
    for ax,name in zip(axes[1,1:],top[1:]):
        em=ax.pcolormesh(truth['x'],truth['y'],errors[name],shading='nearest',cmap='turbo',vmin=0,vmax=emax,rasterized=True)
        ax.set_title(name+' error')
    fig.colorbar(em,ax=axes[1,1:].tolist(),location='right',fraction=.04,pad=.045,label=r'$|\mathbf{u}-\mathbf{u}_{CFD}|/U$')
    for ax in axes.ravel():ax.set(xlabel='$x/L$',ylabel='$y/L$',xlim=(0,1),ylim=(0,1),aspect='equal')
    for ax in axes[:,1:].ravel():
        ax.set_ylabel(''); ax.tick_params(labelleft=False)
    # Explicit positions preserve space for both common colour bars.
    for row in range(2):
        for col in range(3):axes[row,col].set_position([.055+col*.29,.56 if row==0 else .09,.245,.355])
    fig.axes[-2].set_position([.93,.56,.015,.355])
    fig.axes[-1].set_position([.93,.09,.015,.355])
    fig.suptitle(f'Held-out Re = {re}; common native 513 x 513 velocity grid',fontsize=13,y=.985)
    return fig

def plot_profiles(fields,re):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.7,7.5))
    truth=fields['CFD'];n=len(truth['x'])//2
    ref=ghia.reference(re)
    for name in ('CFD','Adam','Selected NN','Linear interpolation'):
        f=fields[name];c=COLORS[name];ls='--' if name=='Adam' else '-'
        axes[0,0].plot(f['u'][:,n],f['y'],color=c,ls=ls,label=name,lw=1.6)
        axes[0,1].plot(f['x'],f['v'][n,:],color=c,ls=ls,lw=1.6)
        if name!='CFD':
            axes[1,0].plot(f['y'],f['u'][:,n]-truth['u'][:,n],color=c,ls=ls,lw=1.5)
            axes[1,1].plot(f['x'],f['v'][n,:]-truth['v'][n,:],color=c,ls=ls,lw=1.5)
    axes[0,0].plot(ref['u'],ref['y'],'o',ms=4,mfc='none',mec='black',label='Ghia: printed samples')
    axes[0,1].plot(ref['x'],ref['v'],'o',ms=4,mfc='none',mec='black')
    if (re,'u') in ghia.FLAGGED:
        mask=~ghia.valid_mask(re,'u')
        axes[0,0].plot(ref['u'][mask],ref['y'][mask],'x',color='#b2182b',ms=7,label='Flagged printed entry')
    axes[0,0].set(xlabel='$u/U$ at $x/L=0.5$',ylabel='$y/L$',ylim=(0,1))
    axes[0,1].set(xlabel='$x/L$',ylabel='$v/U$ at $y/L=0.5$',xlim=(0,1))
    axes[1,0].set(xlabel='$y/L$',ylabel=r'$(u_{pred}-u_{CFD})/U$',xlim=(0,1))
    axes[1,1].set(xlabel='$x/L$',ylabel=r'$(v_{pred}-v_{CFD})/U$',xlim=(0,1))
    for ax in axes.ravel():ax.grid(alpha=.18)
    for ax in axes[1]:ax.axhline(0,color='black',lw=.6)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,frameon=False,fontsize=11)
    fig.suptitle(f'Held-out Re = {re}; all neural curves are three-seed means',fontsize=12,y=.985)
    fig.subplots_adjust(left=.085,right=.98,top=.93,bottom=.15,hspace=.35,wspace=.3)
    return fig
