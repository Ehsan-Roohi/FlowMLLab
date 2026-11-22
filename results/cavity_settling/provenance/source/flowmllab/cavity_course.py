"""Shared seven-Re teaching diagnostics. A reference table is not a CFD field."""
from pathlib import Path
import json
import numpy as np
from .ghia import REYNOLDS, GHIA, FLAGGED, profile_errors, case_roles


def coverage(root):
    """Report actual retained full-field coverage, without promoting it to truth."""
    root=Path(root); rows=[]
    for re in REYNOLDS:
        path=root/f'results/week01_2_formulations/streamfunction-vorticity-re{re}-n128.npz'
        if re==100:
            path=root/'results/week01_2_pressure_velocity/simple-re100-n64.npz'
        new_field=root/f'results/cavity_highre_ghia/n512/re{re}.npz'
        if new_field.exists():path=new_field
        rows.append({'Re':re,'reference_samples':34,'flagged_samples':sum(k[0]==re for k in FLAGGED),
                     'profile_study_role':case_roles()[re],
                     'retained_field':path.relative_to(root).as_posix() if path.exists() else None,
                     'scope':'retained evidence; inspect its own mesh/steady gate' if path.exists()
                             else 'no retained full field; benchmark table only'})
    return rows


def plot_reference(reynolds):
    import matplotlib.pyplot as plt
    from .ghia import valid_mask
    ref=GHIA[reynolds]; fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for ax,c in zip(axs,('u','v')):
        t=ref['y' if c=='u' else 'x'];keep=valid_mask(reynolds,c)
        ax.plot(t[keep],ref[c][keep],'ko',label='Published Ghia samples')
        if (~keep).any(): ax.plot(t[~keep],ref[c][~keep],'rx',ms=9,label='Flagged printed entry')
        ax.set(xlabel='y/L' if c=='u' else 'x/L',ylabel=f'{c}/U',xlim=(0,1));ax.grid(alpha=.2)
    handles,labels=[],[]
    for ax in axs:
        hs,ls=ax.get_legend_handles_labels()
        for h,l in zip(hs,ls):
            if l not in labels:handles.append(h);labels.append(l)
    fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.subplots_adjust(top=.80,bottom=.17,wspace=.3)
    return fig


def compare_field(reynolds,x,y,u,v):
    """Evaluate centre lines using interpolation in BOTH coordinates.

    Native field shape is [y,x]. Interior cell-centre arrays must first receive
    physically correct boundary values; extrapolating them is not allowed.
    """
    x,y,u,v=map(np.asarray,(x,y,u,v))
    if u.shape!=(len(y),len(x)) or v.shape!=u.shape:
        raise ValueError('Expected [y,x] field arrays')
    if not (np.all(np.diff(x)>0) and np.all(np.diff(y)>0)):
        raise ValueError('Coordinates must increase')
    if x[0]>0 or y[0]>0 or x[-1]<1 or y[-1]<1:
        raise ValueError('Supply the true wall values; no extrapolation')
    uc=np.array([np.interp(.5,x,row) for row in u])
    vc=np.array([np.interp(.5,y,col) for col in v.T])
    return {c:profile_errors(reynolds,c,t,a) for c,t,a in [('u',y,uc),('v',x,vc)]}


def verify_frozen_models(root):
    import hashlib
    folder=Path(root)/'results/cavity_ghia_allre'
    info=json.loads((folder/'frozen_selection.json').read_text())
    for name,expected in info['models_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen model changed: {name}')
    return info


def load_frozen_model(root,name,seed):
    from .ghia_surrogate import ModelConfig
    folder=Path(root)/'results/cavity_ghia_allre'
    info=verify_frozen_models(root)
    config=next(c for c in info['configs'] if c['name']==name)
    with np.load(folder/f'{name}_seed{seed}.npz',allow_pickle=False) as z:
        return {'theta':z['theta'],'shapes':[tuple(s) for s in z['shapes']], 'config':ModelConfig(**config)}
