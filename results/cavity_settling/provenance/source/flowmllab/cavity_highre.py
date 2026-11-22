"""Read and independently audit newly computed high-Re cavity CFD evidence."""
from pathlib import Path
import hashlib, json
import numpy as np
from . import ghia

REYNOLDS=(3200,5000,7500,10000)
GRIDS=(128,256,512)
COLORS=('#8c6d31','#0072B2','#D55E00')


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_field(field,reynolds):
    """Recompute equations from native arrays, without the solver's matrices."""
    x,y=field['x'],field['y'];p,w,u,v=(field[k] for k in ('psi','omega','u','v'))
    n=len(x)-1; h=1/n
    if p.shape!=(n+1,n+1) or any(a.shape!=p.shape for a in (w,u,v)):
        raise ValueError('Field dimensions do not match the declared square grid')
    if not all(np.isfinite(a).all() for a in field.values()): raise ValueError('Nonfinite field')
    np.testing.assert_allclose(x,np.linspace(0,1,n+1),atol=1e-14,rtol=0)
    np.testing.assert_allclose(y,x,atol=1e-14,rtol=0)
    lap=lambda a:(a[1:-1,2:]+a[1:-1,:-2]+a[2:,1:-1]+a[:-2,1:-1]-4*a[1:-1,1:-1])/h**2
    psi_defect=lap(p)+w[1:-1,1:-1]
    rhs=lap(w)/reynolds-u[1:-1,1:-1]*(w[1:-1,2:]-w[1:-1,:-2])/(2*h)-v[1:-1,1:-1]*(w[2:,1:-1]-w[:-2,1:-1])/(2*h)
    div=(u[1:-1,2:]-u[1:-1,:-2]+v[2:,1:-1]-v[:-2,1:-1])/(2*h)
    wall_defect=max(np.abs(p[[0,-1],:]).max(),np.abs(p[:,[0,-1]]).max(),
        np.abs(u[0,:]).max(),np.abs(u[:,[0,-1]]).max(),np.abs(u[-1,1:-1]-1).max(),
        np.abs(v[[0,-1],:]).max(),np.abs(v[:,[0,-1]]).max(),
        np.abs(w[0,1:-1]+2*p[1,1:-1]/h**2).max(),
        np.abs(w[-1,1:-1]+2*p[-2,1:-1]/h**2+2/h).max(),
        np.abs(w[1:-1,0]+2*p[1:-1,1]/h**2).max(),
        np.abs(w[1:-1,-1]+2*p[1:-1,-2]/h**2).max())
    velocity_defect=max(np.abs(u[1:-1,1:-1]-(p[2:,1:-1]-p[:-2,1:-1])/(2*h)).max(),
                         np.abs(v[1:-1,1:-1]+(p[1:-1,2:]-p[1:-1,:-2])/(2*h)).max())
    return dict(poisson_linf=float(np.abs(psi_defect).max()),
                steady_vorticity_linf=float(np.abs(rhs).max()),
                continuity_linf=float(np.abs(div).max()),wall_linf=float(wall_defect),
                streamfunction_velocity_linf=float(velocity_defect))


def load_cases(root,grids=GRIDS,reynolds=REYNOLDS):
    root=Path(root);out=root/'results/cavity_highre_ghia';cases=[]
    for re in reynolds:
        for n in grids:
            path=out/f'n{n}'/f're{re}.npz';meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if file_hash(path)!=meta['field_sha256']: raise ValueError(f'Field SHA256 mismatch: {path}')
            if meta['cells']!=n or meta['reynolds']!=re or meta['ghia_used_during_solution']:
                raise ValueError('Run identity or no-fitting contract failed')
            with np.load(path,allow_pickle=False) as z: field={k:z[k] for k in z.files}
            audit=audit_field(field,re)
            if not meta['converged'] or max(audit.values())>=1e-6:
                raise ValueError(f'Independent equation/wall gate failed: {path}: {audit}')
            profiles={'u':(field['y'],field['u'][:,n//2]),'v':(field['x'],field['v'][n//2,:])}
            scores={c:ghia.profile_errors(re,c,*profiles[c]) for c in ('u','v')} if re in ghia.REYNOLDS else {}
            cases.append(dict(Re=re,cells=n,field=field,profiles=profiles,scores=scores,audit=audit,
                              metadata=meta,source=path.relative_to(root).as_posix(),sha256=meta['field_sha256']))
    return cases


def score_rows(cases):
    rows=[]
    for c in cases:
        row=dict(Re=c['Re'],cells=c['cells'],**c['audit'],source=c['source'],sha256=c['sha256'])
        for component in c['scores']:
            row.update({f'{component}_{k}':v for k,v in c['scores'][component].items()})
        rows.append(row)
    return rows


def mesh_rows(cases):
    rows=[];t=np.linspace(0,1,2049)
    for re in sorted({c['Re'] for c in cases}):
        selected=sorted((c for c in cases if c['Re']==re),key=lambda c:c['cells'])
        fine=selected[-1]
        for c in selected[:-1]:
            row=dict(Re=re,cells=c['cells'],reference_cells=fine['cells'])
            stride=fine['cells']//c['cells']
            if stride*c['cells']!=fine['cells']:raise ValueError('Expected nested native grids')
            coarse_state=np.stack([c['field'][k] for k in ('u','v')])
            reference_state=np.stack([fine['field'][k][::stride,::stride] for k in ('u','v')])
            delta=coarse_state-reference_state
            row.update(velocity_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference_state)),
                       velocity_vector_rms_U=float(np.sqrt(np.mean(np.sum(delta**2,axis=0)))),
                       velocity_component_linf_U=float(np.abs(delta).max()))
            for component in ('u','v'):
                a=np.interp(t,*c['profiles'][component]);b=np.interp(t,*fine['profiles'][component])
                row[f'{component}_rmse_U']=float(np.sqrt(np.mean((a-b)**2)))
                row[f'{component}_linf_U']=float(np.max(np.abs(a-b)))
            rows.append(row)
    return rows


def plot_case(cases,re):
    import matplotlib.pyplot as plt
    subset=sorted((c for c in cases if c['Re']==re),key=lambda c:c['cells'])
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.4))
    f=subset[-1]['field']
    axes[0].streamplot(f['x'],f['y'],f['u'],f['v'],color='black',density=2.1,linewidth=.65,arrowsize=.65)
    axes[0].set(xlabel='$x/L$',ylabel='$y/L$',xlim=(0,1),ylim=(0,1),aspect='equal')
    axes[0].set_title(f'Re = {re}: {subset[-1]["cells"]} cells')
    for case,col in zip(subset,COLORS):
        y,u=case['profiles']['u'];x,v=case['profiles']['v']
        axes[1].plot(u,y,color=col,lw=1.5,label=f'{case["cells"]} cells')
        axes[2].plot(x,v,color=col,lw=1.5)
    ref=ghia.reference(re)
    axes[1].plot(ref['u'],ref['y'],'o',ms=4,mfc='none',mec='black',label='Ghia: printed samples')
    axes[2].plot(ref['x'],ref['v'],'o',ms=4,mfc='none',mec='black')
    if (re,'u') in ghia.FLAGGED:
        mask=~ghia.valid_mask(re,'u')
        axes[1].plot(ref['u'][mask],ref['y'][mask],'x',ms=7,color='#b2182b',label='Flagged printed entry')
    axes[1].set(xlabel='$u/U$ at $x/L=0.5$',ylabel='$y/L$',ylim=(0,1),xlim=(-1,1.05))
    axes[2].set(xlabel='$x/L$',ylabel='$v/U$ at $y/L=0.5$',xlim=(0,1),ylim=(-.7,.55))
    for ax in axes[1:]: ax.grid(alpha=.18)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=len(labels),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.055,right=.99,top=.87,bottom=.23,wspace=.34)
    return fig


def plot_overview(cases):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(13.2,7.0),sharex='row',sharey='row')
    for col,re in enumerate(REYNOLDS):
        selected=sorted((c for c in cases if c['Re']==re),key=lambda c:c['cells'])
        for case,color in zip(selected,COLORS):
            y,u=case['profiles']['u'];x,v=case['profiles']['v']
            axes[0,col].plot(u,y,color=color,lw=1.5,label=f'{case["cells"]} cells')
            axes[1,col].plot(x,v,color=color,lw=1.5)
        ref=ghia.reference(re)
        axes[0,col].plot(ref['u'],ref['y'],'o',ms=4,mfc='none',mec='black',label='Ghia: printed samples')
        axes[1,col].plot(ref['x'],ref['v'],'o',ms=4,mfc='none',mec='black')
        if (re,'u') in ghia.FLAGGED:
            flag=~ghia.valid_mask(re,'u')
            axes[0,col].plot(ref['u'][flag],ref['y'][flag],'x',ms=7,color='#b2182b',label='Flagged printed entry')
        axes[0,col].set(title=f'Re = {re}',xlabel='$u/U$ at $x/L=0.5$',xlim=(-1,1.05),ylim=(0,1))
        axes[1,col].set(xlabel='$x/L$',xlim=(0,1),ylim=(-.7,.55))
        axes[0,col].set_xticks([-1,0,1]);axes[1,col].set_xticks([0,.5,1])
        for ax in axes[:,col]:ax.grid(alpha=.18)
    axes[0,0].set_ylabel('$y/L$');axes[1,0].set_ylabel('$v/U$ at $y/L=0.5$')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=5,frameon=False,fontsize=10)
    fig.subplots_adjust(left=.095,right=.975,top=.94,bottom=.15,hspace=.33,wspace=.22)
    return fig


def plot_streamlines(cases):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13.2,3.8),sharex=True,sharey=True)
    for ax,re in zip(axes,REYNOLDS):
        case=max((c for c in cases if c['Re']==re),key=lambda c:c['cells'])
        f=case['field']
        ax.streamplot(f['x'],f['y'],f['u'],f['v'],color='black',density=2.5,linewidth=.6,arrowsize=.6)
        ax.set(title=f'Re = {re}',xlabel='$x/L$',xlim=(0,1),ylim=(0,1),aspect='equal')
    axes[0].set_ylabel('$y/L$')
    for ax in axes:ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    fig.subplots_adjust(left=.07,right=.975,bottom=.18,top=.88,wspace=.2)
    return fig


def load_temporal(root):
    """Verify matched dt pairs without claiming long-time stability."""
    base=Path(root)/'results/cavity_highre_ghia/temporal'
    rows=[];pairs=[]
    for re in REYNOLDS:
        runs=[]
        for dt in (.002,.001):
            folder=base/f're{re}-dt{dt:g}'
            meta=json.loads((folder/'audit.json').read_text(encoding='utf-8'))
            path=folder/'final.npz'
            if file_hash(path)!=meta['final_sha256']:raise ValueError('Temporal final hash mismatch')
            cfg=meta['config']
            if cfg['reynolds']!=re or cfg['dt']!=dt or cfg['cells']!=256 or cfg['duration']!=10:
                raise ValueError('Temporal configuration mismatch')
            times=np.array([r['time'] for r in meta['observations']])
            np.testing.assert_allclose(times,np.arange(21)*.5,atol=1e-12,rtol=0)
            field=dict(np.load(path,allow_pickle=False))
            if not all(np.isfinite(a).all() for a in field.values()):raise ValueError('Nonfinite temporal output')
            runs.append(dict(metadata=meta,field=field))
        coarse,fine=runs
        if coarse['metadata']['config']['initial_omega_sha256']!=fine['metadata']['config']['initial_omega_sha256']:
            raise ValueError('Time-step pair does not share its complete initial state')
        for run in runs:
            expected=Path(root)/'results/cavity_highre_ghia/n256'/f're{re}.npz'
            if file_hash(expected)!=run['metadata']['config']['steady_field_sha256']:
                raise ValueError('Temporal reference differs from retained steady CFD')
        delta=[coarse['field'][k]-fine['field'][k] for k in ('u','v')]
        rms=float(np.sqrt(np.mean(delta[0]**2+delta[1]**2)))
        obs=fine['metadata']['observations']
        relative=rms/max(obs[-1]['velocity_departure_rms_U'],1e-30)
        rows.append(dict(Re=re,cells=256,time_end=10,dt_coarse=.002,dt_fine=.001,
                         paired_velocity_rms_difference_U=rms,
                         paired_velocity_linf_difference_U=float(max(np.abs(a).max() for a in delta)),
                         relative_perturbation_difference=relative,
                         initial_departure_rms_U=obs[0]['velocity_departure_rms_U'],
                         final_departure_rms_U=obs[-1]['velocity_departure_rms_U'],
                         finite_horizon_growth=obs[-1]['velocity_departure_rms_U']/obs[0]['velocity_departure_rms_U'],
                         finite_horizon_time_check_pass=bool(rms<1e-6 and relative<.01)))
        pairs.append(dict(Re=re,runs=runs))
    return rows,pairs


def plot_temporal(pairs):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    colors=('#0072B2','#D55E00','#009E73','#CC79A7')
    for pair,color in zip(pairs,colors):
        coarse,fine=pair['runs'];a=coarse['metadata']['observations'];b=fine['metadata']['observations']
        t=[r['time'] for r in b]
        axes[0].plot(t,[r['velocity_departure_rms_U'] for r in b],color=color,label=f'Re = {pair["Re"]}')
        axes[0].plot(t,[r['velocity_departure_rms_U'] for r in a],'.',ms=4,color=color)
        axes[1].plot(t,[r['u_center']-b[0]['u_center'] for r in b],color=color)
        axes[1].plot(t,[r['u_center']-a[0]['u_center'] for r in a],'.',ms=4,color=color)
    axes[0].set(xlabel='$tU/L$',ylabel='Velocity departure from steady root (RMS / U)')
    axes[1].set(xlabel='$tU/L$',ylabel='Change in centre-point $u/U$ from initial value')
    for ax in axes:
        ax.grid(alpha=.18);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    handles,labels=axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([],[],color='black'),Line2D([],[],color='black',marker='.',ls='none')]
    labels += ['dt = 0.001','dt = 0.002']
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=6,frameon=False,fontsize=9)
    fig.subplots_adjust(left=.09,right=.985,bottom=.25,top=.9,wspace=.35)
    return fig
