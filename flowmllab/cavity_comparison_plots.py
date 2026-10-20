"""Interactive four-formulation plots with distinct native residual axes."""
import numpy as np
from .pressure_velocity import centerlines,GHIA_Y,GHIA_X
from .pressure_velocity_experiments import REFERENCE
from .cavity_visuals import RGB,field_normalization,field_scale_note,flow_structure

def all_formulation_figures(rows):
    import matplotlib.pyplot as plt
    colors={'simple':'#1748b7','piso':'#e23028','pimple':'#009f59','streamfunction-vorticity':'#703ac4'}
    re_values=sorted({int(r['config']['reynolds']) for r in rows})
    output={}
    fig,axes=plt.subplots(len(re_values),2,figsize=(10,4*len(re_values)),squeeze=False,layout='constrained')
    for i,re in enumerate(re_values):
        for r in rows:
            c=r['config'];m=c['method']
            if c['reynolds']!=re:continue
            y,u,x,v=centerlines(r);label=m+(' [NOT STEADY]' if not r['converged'] else '')
            axes[i,0].plot(u,y,color=colors[m],label=label)
            axes[i,1].plot(x,v,color=colors[m])
        axes[i,0].plot(REFERENCE[re]['u'],GHIA_Y,'ko',ms=3,label='Ghia reference')
        axes[i,1].plot(GHIA_X,REFERENCE[re]['v'],'ko',ms=3)
        axes[i,0].set(xlabel='u / U',ylabel='y / L',title=f'Re={re}: vertical centerline')
        axes[i,1].set(xlabel='x / L',ylabel='v / U',title=f'Re={re}: horizontal centerline')
        axes[i,0].legend(fontsize=8)
    output['centerlines']=fig
    fig,axes=plt.subplots(len(re_values),3,figsize=(13,3.6*len(re_values)),squeeze=False,layout='constrained')
    for i,re in enumerate(re_values):
        group=[r for r in rows if r['config']['reynolds']==re]
        for r in group:
            m=r['config']['method']
            if m=='streamfunction-vorticity':
                axes[i,1].semilogy(r['residual_steps'],r['history_rate'],color=colors[m],label='FD relative update / dt')
            else:axes[i,0].semilogy(r['history'][:,0],r['history'][:,3],color=colors[m],label=m.upper())
        axes[i,0].set(xlabel='FV coupling iterations',ylabel='Momentum defect, Linf',title=f'Re={re}: FV equations')
        axes[i,1].set(xlabel='FD explicit steps',ylabel='Relative vorticity update / dt',title=f'Re={re}: independent FD')
        for ax in axes[i,:2]:
            if ax.lines:ax.legend(fontsize=8)
        values=[r['runtime_seconds'] for r in group]
        axes[i,2].bar(range(len(group)),values,color=[colors[r['config']['method']] for r in group])
        axes[i,2].set_xticks(range(len(group)),[r['config']['method'].replace('streamfunction-vorticity','psi-omega') for r in group],rotation=25,ha='right')
        axes[i,2].set(ylabel='Solver-call wall time (s)',title='All computed methods')
        axes[i,2].margins(y=.2)
        for j,value in enumerate(values):axes[i,2].text(j,value,f'{value:.2f}s',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Separate native convergence measures; timings depend on settings and host')
    output['convergence']=fig
    fig,axes=plt.subplots(len(rows),3,figsize=(12,3.8*len(rows)),squeeze=False,layout='constrained')
    keys=['speed','p','omega'];titles=['Speed / U','Pressure / rho U^2','Vorticity L / U']
    for j,(key,title) in enumerate(zip(keys,titles)):
        values=[np.hypot(r['u'],r['v']) if key=='speed' else r[key] for r in rows]
        norm=field_normalization(key,values)
        for i,(r,value) in enumerate(zip(rows,values)):
            ax=axes[i,j];im=ax.pcolormesh(r['x'],r['y'],value,cmap=RGB,norm=norm,shading='nearest')
            flow_structure(ax,r['x'],r['y'],r['u'],r['v'])
            c=r['config'];ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xlabel='x/L',ylabel='y/L',title=f"Re={c['reynolds']}: {c['method']}")
        fig.colorbar(im,ax=list(axes[:,j]),label=title,shrink=.8)
    fig.suptitle('Computed fields: all requested methods; signed fields use symmetric-log color scales')
    output['fields']=fig
    return output
