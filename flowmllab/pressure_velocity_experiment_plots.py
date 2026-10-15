"""Physical fields and matched diagnostics for experimental cavity runs."""
import numpy as np
from .pressure_velocity import centerlines,GHIA_X,GHIA_Y
from .pressure_velocity_experiments import REFERENCE

def comparison_figures(rows):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from pathlib import Path
    for name in ['times.ttf','timesbd.ttf','timesi.ttf']:
        path=Path('C:/Windows/Fonts')/name
        if path.exists():font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':11,'mathtext.fontset':'stix','axes.spines.top':False,'axes.spines.right':False})
    reynolds=sorted({int(r['config']['reynolds']) for r in rows})
    colors={'simple':'#205b8f','piso':'#bc5826','pimple':'#298259'}
    output={}
    fig,axes=plt.subplots(len(reynolds),2,figsize=(10,4.3*len(reynolds)),squeeze=False,layout='constrained')
    for i,re in enumerate(reynolds):
        for r in rows:
            cfg=r['config']
            if cfg['reynolds']!=re:continue
            y,u,x,v=centerlines(r)
            label=f"{cfg['method'].upper()}, N={cfg['cells']}, {cfg['convection']}"
            style=':' if cfg['convection']=='upwind' else ('-' if cfg['cells']==32 else '--')
            if not r['converged']:label+=' (not converged)'
            for ax,a,b in [(axes[i,0],u,y),(axes[i,1],x,v)]:
                ax.plot(a,b,style,color=colors[cfg['method']],lw=1.5,label=label)
        axes[i,0].plot(REFERENCE[re]['u'],GHIA_Y,'ko',ms=4,label='Ghia (1982)')
        axes[i,1].plot(GHIA_X,REFERENCE[re]['v'],'ko',ms=4)
        axes[i,0].set(xlabel=r'$u/U$',ylabel=r'$y/L$',title=f'Re={re}: vertical centerline')
        axes[i,1].set(xlabel=r'$x/L$',ylabel=r'$v/U$',title=f'Re={re}: horizontal centerline')
        axes[i,0].legend(fontsize=8,loc='best')
    fig.suptitle('Higher-Re cavity: convergence does not remove mesh error')
    output['centerlines']=fig
    fig,axes=plt.subplots(len(reynolds),2,figsize=(10,3.5*len(reynolds)),squeeze=False,layout='constrained')
    for i,re in enumerate(reynolds):
        for r in rows:
            cfg=r['config']
            if cfg['reynolds']!=re:continue
            h=r['history'];cost=1 if cfg['method']=='simple' else cfg['pressure_correctors']*(cfg['outer_correctors'] if cfg['method']=='pimple' else 1)
            label=f"{cfg['method'].upper()}, N={cfg['cells']}, {cfg['convection']}"
            for ax,j in [(axes[i,0],3),(axes[i,1],4)]:
                ax.semilogy(h[:,0]*cost,np.maximum(h[:,j],1e-17),':' if cfg['convection']=='upwind' else ('-' if cfg['cells']==32 else '--'),color=colors[cfg['method']],label=label)
                ax.set_xlabel('Cumulative pressure solves')
        axes[i,0].axhline(1e-6,color='black',ls=':',lw=1)
        axes[i,0].set(ylabel='Steady momentum defect, Linf',title=f'Re={re}')
        axes[i,1].set(ylabel='Cell divergence, Linf',title=f'Re={re}: mass conservation')
        axes[i,0].legend(fontsize=8)
    fig.suptitle('Coupling work and equation convergence (different meshes solve different equations)')
    output['convergence']=fig
    selected=[]
    for re in reynolds:
        group=[r for r in rows if r['config']['reynolds']==re]
        central=[r for r in group if r['config']['convection']=='central']
        selected.append(max(central or group,key=lambda r:(r['config']['cells'],r['config']['method']=='simple')))
    fig,axes=plt.subplots(len(selected),3,figsize=(13,4.2*len(selected)),squeeze=False,layout='constrained')
    pressure_max=max(np.abs(r['p']).max() for r in selected)
    omega_max=max(np.abs(r['omega']).max() for r in selected)
    for i,r in enumerate(selected):
        x,y=r['x'],r['y'];cfg=r['config']
        data=[np.hypot(r['u'],r['v']),r['p'],r['omega']]
        limits=[(0,1),(-pressure_max,pressure_max),(-omega_max,omega_max)]
        for ax,z,title,cmap,(lo,hi) in zip(axes[i],data,['Speed / U','Zero-mean pressure / rho U^2','Vorticity L / U'],['viridis','RdBu_r','RdBu_r'],limits):
            im=ax.pcolormesh(x,y,z,cmap=cmap,vmin=lo,vmax=hi,shading='nearest',rasterized=True)
            ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xlabel='x / L',ylabel='y / L',title=f"Re={cfg['reynolds']}, N={cfg['cells']}: {title}")
            fig.colorbar(im,ax=ax,shrink=.8)
        axes[i,0].streamplot(x,y,r['u'],r['v'],density=1.1,color='white',linewidth=.55,arrowsize=.7)
    fig.suptitle('Finest retained central solution per Re; shared physical color scales')
    output['fields']=fig
    return output
