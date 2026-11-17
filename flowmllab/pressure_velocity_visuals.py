"""RGB cavity comparisons drawn from retained or freshly computed native fields.

Only presentation is changed: no filtering, rescaling of field values, pressure
regauging, or CFD solve takes place here. Limits contain all supplied samples.
"""
import textwrap
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm

SPECTRUM = LinearSegmentedColormap.from_list('flowml_clear_rgb', [
    (0.00, '#1943df'), (0.18, '#0087fa'), (0.35, '#00d4e5'),
    (0.50, '#00c86c'), (0.67, '#dce900'), (0.82, '#ff9700'),
    (1.00, '#e91c24')], N=512)
METHOD_NAMES = {'simple':'SIMPLE', 'piso':'PISO', 'pimple':'PIMPLE',
                'streamfunction-vorticity':r'$\psi$-$\omega$ (FD)'}


def values_for(run, key):
    return np.hypot(run['u'],run['v']) if key=='speed' else np.asarray(run[key])


def color_normalization(key, values):
    if not values or not all(np.isfinite(v).all() for v in values):
        raise ValueError('Finite, nonempty native fields are required')
    bound=max(float(np.max(np.abs(v))) for v in values)
    if key=='speed':
        return Normalize(0,max(1.,np.ceil(bound*10)/10))
    # Rounded physical endpoints keep the legend readable without clipping.
    unit=.1 if key=='p' else 50.
    bound=max(unit, np.ceil(bound/unit)*unit)
    return SymLogNorm(linthresh=.01 if key=='p' else 1.,linscale=1.,
                      vmin=-bound,vmax=bound,base=10)


def streamlines(ax, run):
    """One moderately spaced integration pass, with small direction arrows."""
    ax.streamplot(run['x'],run['y'],run['u'],run['v'],density=1.05,
        color='#101a24',linewidth=.42,arrowsize=.45,minlength=.045,
        maxlength=4.,zorder=3)


def comparison_figure(runs,key,*,norm=None,labels=None,columns=2,
                      condition=None):
    """Return a shared-scale comparison; native data are never modified."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import FuncFormatter
    values=[values_for(r,key) for r in runs]
    norm=norm or color_normalization(key,values)
    if any(float(v.min())<norm.vmin or float(v.max())>norm.vmax for v in values):
        raise ValueError('Color limits would clip a supplied extremum')
    rows=int(np.ceil(len(runs)/columns))
    style={'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
           'mathtext.fontset':'stix','font.size':11,'axes.titlesize':12.5,
           'axes.labelsize':12,'xtick.labelsize':10.5,'ytick.labelsize':10.5,
           'figure.facecolor':'white','axes.facecolor':'white',
           'axes.linewidth':.8,'figure.constrained_layout.w_pad':.045,
           'figure.constrained_layout.h_pad':.045}
    with plt.rc_context(style):
        fig,grid=plt.subplots(rows,columns,squeeze=False,figsize=(6.6,3.05*rows+.15),
                              layout='constrained')
        levels=norm.inverse(np.linspace(0,1,129))
        for i,(ax,run,z) in enumerate(zip(grid.flat,runs,values)):
            x=np.asarray(run['x']);y=np.asarray(run['y'])
            # Nearest values cover half-cell edge bands; interior contours use
            # linear interpolation between native samples, with no smoothing.
            ax.pcolormesh(x,y,z,cmap=SPECTRUM,norm=norm,shading='nearest',zorder=-3)
            ax.contourf(x,y,z,levels=levels,cmap=SPECTRUM,norm=norm,
                        antialiased=False,zorder=-2)
            ax.set_rasterization_zorder(-1)
            streamlines(ax,run)
            label=(labels[i] if labels else METHOD_NAMES.get(run['config']['method'],run['config']['method']))
            if labels and '\n' not in label:
                label='\n'.join(textwrap.wrap(label,24,break_long_words=False))
            ax.set_title(f'({chr(97+i)}) '+label,pad=7)
            ax.set(aspect='equal',xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
            ax.tick_params(direction='out',length=3,width=.7,pad=3)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_:f'{y:g}'))
            if i//columns==rows-1:ax.set_xlabel(r'$x/L$')
            if i%columns==0:ax.set_ylabel(r'$y/L$')
            else:ax.tick_params(labelleft=False)
            for spine in ax.spines.values():spine.set_visible(True)
        for ax in list(grid.flat)[len(runs):]:ax.set_axis_off()
        if condition is None:
            c=runs[0]['config'];condition=rf'$Re={c["reynolds"]:g},\quad h/L=1/{c["cells"]}$'
        fig.suptitle(condition,fontsize=12.5)
        bar=fig.colorbar(ScalarMappable(norm=norm,cmap=SPECTRUM),
            ax=list(grid.flat),location='bottom',fraction=.055,shrink=.88,aspect=36,pad=.045)
        labels_by_key={'speed':r'$|\mathbf{u}|/U_{\mathrm{lid}}$',
                       'p':r'$p/(\rho U_{\mathrm{lid}}^2)$  (zero mean; symmetric log)',
                       'omega':r'$\omega L/U_{\mathrm{lid}}$  (symmetric log)'}
        bar.set_label(labels_by_key[key],fontsize=11.5,labelpad=5)
        if key=='speed':ticks=np.linspace(0,norm.vmax,5)
        elif key=='p':ticks=[norm.vmin,-.1,-.01,0,.01,.1,norm.vmax]
        else:ticks=[norm.vmin,-20,-2,0,2,20,norm.vmax]
        bar.set_ticks(sorted(set(t for t in ticks if norm.vmin<=t<=norm.vmax)))
        bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
        bar.ax.minorticks_off();bar.ax.tick_params(labelsize=9.5,length=3,width=.6,pad=3)
        bar.outline.set_linewidth(.6)
    return fig


def experiment_field_figures(runs):
    """Split interactive field comparisons by Reynolds number and quantity."""
    figures={}
    for key in ['speed','p','omega']:
        norm=color_normalization(key,[values_for(r,key) for r in runs])
        for re in sorted({r['config']['reynolds'] for r in runs}):
            group=[r for r in runs if r['config']['reynolds']==re]
            labels=[]
            for r in group:
                c=r['config']
                label=METHOD_NAMES.get(c['method'],c['method'])
                label+=f"\nh/L=1/{c['cells']}, {c.get('convection','central')}"
                if not r.get('converged',False):label+='\nNOT CONVERGED'
                labels.append(label)
            figures[f'{key}_re{re:g}']=comparison_figure(group,key,norm=norm,
                labels=labels,condition=rf'$Re={re:g}$; computed fields')
    return figures
