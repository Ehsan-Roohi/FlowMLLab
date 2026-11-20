"""Print-sized physical plots from the executed Lab 2 native prediction archive."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'qa'))
from lecture_typography import configure_matplotlib
if os.name == 'nt': os.environ.setdefault('FLOWMLLAB_TNR_DIR','C:/Windows/Fonts')

def build(scalar_path=None):
    configure_matplotlib()
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'legend.fontsize':9,
                         'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9})
    source=ROOT/'results/week04_lab2/native_test_predictions.npz'
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source,allow_pickle=False) as f: d={k:f[k].copy() for k in f.files}
    if hashlib.sha256(source.read_bytes()).hexdigest()!=before:
        raise RuntimeError('Lab2 prediction archive changed during figure generation')
    out=ROOT/'results/week04_lecture';out.mkdir(parents=True,exist_ok=True)
    x,y=d['x'],d['y'];X,Y=np.meshgrid(x,y)
    methods=[('CFD','black','-'),('Interpolation','#777777','--'),
             ('Coordinate_DNN','#c000b0','-.'),('Boundary-aware_DNN','#00767e',':')]
    re=275
    fields={key:(d[f'Re{re}_{key}_u'],d[f'Re{re}_{key}_v']) for key,_,_ in methods}
    u,v=fields['CFD'];speed=np.hypot(u,v)
    err={key:np.hypot(a-u,b-v) for key,(a,b) in fields.items() if key!='CFD'}
    fig,axes=plt.subplots(2,2,figsize=(7.3,7.1))
    fig.subplots_adjust(left=.085,right=.97,top=.95,bottom=.25,wspace=.35,hspace=.34)
    allmax=max(float(np.hypot(*ab).max()) for ab in fields.values())
    levels=np.linspace(0,allmax,19)
    ax=axes[0,0]
    filled=ax.contourf(X,Y,speed,levels=levels,cmap='jet')
    common=[.05,.10,.20,.40,.60,.80,.95]
    for key,color,style in methods:
        ax.contour(X,Y,np.hypot(*fields[key]),levels=common,colors=color,linestyles=style,linewidths=.8)
    ax.set_title('CFD and model speed contours')
    cax=fig.add_axes([.10,.145,.34,.015]);cb=fig.colorbar(filled,cax=cax,orientation='horizontal',ticks=np.linspace(0,allmax,5))
    cb.ax.tick_params(labelsize=7);cb.set_label(r'$|\mathbf{u}|/U$',fontsize=9)
    emax=max(float(e.max()) for e in err.values())
    for ax,(key,_,_) in zip([axes[0,1],axes[1,0],axes[1,1]],methods[1:]):
        error=ax.contourf(X,Y,err[key],levels=np.linspace(0,emax,17),cmap='jet')
        ax.set_title(key.replace('_',' ')+' error')
    ecax=fig.add_axes([.60,.145,.34,.015]);ecb=fig.colorbar(error,cax=ecax,orientation='horizontal',ticks=np.linspace(0,emax,5))
    ecb.ax.tick_params(labelsize=7)
    ecb.set_label(r'$\|\widehat{\mathbf{u}}-\mathbf{u}_{CFD}\|/U$ (linear)',fontsize=9)
    for ax in axes.flat:
        ax.set(xlabel=r'$x/L$',ylabel=r'$y/L$',xlim=(0,1),ylim=(0,1),aspect='equal')
    handles=[Line2D([],[],color=c,ls=s,lw=1.3,label=k.replace('_',' ')) for k,c,s in methods]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
    fig.savefig(out/'lab2_re275_fields.png',dpi=220)
    fig.savefig(out/'lab2_re275_fields.pdf')
    plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(7.0,6.0))
    fig.subplots_adjust(left=.10,right=.97,top=.94,bottom=.16,wspace=.31,hspace=.33)
    j=int(np.argmin(abs(y-.5)));i=int(np.argmin(abs(x-.5)))
    for key,color,style in methods:
        a,b=fields[key]
        axs[0,0].plot(a[:,i],y,c=color,ls=style,lw=1.3,label=key.replace('_',' '))
        axs[0,1].plot(x,b[j,:],c=color,ls=style,lw=1.3)
        if key!='CFD':
            axs[1,0].plot(y,a[:,i]-u[:,i],c=color,ls=style,lw=1.3)
            axs[1,1].plot(x,b[j,:]-v[j,:],c=color,ls=style,lw=1.3)
    axs[0,0].set(xlabel=r'$u/U$',ylabel=r'$y/L$',title=r'$x/L=0.5$')
    axs[0,1].set(xlabel=r'$x/L$',ylabel=r'$v/U$',title=r'$y/L=0.5$')
    axs[1,0].set(xlabel=r'$y/L$',ylabel=r'$(\widehat u-u_{CFD})/U$')
    axs[1,1].set(xlabel=r'$x/L$',ylabel=r'$(\widehat v-v_{CFD})/U$')
    from matplotlib.ticker import MaxNLocator
    for ax in axs[1]:
        ax.axhline(0,color='black',lw=.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    for ax in axs.flat:ax.grid(alpha=.18)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False)
    fig.savefig(out/'lab2_re275_profiles.png',dpi=220)
    fig.savefig(out/'lab2_re275_profiles.pdf')
    plt.close(fig)
    scalar_source=Path(scalar_path) if scalar_path is not None else ROOT/'results/week04_lab2/scalar_predictions.npz'
    with np.load(scalar_source,allow_pickle=False) as f: scalar={k:f[k].copy() for k in f.files}
    fig,axs=plt.subplots(3,1,figsize=(7.0,6.7),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,top=.97,bottom=.17,hspace=.26)
    labels=[r'$\psi_{\min}/(UL)$',r'$x_v/L$',r'$y_v/L$']
    dense=scalar['dense_Re'];nn=scalar['neural_dense_by_seed']
    for j,ax in enumerate(axs):
        ax.plot(dense,scalar['interpolation_dense'][:,j],color='#777777',ls='--',label='Interpolation')
        ax.plot(dense,scalar['polynomial_dense'][:,j],color='#0071bd',ls='-.',label='Quadratic')
        ax.plot(dense,nn[:,:,j].mean(axis=0),color='#9f008f',label='Scalar DNN mean')
        ax.fill_between(dense,nn[:,:,j].min(axis=0),nn[:,:,j].max(axis=0),color='#9f008f',alpha=.12,label='Three-seed range')
        for role,marker,color in [('fit','o','black'),('validation','s','#8b5b00'),('test','x','#be0000')]:
            selected=np.array([str(s)==role for s in scalar['role']])
            if selected.any():ax.scatter(scalar['Re'][selected],scalar['reference'][selected,j],marker=marker,color=color,s=21,label=role.capitalize(),zorder=5)
        ax.set_ylabel(labels[j]);ax.grid(alpha=.18)
    axs[-1].set_xlabel('Reynolds number')
    hs,ls=axs[0].get_legend_handles_labels()
    fig.legend(hs,ls,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False)
    fig.savefig(out/'lab2_scalar_trends.png',dpi=220)
    fig.savefig(out/'lab2_scalar_trends.pdf');plt.close(fig)
    record={'source':source.relative_to(ROOT).as_posix(),'source_sha256':before,'Re':re,'native_grid':[len(y),len(x)],
            'methods':[k for k,_,_ in methods], 'display':'Native data; common contour levels; full linear vector-error scale; no smoothing',
            'scalar_source':scalar_source.relative_to(ROOT).as_posix(),
            'scalar_source_sha256':hashlib.sha256(scalar_source.read_bytes()).hexdigest(),
            'figure_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('lab2_*')}}
    (ROOT/'qa/week04_lecture_figures.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    return record

if __name__=='__main__': print(json.dumps(build(),indent=2))
