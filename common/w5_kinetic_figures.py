"""Native-field comparisons for the existing finite-window kinetic teaching project."""
import numpy as np


def plot_kinetic_evidence(data, index, predictions):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    x,y=np.asarray(data['x']),np.asarray(data['y'])
    X,Y=np.meshgrid(x,y);Uw=float(data['Uwall'][index])
    u,v,T=data['u'][index]/Uw,data['v'][index]/Uw,data['T'][index]
    figures=[]
    temperature_levels=np.linspace(min(float(T.min()),*[float(a[2].min())for a in predictions.values()]),max(float(T.max()),*[float(a[2].max())for a in predictions.values()]),31)
    error_max=max(float(np.hypot(a[0]-u,a[1]-v).max())for a in predictions.values())
    speed=np.hypot(u,v);speed_levels=np.linspace(0,max(float(speed.max()),1e-12),31)
    for name,(up,vp,Tp) in predictions.items():
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        im=axes[0,0].contourf(X,Y,speed,levels=speed_levels,cmap='turbo')
        axes[0,0].streamplot(x,y,u,v,color='black',density=1.0,linewidth=.65,arrowsize=.6)
        stream=axes[0,0].streamplot(x,y,up,vp,color='magenta',density=1.0,linewidth=.8,arrowsize=.6)
        stream.lines.set_linestyle('--')
        fig.colorbar(im,ax=axes[0,0],label='Speed / Uwall')
        signed_temperature=np.array([np.interp(.5,x,row)for row in (Tp-T)])
        axes[0,1].plot(signed_temperature,y,color='magenta',ls='--')
        axes[0,1].axvline(0,color='black',lw=.8)
        axes[0,1].set(xlabel='(Prediction - reference) / Twall',ylabel='y/L',title='Signed temperature discrepancy')
        axes[0,1].grid(alpha=.2)
        err=np.hypot(up-u,vp-v)
        im=axes[0,2].contourf(X,Y,err,levels=np.linspace(0,max(error_max,1e-12),31),cmap='turbo')
        fig.colorbar(im,ax=axes[0,2],label='Velocity vector error / Uwall')
        for ax,title in zip([axes[0,0],axes[0,2]],['Speed and streamlines','Full linear error range']):
            ax.set(xlabel='x/L',ylabel='y/L',aspect='equal',title=title)
        for ax,truth,pred,label in zip(axes[1],[u,v,T],[up,vp,Tp],['u / Uwall','v / Uwall','T / Twall']):
            reference=np.array([np.interp(.5,x,row)for row in truth])
            estimate=np.array([np.interp(.5,x,row)for row in pred])
            ax.plot(reference,y,color='black',label='mini DSMC')
            ax.plot(estimate,y,color='magenta',ls='--',label=name)
            ax.set(xlabel=label,ylabel='y/L',title='x/L = 0.5; native samples')
            ax.grid(alpha=.2)
        fig.legend(handles=[Patch(facecolor=plt.get_cmap('turbo')(.5),label='mini DSMC'),Line2D([],[],color='black',label='mini DSMC lines'),Line2D([],[],color='magenta',ls='--',label=name)],loc='outside lower center',ncol=3)
        fig.suptitle(f"Kn = {float(data['Kn'][index]):g}; Uwall = {Uw:g}; seed = {int(data['seed'][index])}; native {len(x)} × {len(y)} grid")
        figures.append(fig)
    return figures
