"""Presentation of the unchanged retained Week 7.4 lift example."""
from pathlib import Path

def plot_retained_lift_example(metrics, output=None):
    import matplotlib.pyplot as plt
    example=metrics['example']
    methods=('pretrained_encoder','random_encoder','pod32')
    labels=('pretrained encoder + ridge','random encoder + ridge','POD-32 coefficients + ridge')
    colors=('#c72b2b','#8053ac','#1273b5')
    styles=('--','-.',':')
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    ax.plot(example['time'],example['truth'],color='black',lw=1.8,label='LBM lift')
    for key,label,color,style in zip(methods,labels,colors,styles):
        ax.plot(example['time'],example[key],color=color,ls=style,lw=1.5,label=label)
    k=16 if metrics['protocol'].get('quick') else 32
    ax.set(xlabel='Lattice timestep',ylabel='Lift coefficient',title=f'Re = 115; {k} early labels; separated evaluation window')
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.86))
    if output is not None:
        target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(target,dpi=180,bbox_inches='tight')
    return fig
