"""Plot observed continuation loss, without attributing it to an older field."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=np.loadtxt(ROOT/'data/week13_deep_cavity/loss_continuation.dat')
assert d.shape==(74,3) and np.isfinite(d).all()
assert np.all(np.diff(d[:,0])>0) and np.all(d[:,1:]>0)
fig,ax=plt.subplots(figsize=(9,5.5))
for y,label,color in [(d[:,1],'Momentum x','#1764ab'),(d[:,2],'Momentum y','#d95f02'),(d[:,1:].sum(1),'Total','#26384a')]:
    ax.semilogy(d[:,0],y,label=label,color=color,lw=2)
ax.set(xlabel='Logged step in the resumed process',ylabel='Training mean-square residual',
       title='Deep-cavity PINN: retained continuation history')
ax.grid(alpha=.2,which='both')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
fig.text(.5,.02,'History ends at checkpoint 65711; supplied field is checkpoint 55118.\nNot a from-scratch history, CFD error, or independent test loss.',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.3)
out=ROOT/'results/week13_deep_cavity/loss_continuation.png'
fig.savefig(out,dpi=180,bbox_inches='tight')
print(out)
print('rows',len(d),'last total',d[-1,1:].sum())
